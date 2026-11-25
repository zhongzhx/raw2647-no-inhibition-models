#!/usr/bin/env python3
"""Audit, de-conflict, and deduplicate the current RAW264.7 workbook.

This script is the sole entry point for constructing the downstream modeling
manifest in the 2026-07-27 rebuild. It records one terminal disposition for
every source row. NS and legacy ND annotations receive an inactive label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rdkit import Chem


ROOT = Path(__file__).resolve().parents[2]
DATA_PROCESSING_DIR = ROOT / "src" / "code" / "data_processing"
UTILS_DIR = ROOT / "src" / "code" / "utils"
for module_dir in (DATA_PROCESSING_DIR, UTILS_DIR):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

from batch_process_smiles_multitask import (  # noqa: E402
    is_unreliable_label_confidence,
    normalize_text,
    parse_ic50_label,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonicalize_smiles(value: Any) -> tuple[str, str]:
    smiles = normalize_text(value)
    if not smiles:
        raise ValueError("empty_smiles")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("rdkit_parse_failed")
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    inchikey = Chem.MolToInchiKey(mol)
    if not inchikey:
        raise ValueError("inchikey_generation_failed")
    return canonical, inchikey


def is_ns_annotation(value: Any) -> bool:
    compact = normalize_text(value).lower().replace(".", "").replace(" ", "")
    return compact == "ns"


def parsed_record(
    row_index: int,
    row: pd.Series,
    canonical_smiles: str,
    inchikey: str,
) -> dict[str, Any]:
    if is_unreliable_label_confidence(row.get("label_confidence")):
        raise ValueError("unreliable_label_confidence")
    label = parse_ic50_label(
        raw_value=row.get("ic50_raw_uM"),
        numeric_value=row.get("ic50_uM_numeric"),
        pic50_value=row.get("pic50_from_ic50_numeric"),
    )
    threshold = float(label.censored_threshold_uM)
    is_censored = int(label.is_censored_gt50)
    classification_weight = (
        0.8 if is_censored and math.isfinite(threshold) and threshold < 50.0 else 1.0
    )
    return {
        **row.to_dict(),
        "original_row_index": int(row_index),
        "source_excel_row": int(row_index) + 2,
        "canonical_smiles": canonical_smiles,
        "inchikey": inchikey,
        "audit_active_label": int(label.active_label),
        "audit_has_classification_label": 1,
        "audit_classification_weight": float(classification_weight),
        "audit_pic50": float(label.pic50),
        "audit_has_exact_pic50": int(label.has_exact_pic50),
        "audit_censor_type": "right" if is_censored else "none",
        "audit_censored_threshold_uM": threshold,
    }


def conflict_reason(group: pd.DataFrame, pic50_spread_threshold: float) -> str:
    reasons: list[str] = []
    if group["audit_active_label"].nunique() > 1:
        reasons.append("classification_label_conflict")
    exact = group.loc[group["audit_has_exact_pic50"].eq(1), "audit_pic50"].dropna()
    if len(exact) > 1 and float(exact.max() - exact.min()) > pic50_spread_threshold:
        reasons.append("exact_pic50_spread_conflict")
    return ";".join(reasons)


def representative_rank(row: pd.Series) -> tuple[int, int, int]:
    confidence = normalize_text(row.get("label_confidence")).lower()
    confidence_rank = 1 if confidence == "high" else 0
    return (
        int(row["audit_has_exact_pic50"]),
        confidence_rank,
        -int(row["original_row_index"]),
    )


def audit_dataframe(
    source: pd.DataFrame,
    pic50_spread_threshold: float,
) -> dict[str, Any]:
    required = {"isomeric_smiles", "ic50_raw_uM"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise ValueError(f"missing_required_columns={missing}")

    ns_count = int(source["ic50_raw_uM"].map(is_ns_annotation).sum())
    valid_records: list[dict[str, Any]] = []
    invalid_smiles_records: list[dict[str, Any]] = []
    label_failure_records: list[dict[str, Any]] = []
    for row_index, row in source.iterrows():
        base = {
            **row.to_dict(),
            "original_row_index": int(row_index),
            "source_excel_row": int(row_index) + 2,
        }
        try:
            canonical_smiles, inchikey = canonicalize_smiles(row.get("isomeric_smiles"))
        except Exception as exc:
            invalid_smiles_records.append({**base, "audit_reason": str(exc)})
            continue
        try:
            valid_records.append(
                parsed_record(row_index, row, canonical_smiles, inchikey)
            )
        except Exception as exc:
            label_failure_records.append(
                {
                    **base,
                    "canonical_smiles": canonical_smiles,
                    "inchikey": inchikey,
                    "audit_reason": str(exc),
                }
            )

    valid = pd.DataFrame(valid_records)
    if valid.empty:
        raise RuntimeError("no_valid_rows_after_structure_and_label_parsing")

    conflict_parts: list[pd.DataFrame] = []
    classification_conflict_groups = 0
    regression_conflict_groups = 0
    for _, group in valid.groupby("canonical_smiles", sort=False):
        reason = conflict_reason(group, pic50_spread_threshold)
        if not reason:
            continue
        classification_conflict_groups += int(
            "classification_label_conflict" in reason
        )
        regression_conflict_groups += int(
            "exact_pic50_spread_conflict" in reason
        )
        flagged = group.copy()
        flagged["audit_reason"] = reason
        conflict_parts.append(flagged)
    conflicts = (
        pd.concat(conflict_parts, ignore_index=True)
        if conflict_parts
        else pd.DataFrame(columns=[*valid.columns, "audit_reason"])
    )

    conflict_keys = set(conflicts.get("canonical_smiles", []))
    nonconflicting = valid.loc[
        ~valid["canonical_smiles"].isin(conflict_keys)
    ].copy()

    kept_indices: list[int] = []
    duplicate_parts: list[pd.DataFrame] = []
    duplicate_group_count = 0
    for _, group in nonconflicting.groupby("canonical_smiles", sort=False):
        ranked = sorted(
            group.index.tolist(),
            key=lambda idx: representative_rank(group.loc[idx]),
            reverse=True,
        )
        kept_indices.append(ranked[0])
        if len(ranked) <= 1:
            continue
        duplicate_group_count += 1
        removed = group.loc[ranked[1:]].copy()
        removed["audit_reason"] = "duplicate_canonical_smiles"
        removed["kept_original_row_index"] = int(
            group.loc[ranked[0], "original_row_index"]
        )
        duplicate_parts.append(removed)

    cleaned = (
        nonconflicting.loc[kept_indices]
        .sort_values("original_row_index")
        .reset_index(drop=True)
    )
    duplicates = (
        pd.concat(duplicate_parts, ignore_index=True)
        if duplicate_parts
        else pd.DataFrame(
            columns=[*valid.columns, "audit_reason", "kept_original_row_index"]
        )
    )

    disposition = pd.DataFrame(
        {
            "original_row_index": source.index.astype(int),
            "source_excel_row": source.index.astype(int) + 2,
            "compound_name_or_label": source.get("compound_name_or_label"),
            "isomeric_smiles": source.get("isomeric_smiles"),
            "terminal_status": "kept",
            "terminal_reason": "eligible_unique_nonconflicting_structure",
            "kept_original_row_index": pd.NA,
        }
    ).set_index("original_row_index")

    def mark(records: pd.DataFrame, status: str) -> None:
        if records.empty:
            return
        for row in records.to_dict(orient="records"):
            idx = int(row["original_row_index"])
            disposition.loc[idx, "terminal_status"] = status
            disposition.loc[idx, "terminal_reason"] = row["audit_reason"]
            if "kept_original_row_index" in row and pd.notna(
                row["kept_original_row_index"]
            ):
                disposition.loc[idx, "kept_original_row_index"] = int(
                    row["kept_original_row_index"]
                )

    mark(pd.DataFrame(invalid_smiles_records), "excluded_invalid_smiles")
    mark(pd.DataFrame(label_failure_records), "excluded_label_parse_failure")
    mark(conflicts, "excluded_conflicting_duplicate_group")
    mark(duplicates, "excluded_duplicate")
    disposition = disposition.reset_index().sort_values("original_row_index")

    status_counts = {
        str(key): int(value)
        for key, value in disposition["terminal_status"].value_counts().items()
    }
    stats = {
        "raw_rows": int(len(source)),
        "ns_rows": ns_count,
        "nd_rows": int(
            source["ic50_raw_uM"]
            .astype(str)
            .str.strip()
            .str.lower()
            .str.replace(".", "", regex=False)
            .eq("nd")
            .sum()
        ),
        "valid_parseable_rows": int(len(valid)),
        "invalid_smiles_rows": int(len(invalid_smiles_records)),
        "label_parse_failed_rows": int(len(label_failure_records)),
        "unique_canonical_smiles_before_cleaning": int(
            valid["canonical_smiles"].nunique()
        ),
        "duplicate_groups_before_conflict_removal": int(
            (valid.groupby("canonical_smiles").size() > 1).sum()
        ),
        "conflict_groups_union": int(len(conflict_keys)),
        "classification_conflict_groups": int(classification_conflict_groups),
        "regression_conflict_groups_pic50_spread_gt_threshold": int(
            regression_conflict_groups
        ),
        "conflict_rows_excluded": int(len(conflicts)),
        "duplicate_groups_after_conflict_removal": int(duplicate_group_count),
        "duplicate_rows_excluded_after_conflict_removal": int(len(duplicates)),
        "clean_rows": int(len(cleaned)),
        "clean_known_active": int(cleaned["audit_active_label"].eq(1).sum()),
        "clean_known_inactive": int(cleaned["audit_active_label"].eq(0).sum()),
        "clean_exact_pic50": int(cleaned["audit_has_exact_pic50"].eq(1).sum()),
        "clean_censored": int(cleaned["audit_censor_type"].ne("none").sum()),
        "terminal_status_counts": status_counts,
        "regression_pic50_spread_conflict_threshold": float(
            pic50_spread_threshold
        ),
    }
    if sum(status_counts.values()) != len(source):
        raise AssertionError("terminal disposition does not reconcile to raw rows")
    if stats["clean_rows"] != status_counts.get("kept", 0):
        raise AssertionError("clean row count does not match kept dispositions")
    if cleaned["canonical_smiles"].duplicated().any():
        raise AssertionError("cleaned manifest still contains duplicate structures")

    return {
        "valid": valid,
        "invalid_smiles": pd.DataFrame(invalid_smiles_records),
        "label_failures": pd.DataFrame(label_failure_records),
        "conflicts": conflicts,
        "duplicates": duplicates,
        "cleaned": cleaned,
        "disposition": disposition,
        "stats": stats,
    }


def write_result(
    result: dict[str, Any],
    input_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = {
        "all_valid_parsed_rows.csv": result["valid"],
        "invalid_smiles_rows.csv": result["invalid_smiles"],
        "label_parse_failed_rows.csv": result["label_failures"],
        "conflicting_duplicate_rows.csv": result["conflicts"],
        "duplicate_removed_rows.csv": result["duplicates"],
        "final_modeling_manifest.csv": result["cleaned"],
        "sample_exclusion_log.csv": result["disposition"],
    }
    for filename, frame in exports.items():
        frame.to_csv(output_dir / filename, index=False)

    summary = {
        "schema_version": 1,
        "input_path": str(input_path.resolve()),
        "input_sha256": sha256_file(input_path),
        **result["stats"],
        "final_modeling_manifest_path": str(
            (output_dir / "final_modeling_manifest.csv").resolve()
        ),
        "sample_exclusion_log_path": str(
            (output_dir / "sample_exclusion_log.csv").resolve()
        ),
    }
    (output_dir / "data_flow_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pic50-spread-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    source = pd.read_excel(input_path)
    result = audit_dataframe(source, args.pic50_spread_threshold)
    write_result(result, input_path, Path(args.output_dir))


if __name__ == "__main__":
    main()
