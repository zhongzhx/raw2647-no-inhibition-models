#!/usr/bin/env python3
"""Complete missing scaffold-family metadata from SMILES without altering raw data.

The script writes a row-preserving CSV payload for the spreadsheet exporter and
an auditable summary. Existing annotations are never overwritten. Parseable
missing rows are assigned from the current modelling-set completion map, an
identical annotated Murcko scaffold, or the deterministic broad structural
rules used by the chemical-space analysis.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem

from generate_current_chemical_space import broad_structural_family, murcko_smiles


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "train data.xlsx"
FINAL_AUDIT = (
    ROOT
    / "outputs"
    / "rebuild_current_20260727"
    / "tables"
    / "chemical_space"
    / "scaffold_family_completion_audit.csv"
)
OUT_DIR = ROOT / "outputs" / "rebuild_current_20260727" / "data_audit"
PAYLOAD = OUT_DIR / "train_scaffold_family_completion_payload.csv"
SUMMARY = OUT_DIR / "train_scaffold_family_completion_summary.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cleaned_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def canonicalize(smiles: str) -> tuple[Chem.Mol | None, str]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, ""
    return molecule, Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )


def main() -> None:
    raw = pd.read_excel(RAW, sheet_name="train data")
    required = {"isomeric_smiles", "scaffold_family"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Raw workbook is missing columns: {sorted(missing)}")

    current = pd.read_csv(FINAL_AUDIT)
    exact_map = (
        current.drop_duplicates("canonical_smiles")
        .set_index("canonical_smiles")["scaffold_family_completed"]
        .astype(str)
        .to_dict()
    )
    scaffold_map: dict[str, str] = {}
    references: dict[tuple[str, str], str] = {}
    for scaffold, group in current.groupby("murcko_scaffold", dropna=False):
        if not scaffold or scaffold == "[ACYCLIC]":
            continue
        counts = Counter(group["scaffold_family_completed"].astype(str))
        family = sorted(counts, key=lambda value: (-counts[value], value))[0]
        scaffold_map[str(scaffold)] = family
        reference = group.loc[
            group["scaffold_family_completed"].astype(str).eq(family), "inchikey"
        ].iloc[0]
        references[(str(scaffold), family)] = str(reference)

    rows: list[dict[str, object]] = []
    for row_index, row in raw.reset_index(drop=True).iterrows():
        smiles = cleaned_text(row["isomeric_smiles"])
        original = cleaned_text(row["scaffold_family"])
        molecule, canonical = canonicalize(smiles)
        scaffold = murcko_smiles(molecule) if molecule is not None else ""
        family = original
        source = "existing_annotation" if original else ""
        reference = ""
        if original:
            reference = "same_row"
        elif molecule is None:
            source = "invalid_or_missing_smiles"
        elif canonical in exact_map:
            family = exact_map[canonical]
            source = "current_modeling_exact_structure"
        elif scaffold in scaffold_map:
            family = scaffold_map[scaffold]
            source = "identical_murcko_scaffold"
            reference = references.get((scaffold, family), "")
        else:
            family = broad_structural_family(molecule)
            source = "deterministic_structural_rule"

        rows.append(
            {
                "source_row_number": row_index + 2,
                "isomeric_smiles": smiles,
                "canonical_smiles": canonical,
                "murcko_scaffold": scaffold,
                "scaffold_family_original": original,
                "scaffold_family_completed": family,
                "scaffold_family_source": source,
                "scaffold_family_reference": reference,
                "smiles_parseable": int(molecule is not None),
            }
        )

    payload = pd.DataFrame(rows)
    if len(payload) != len(raw):
        raise RuntimeError("Row-preserving scaffold completion failed")
    parseable_missing = (
        payload["smiles_parseable"].eq(1)
        & payload["scaffold_family_original"].eq("")
        & payload["scaffold_family_completed"].eq("")
    )
    if parseable_missing.any():
        raise RuntimeError("A parseable SMILES was left without a scaffold family")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload.to_csv(PAYLOAD, index=False)
    summary = {
        "raw_workbook": str(RAW.relative_to(ROOT)),
        "raw_sha256": sha256(RAW),
        "row_count": int(len(payload)),
        "parseable_smiles": int(payload["smiles_parseable"].sum()),
        "invalid_or_missing_smiles": int(payload["smiles_parseable"].eq(0).sum()),
        "existing_annotations_preserved": int(
            payload["scaffold_family_original"].ne("").sum()
        ),
        "missing_before": int(payload["scaffold_family_original"].eq("").sum()),
        "filled_parseable_missing": int(
            (
                payload["scaffold_family_original"].eq("")
                & payload["scaffold_family_completed"].ne("")
            ).sum()
        ),
        "missing_after": int(payload["scaffold_family_completed"].eq("").sum()),
        "assignment_sources": {
            str(key): int(value)
            for key, value in payload["scaffold_family_source"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
        "payload_csv": str(PAYLOAD.relative_to(ROOT)),
        "raw_file_modified": False,
    }
    SUMMARY.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
