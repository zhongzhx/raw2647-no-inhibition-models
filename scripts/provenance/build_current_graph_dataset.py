#!/usr/bin/env python3
"""Build the formal 2D graph dataset from the audited current manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src" / "code" / "data_processing", ROOT / "src" / "code" / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from batch_process_smiles_multitask import process_dataframe, write_outputs  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    source = pd.read_csv(manifest_path)
    graphs, processed, failed, skipped, duplicates, conflicts = process_dataframe(
        df=source,
        strict=True,
        add_global_features=True,
        add_fingerprint=True,
        add_3d=False,
        explicit_hydrogens=False,
    )
    if failed or skipped or duplicates or conflicts:
        raise RuntimeError(
            "Audited manifest changed during graph construction: "
            f"failed={len(failed)}, skipped={len(skipped)}, "
            f"duplicates={len(duplicates)}, conflicts={len(conflicts)}"
        )
    graph_by_key = {
        str(row["inchikey"]): (graph, row)
        for graph, row in zip(graphs, processed)
    }
    if len(graph_by_key) != len(processed):
        raise RuntimeError("Graph construction produced duplicate InChIKeys")
    source_keys = source["inchikey"].astype(str).tolist()
    missing_keys = [key for key in source_keys if key not in graph_by_key]
    if missing_keys:
        raise RuntimeError(f"Graph construction lost {len(missing_keys)} audited structures")
    graphs = [graph_by_key[key][0] for key in source_keys]
    processed = [graph_by_key[key][1] for key in source_keys]
    review = pd.DataFrame(processed)
    if review["inchikey"].astype(str).tolist() != source_keys:
        raise RuntimeError("Graph construction changed the audited manifest order")
    summary = write_outputs(
        output_dir=output_dir,
        graph_list=graphs,
        processed_rows=processed,
        failed_smiles=failed,
        skipped_labels=skipped,
        duplicate_rows=duplicates,
        conflict_rows=conflicts,
        total_rows=len(source),
    )
    if summary["valid_graphs"] != 955:
        raise RuntimeError(f"Expected 955 graphs, found {summary['valid_graphs']}")
    if summary["active_label_1_count"] != 461:
        raise RuntimeError("Active-label count changed during graph construction")
    if summary["active_label_0_count"] != 494:
        raise RuntimeError("Inactive-label count changed during graph construction")
    provenance = {
        "schema_version": 1,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": sha256_file(manifest_path),
        "processed_graphs_sha256": sha256_file(output_dir / "processed_graphs.pt"),
        "processed_index_sha256": sha256_file(output_dir / "processed_index.csv"),
        **summary,
    }
    (output_dir / "graph_dataset_manifest.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
