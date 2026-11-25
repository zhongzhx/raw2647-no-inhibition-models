#!/usr/bin/env python3
"""Add an auditable one-hot scaffold-family feature to the current graph set."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT / "src" / "code", ROOT / "src" / "code" / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pipeline_utils import load_graphs  # noqa: E402


GRAPH_DIR = ROOT / "outputs" / "rebuild_current_20260727" / "graph_dataset"
AUDIT = (
    ROOT
    / "outputs"
    / "rebuild_current_20260727"
    / "tables"
    / "chemical_space"
    / "scaffold_family_completion_audit.csv"
)
OUT = ROOT / "outputs" / "rebuild_current_20260727" / "graph_dataset_family"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    graphs = load_graphs(GRAPH_DIR / "processed_graphs.pt")
    review = pd.read_csv(GRAPH_DIR / "processed_index.csv")
    audit = pd.read_csv(AUDIT)
    if len(graphs) != 955 or len(review) != 955 or len(audit) != 955:
        raise RuntimeError("Family graph dataset requires the current 955-molecule set")
    if review["inchikey"].tolist() != audit["inchikey"].tolist():
        audit = review[["inchikey"]].merge(
            audit, on="inchikey", how="left", validate="one_to_one"
        )
    if audit["scaffold_family_completed"].isna().any():
        raise RuntimeError("Scaffold-family audit contains missing values")

    families = sorted(audit["scaffold_family_completed"].astype(str).unique())
    family_to_index = {family: index for index, family in enumerate(families)}
    for graph, family in zip(
        graphs, audit["scaffold_family_completed"].astype(str), strict=True
    ):
        values = torch.zeros(len(families), dtype=torch.float32)
        values[family_to_index[family]] = 1.0
        graph.scaffold_family_fp = values.reshape(1, -1)
        graph.scaffold_family_index = torch.tensor(
            [family_to_index[family]], dtype=torch.long
        )
        graph.scaffold_family_name = family

    enriched = review.copy()
    enriched["scaffold_family_original"] = audit["scaffold_family_original"]
    enriched["scaffold_family_completed"] = audit["scaffold_family_completed"]
    enriched["scaffold_family_source"] = audit["scaffold_family_source"]
    enriched["murcko_scaffold_family_audit"] = audit["murcko_scaffold"]
    if enriched["scaffold_family_completed"].nunique() != len(families):
        raise RuntimeError("Family vocabulary and review table disagree")

    OUT.mkdir(parents=True, exist_ok=True)
    graph_path = OUT / "processed_graphs.pt"
    index_path = OUT / "processed_index.csv"
    vocab_path = OUT / "scaffold_family_vocabulary.json"
    summary_path = OUT / "dataset_summary.json"
    torch.save(graphs, graph_path)
    enriched.to_csv(index_path, index=False)
    vocab_path.write_text(
        json.dumps(
            {
                "encoding": "one_hot",
                "dimension": len(families),
                "families": families,
                "family_to_index": family_to_index,
                "label_independent_vocabulary": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    source_counts = (
        enriched["scaffold_family_source"].value_counts().sort_index().to_dict()
    )
    summary = {
        "graph_count": len(graphs),
        "family_dimension": len(families),
        "unique_families": len(families),
        "source_counts": {str(key): int(value) for key, value in source_counts.items()},
        "activity_label_used_for_assignment": False,
        "feature_scope": "A2/A3 hybrid models and traditional baselines only",
        "a0_a1_scope": "graph-only; unchanged for fair initialization ablation",
        "source_graph_sha256": sha256(GRAPH_DIR / "processed_graphs.pt"),
        "family_graph_sha256": sha256(graph_path),
        "family_audit_sha256": sha256(AUDIT),
        "one_hot_sums_all_one": bool(
            np.allclose(
                [
                    float(graph.scaffold_family_fp.sum().item())
                    for graph in graphs
                ],
                1.0,
            )
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
