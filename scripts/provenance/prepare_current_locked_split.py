#!/usr/bin/env python3
"""Create the pre-registered 765/190 current-data split and access lock."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "src" / "code" / "experiments",
    ROOT / "src" / "code" / "models",
    ROOT / "src" / "code" / "utils",
    ROOT / "src" / "code" / "data_processing",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from formal_transfer_ablation_core import add_group_metadata  # noqa: E402
from locked_validation_core import build_locked_split  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/current_rebuild_v3_locked_validation.json",
    )
    return parser.parse_args()


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def main() -> None:
    cli = parse_args()
    config_path = resolve(ROOT, cli.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    review_path = resolve(ROOT, config["review_csv"])
    output_root = resolve(ROOT, config["output_dir"]) / "split_lock"

    review = add_group_metadata(pd.read_csv(review_path))
    labels = review["active_label"].to_numpy(dtype=int)
    locked = build_locked_split(review, labels, config["locked_split"])

    atomic_csv(output_root / "current_split_membership.csv", locked.membership)
    atomic_csv(
        output_root / "development_index.csv",
        locked.membership.loc[
            locked.membership["partition"].eq("development")
        ].reset_index(drop=True),
    )
    atomic_csv(
        output_root / "locked_validation_index.csv",
        locked.membership.loc[
            locked.membership["partition"].eq("locked_internal_validation")
        ].reset_index(drop=True),
    )

    audit = {
        **locked.audit,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "review_path": str(review_path),
        "review_sha256": sha256_file(review_path),
        "membership_sha256": "",
    }
    membership_path = output_root / "current_split_membership.csv"
    audit["membership_sha256"] = sha256_file(membership_path)
    atomic_json(output_root / "locked_split_audit.json", audit)

    atomic_json(
        output_root / "VALIDATION_ACCESS_LOCK.json",
        {
            "status": "LOCKED_DO_NOT_SCORE_DURING_DEVELOPMENT",
            "validation_count": int(len(locked.validation_indices)),
            "allowed_before_development_lock": [
                "InChIKey overlap audit",
                "SMILES and Murcko scaffold isolation audit",
                "pre-registered class-count audit",
            ],
            "forbidden_before_development_lock": [
                "model performance",
                "model or epoch selection",
                "feature selection",
                "probability calibration",
                "fusion coefficient selection",
                "interpretability subgroup selection",
            ],
            "unlock_requires": [
                "approved pretraining checkpoint",
                "completed development OOF",
                "selected model lock",
                "fusion procedure lock",
                "RUN_LOCK.json",
            ],
            "membership_sha256": audit["membership_sha256"],
        },
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
