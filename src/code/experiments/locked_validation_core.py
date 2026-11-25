"""Locked current-data development/validation split utilities.

This module is intentionally independent of model code.  It creates the
pre-registered 765/190 scaffold split and the metadata needed to keep the
190-molecule validation set outside all development training and scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


@dataclass(frozen=True)
class LockedSplit:
    development_indices: np.ndarray
    validation_indices: np.ndarray
    membership: pd.DataFrame
    audit: dict[str, Any]


def build_locked_split(
    review: pd.DataFrame,
    labels: np.ndarray,
    split_config: dict[str, Any],
) -> LockedSplit:
    """Build and strictly verify the pre-registered scaffold split."""

    required = {"inchikey", "scaffold_group", "active_label"}
    missing = sorted(required - set(review.columns))
    if missing:
        raise ValueError(f"Review manifest lacks locked-split columns: {missing}")
    labels = np.asarray(labels, dtype=int)
    if len(review) != len(labels):
        raise ValueError("Review and label lengths differ")
    if not np.array_equal(labels, review["active_label"].to_numpy(dtype=int)):
        raise ValueError("Review and label arrays disagree")
    if review["inchikey"].astype(str).nunique() != len(review):
        raise ValueError("Locked split requires unique InChIKeys")

    splitter = StratifiedGroupKFold(
        n_splits=int(split_config["n_splits"]),
        shuffle=bool(split_config["shuffle"]),
        random_state=int(split_config["random_state"]),
    )
    groups = review[str(split_config["group"])].astype(str).to_numpy()
    folds = list(splitter.split(np.zeros((len(review), 1)), labels, groups))
    locked_fold = int(split_config["locked_fold"])
    if locked_fold < 0 or locked_fold >= len(folds):
        raise ValueError("locked_fold is outside the generated folds")
    development, validation = (
        np.asarray(folds[locked_fold][0], dtype=int),
        np.asarray(folds[locked_fold][1], dtype=int),
    )

    expected = {
        "development_count": int(split_config["expected_development_count"]),
        "validation_count": int(split_config["expected_validation_count"]),
        "development_active": int(split_config["expected_development_active"]),
        "validation_active": int(split_config["expected_validation_active"]),
    }
    observed = {
        "development_count": int(len(development)),
        "validation_count": int(len(validation)),
        "development_active": int(labels[development].sum()),
        "validation_active": int(labels[validation].sum()),
    }
    if observed != expected:
        raise RuntimeError(
            f"Locked split no longer reproduces the pre-registered counts: "
            f"expected={expected}, observed={observed}"
        )

    development_inchikey = set(review.iloc[development]["inchikey"].astype(str))
    validation_inchikey = set(review.iloc[validation]["inchikey"].astype(str))
    development_scaffold = set(
        review.iloc[development]["scaffold_group"].astype(str)
    )
    validation_scaffold = set(review.iloc[validation]["scaffold_group"].astype(str))
    structure_overlap = development_inchikey & validation_inchikey
    scaffold_overlap = development_scaffold & validation_scaffold
    if structure_overlap or scaffold_overlap:
        raise RuntimeError("Locked development/validation split leaks structure groups")

    role = np.full(len(review), "locked_internal_validation", dtype=object)
    role[development] = "development"
    membership = review[
        ["inchikey", "canonical_smiles", "scaffold_group", "active_label"]
    ].copy()
    membership.insert(0, "sample_index", np.arange(len(review), dtype=int))
    membership.insert(1, "partition", role)
    membership["locked_fold"] = locked_fold

    audit = {
        "method": "StratifiedGroupKFold",
        "n_splits": int(split_config["n_splits"]),
        "shuffle": bool(split_config["shuffle"]),
        "random_state": int(split_config["random_state"]),
        "locked_fold": locked_fold,
        **observed,
        "development_inactive": int(len(development) - labels[development].sum()),
        "validation_inactive": int(len(validation) - labels[validation].sum()),
        "development_scaffold_count": int(len(development_scaffold)),
        "validation_scaffold_count": int(len(validation_scaffold)),
        "inchikey_overlap_count": int(len(structure_overlap)),
        "scaffold_overlap_count": int(len(scaffold_overlap)),
        "development_prevalence": float(labels[development].mean()),
        "validation_prevalence": float(labels[validation].mean()),
        "validation_access_status": "LOCKED_DO_NOT_SCORE_DURING_DEVELOPMENT",
    }
    return LockedSplit(development, validation, membership, audit)
