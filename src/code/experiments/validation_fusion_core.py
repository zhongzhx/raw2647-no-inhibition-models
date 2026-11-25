"""Leakage-controlled A3/T1 probability fusion for locked validation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold


REQUIRED_COLUMNS = {
    "sample_index",
    "inchikey",
    "scaffold_group",
    "active_label",
    "probability_a3",
    "probability_t1",
}


@dataclass(frozen=True)
class FusionCrossFitResult:
    predictions: pd.DataFrame
    grid: pd.DataFrame
    fold_audit: pd.DataFrame
    full_application_alpha: float


def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Fusion frame is missing columns: {sorted(missing)}")
    checked = frame.copy().reset_index(drop=True)
    if checked["sample_index"].duplicated().any():
        raise ValueError("Fusion frame must contain one row per validation molecule")
    if checked["inchikey"].duplicated().any():
        raise ValueError("Fusion frame must be InChIKey-unique")
    if not checked["active_label"].isin([0, 1]).all():
        raise ValueError("Fusion labels must be binary")
    for column in ("probability_a3", "probability_t1"):
        values = pd.to_numeric(checked[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError(f"{column} must contain finite probabilities in [0, 1]")
        checked[column] = values
    return checked


def build_validation_folds(
    frame: pd.DataFrame,
    *,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """Lock scaffold-disjoint validation folds before fusion coefficients are fit."""

    checked = _validate_frame(frame)
    labels = checked["active_label"].to_numpy(dtype=int)
    groups = checked["scaffold_group"].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_id = np.full(len(checked), -1, dtype=int)
    for fold, (train_local, evaluation_local) in enumerate(
        splitter.split(np.zeros((len(checked), 1)), labels, groups)
    ):
        train_groups = set(groups[train_local])
        evaluation_groups = set(groups[evaluation_local])
        if train_groups & evaluation_groups:
            raise RuntimeError("Validation fusion fold has scaffold leakage")
        if len(np.unique(labels[train_local])) != 2:
            raise RuntimeError("Validation fusion selection partition lacks a class")
        if len(np.unique(labels[evaluation_local])) != 2:
            raise RuntimeError("Validation fusion evaluation partition lacks a class")
        fold_id[evaluation_local] = fold
    if np.any(fold_id < 0):
        raise RuntimeError("Validation fusion fold assignment is incomplete")
    output = checked[
        ["sample_index", "inchikey", "scaffold_group", "active_label"]
    ].copy()
    output["validation_fold"] = fold_id
    return output


def _validated_alpha_grid(values: Iterable[float]) -> np.ndarray:
    grid = np.asarray(list(values), dtype=float)
    if grid.ndim != 1 or len(grid) == 0:
        raise ValueError("Fusion alpha grid must be a non-empty vector")
    if not np.isfinite(grid).all() or np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError("Fusion alpha values must be finite and in [0, 1]")
    if len(np.unique(grid)) != len(grid):
        raise ValueError("Fusion alpha grid contains duplicates")
    return np.sort(grid)


def _grid_metrics(
    labels: np.ndarray,
    probability_a3: np.ndarray,
    probability_t1: np.ndarray,
    alpha_grid: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for alpha in alpha_grid:
        fused = alpha * probability_a3 + (1.0 - alpha) * probability_t1
        rows.append(
            {
                "alpha": float(alpha),
                "pr_auc": float(average_precision_score(labels, fused)),
                "brier": float(brier_score_loss(labels, fused)),
            }
        )
    return pd.DataFrame(rows)


def select_alpha(grid_metrics: pd.DataFrame) -> float:
    """Apply the locked PR-AUC, Brier, distance-to-0.5 ordering."""

    required = {"alpha", "pr_auc", "brier"}
    if required - set(grid_metrics.columns):
        raise ValueError("Fusion grid metrics are incomplete")
    ranked = grid_metrics.assign(
        distance_to_0_5=(grid_metrics["alpha"].astype(float) - 0.5).abs()
    ).sort_values(
        ["pr_auc", "brier", "distance_to_0_5", "alpha"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["alpha"])


def crossfit_validation_fusion(
    frame: pd.DataFrame,
    fold_membership: pd.DataFrame,
    *,
    alpha_grid: Iterable[float],
) -> FusionCrossFitResult:
    """Choose alpha on four validation folds and predict the untouched fifth fold."""

    checked = _validate_frame(frame)
    required_membership = {"sample_index", "inchikey", "validation_fold"}
    if required_membership - set(fold_membership.columns):
        raise ValueError("Validation fold membership is incomplete")
    membership = fold_membership[
        ["sample_index", "inchikey", "validation_fold"]
    ].copy()
    if membership["sample_index"].duplicated().any():
        raise ValueError("Validation fold membership contains duplicate samples")
    merged = checked.merge(
        membership,
        on=["sample_index", "inchikey"],
        how="left",
        validate="one_to_one",
    )
    if merged["validation_fold"].isna().any():
        raise ValueError("Validation fold membership does not cover all samples")
    merged["validation_fold"] = merged["validation_fold"].astype(int)
    grid = _validated_alpha_grid(alpha_grid)
    fused_probability = np.full(len(merged), np.nan, dtype=float)
    selected_alpha = np.full(len(merged), np.nan, dtype=float)
    grid_frames: list[pd.DataFrame] = []
    audit_rows: list[dict[str, int | float]] = []

    for evaluation_fold in sorted(merged["validation_fold"].unique()):
        evaluation_mask = merged["validation_fold"].eq(evaluation_fold).to_numpy()
        selection_mask = ~evaluation_mask
        selection_groups = set(
            merged.loc[selection_mask, "scaffold_group"].astype(str)
        )
        evaluation_groups = set(
            merged.loc[evaluation_mask, "scaffold_group"].astype(str)
        )
        overlap = selection_groups & evaluation_groups
        if overlap:
            raise RuntimeError("Validation fusion cross-fit has scaffold leakage")
        selection_grid = _grid_metrics(
            merged.loc[selection_mask, "active_label"].to_numpy(dtype=int),
            merged.loc[selection_mask, "probability_a3"].to_numpy(dtype=float),
            merged.loc[selection_mask, "probability_t1"].to_numpy(dtype=float),
            grid,
        )
        alpha = select_alpha(selection_grid)
        selection_grid.insert(0, "evaluation_fold", int(evaluation_fold))
        selection_grid["selected"] = selection_grid["alpha"].eq(alpha).astype(int)
        selection_grid["selection_sample_count"] = int(selection_mask.sum())
        selection_grid["evaluation_sample_count"] = int(evaluation_mask.sum())
        grid_frames.append(selection_grid)
        fused_probability[evaluation_mask] = (
            alpha
            * merged.loc[evaluation_mask, "probability_a3"].to_numpy(dtype=float)
            + (1.0 - alpha)
            * merged.loc[evaluation_mask, "probability_t1"].to_numpy(dtype=float)
        )
        selected_alpha[evaluation_mask] = alpha
        audit_rows.append(
            {
                "evaluation_fold": int(evaluation_fold),
                "selected_alpha": alpha,
                "selection_sample_count": int(selection_mask.sum()),
                "evaluation_sample_count": int(evaluation_mask.sum()),
                "selection_scaffold_count": len(selection_groups),
                "evaluation_scaffold_count": len(evaluation_groups),
                "scaffold_overlap_count": len(overlap),
            }
        )

    if not np.isfinite(fused_probability).all():
        raise RuntimeError("Validation fusion cross-fit predictions are incomplete")
    predictions = merged.copy()
    predictions["fusion_alpha"] = selected_alpha
    predictions["probability_f0"] = fused_probability
    full_grid = _grid_metrics(
        merged["active_label"].to_numpy(dtype=int),
        merged["probability_a3"].to_numpy(dtype=float),
        merged["probability_t1"].to_numpy(dtype=float),
        grid,
    )
    full_application_alpha = select_alpha(full_grid)
    full_grid.insert(0, "evaluation_fold", -1)
    full_grid["selected"] = (
        full_grid["alpha"].eq(full_application_alpha).astype(int)
    )
    full_grid["selection_sample_count"] = len(merged)
    full_grid["evaluation_sample_count"] = 0
    all_grid = pd.concat([*grid_frames, full_grid], ignore_index=True)
    return FusionCrossFitResult(
        predictions=predictions,
        grid=all_grid,
        fold_audit=pd.DataFrame(audit_rows),
        full_application_alpha=full_application_alpha,
    )
