#!/usr/bin/env python3
"""Recompute F0 without using locked-validation labels for weight selection.

The single global coefficient is selected from strict development OOF
predictions, frozen, and then applied once to the existing locked-validation
predictions. The completed locked-run directory is treated as immutable; all
revision artifacts are written to a separate output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = (
    ROOT / "outputs" / "rebuild_current_20260727" / "downstream_locked_validation"
)
DEFAULT_OUTPUT_DIR = (
    ROOT / "outputs" / "rebuild_current_20260727" / "revision_f0_development_oof"
)
MODEL_A3 = "A3"
MODEL_T1 = "T1"
PRESELECTED_BEST = "T3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "current_rebuild_v3_locked_validation.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def expected_calibration_error(
    labels: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (
                probability <= edges[index + 1]
            )
        else:
            mask = (probability >= edges[index]) & (
                probability < edges[index + 1]
            )
        if mask.any():
            value += float(mask.mean()) * abs(
                float(labels[mask].mean()) - float(probability[mask].mean())
            )
    return float(value)


def classification_metrics(
    labels: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    order = np.argsort(-probability)
    prevalence = float(labels.mean())
    k10pct = max(1, int(math.ceil(0.10 * len(labels))))
    k20pct = max(1, int(math.ceil(0.20 * len(labels))))
    precision10pct = float(labels[order[:k10pct]].mean())
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(labels, probability)),
        "pr_auc": float(average_precision_score(labels, probability)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predicted)
        ),
        "brier": float(brier_score_loss(labels, probability)),
        "ece": expected_calibration_error(labels, probability, bins=10),
        "precision_at_10pct": precision10pct,
        "precision_at_20pct": float(labels[order[:k20pct]].mean()),
        "ef_at_10pct": precision10pct / max(prevalence, 1e-12),
        "active_at_10": float(labels[order[: min(10, len(labels))]].sum()),
        "active_at_20": float(labels[order[: min(20, len(labels))]].sum()),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def grid_metrics(frame: pd.DataFrame, alpha_grid: Iterable[float]) -> pd.DataFrame:
    labels = frame["active_label"].to_numpy(dtype=int)
    probability_a3 = frame[MODEL_A3].to_numpy(dtype=float)
    probability_t1 = frame[MODEL_T1].to_numpy(dtype=float)
    rows = []
    for alpha in alpha_grid:
        fused = alpha * probability_a3 + (1.0 - alpha) * probability_t1
        rows.append(
            {
                "alpha_a3": float(alpha),
                "weight_t1": float(1.0 - alpha),
                "pr_auc": float(average_precision_score(labels, fused)),
                "roc_auc": float(roc_auc_score(labels, fused)),
                "brier": float(brier_score_loss(labels, fused)),
            }
        )
    return pd.DataFrame(rows)


def select_alpha(grid: pd.DataFrame) -> float:
    ranked = grid.assign(
        distance_to_0_5=(grid["alpha_a3"].astype(float) - 0.5).abs()
    ).sort_values(
        ["pr_auc", "brier", "distance_to_0_5", "alpha_a3"],
        ascending=[False, True, True, True],
        kind="mergesort",
    )
    return float(ranked.iloc[0]["alpha_a3"])


def metric_value(metric: str, labels: np.ndarray, probability: np.ndarray) -> float:
    if metric == "pr_auc":
        return float(average_precision_score(labels, probability))
    if metric == "roc_auc":
        return float(roc_auc_score(labels, probability))
    if metric == "brier":
        return float(brier_score_loss(labels, probability))
    raise ValueError(metric)


def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    candidate_column: str,
    reference_column: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    checked = frame.reset_index(drop=True)
    groups = {
        key: values.index.to_numpy(dtype=int)
        for key, values in checked.groupby("scaffold_group", sort=True)
    }
    scaffold_keys = np.asarray(list(groups), dtype=object)
    labels = checked["active_label"].to_numpy(dtype=int)
    candidate = checked[candidate_column].to_numpy(dtype=float)
    reference = checked[reference_column].to_numpy(dtype=float)
    observed = metric_value(metric, labels, candidate) - metric_value(
        metric, labels, reference
    )
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(
            scaffold_keys, size=len(scaffold_keys), replace=True
        )
        sampled_indices = np.concatenate([groups[key] for key in sampled_groups])
        sampled_labels = labels[sampled_indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        differences.append(
            metric_value(metric, sampled_labels, candidate[sampled_indices])
            - metric_value(metric, sampled_labels, reference[sampled_indices])
        )
    values = np.asarray(differences, dtype=float)
    if len(values) < int(0.95 * repetitions):
        raise RuntimeError("Too many invalid scaffold-bootstrap replicates")
    low, high = np.quantile(values, [0.025, 0.975])
    p_value = min(
        1.0,
        2.0
        * min(
            (np.sum(values <= 0) + 1) / (len(values) + 1),
            (np.sum(values >= 0) + 1) / (len(values) + 1),
        ),
    )
    return {
        "difference_candidate_minus_reference": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_p_value_two_sided": float(p_value),
        "valid_bootstrap_repetitions": int(len(values)),
        "scaffold_cluster_count": int(len(scaffold_keys)),
    }


def absolute_cluster_bootstrap(
    frame: pd.DataFrame,
    probability_column: str,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, float | int]:
    checked = frame.reset_index(drop=True)
    groups = {
        key: values.index.to_numpy(dtype=int)
        for key, values in checked.groupby("scaffold_group", sort=True)
    }
    scaffold_keys = np.asarray(list(groups), dtype=object)
    labels = checked["active_label"].to_numpy(dtype=int)
    probability = checked[probability_column].to_numpy(dtype=float)
    observed = metric_value(metric, labels, probability)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(
            scaffold_keys, size=len(scaffold_keys), replace=True
        )
        sampled_indices = np.concatenate([groups[key] for key in sampled_groups])
        sampled_labels = labels[sampled_indices]
        if len(np.unique(sampled_labels)) < 2:
            continue
        values.append(
            metric_value(metric, sampled_labels, probability[sampled_indices])
        )
    values_array = np.asarray(values, dtype=float)
    if len(values_array) < int(0.95 * repetitions):
        raise RuntimeError("Too many invalid absolute bootstrap replicates")
    low, high = np.quantile(values_array, [0.025, 0.975])
    return {
        "estimate": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "valid_bootstrap_repetitions": int(len(values_array)),
        "scaffold_cluster_count": int(len(scaffold_keys)),
    }




def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    alpha_grid = [float(value) for value in config["fusion_policy"]["alpha_grid"]]
    repetitions = int(config["statistics"]["cluster_bootstrap_repetitions"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    threshold = float(config["training"]["classification_threshold"])

    development_path = run_dir / "final_ensemble_oof_predictions.csv"
    assignment_path = run_dir / "seed_averaged_oof_predictions.csv"
    validation_path = (
        run_dir
        / "locked_validation_evaluation"
        / "validation_seed_averaged_predictions.csv"
    )
    development_primary_path = (
        run_dir / "primary_comparisons_scaffold_cluster_bootstrap.csv"
    )
    disagreement_path = (
        run_dir.parent
        / "tables"
        / "interpretability"
        / "F12_disagreement_molecule_data.csv"
    )

    development_long = pd.read_csv(development_path)
    development_long = development_long.loc[
        development_long["model_id"].isin([MODEL_A3, MODEL_T1])
    ].copy()
    if len(development_long) != 2 * 765:
        raise RuntimeError("Development OOF predictions do not cover A3/T1 x 765")
    if not development_long["assignment_count"].eq(5).all():
        raise RuntimeError("Development ensemble does not average five assignments")
    identity = [
        "sample_index",
        "inchikey",
        "canonical_smiles",
        "scaffold_group",
        "active_label",
    ]
    development = development_long.pivot(
        index=identity, columns="model_id", values="probability"
    ).reset_index()
    if len(development) != 765 or development["inchikey"].nunique() != 765:
        raise RuntimeError("Development OOF fusion frame is not 765-molecule unique")

    grid = grid_metrics(development, alpha_grid)
    selected_alpha = select_alpha(grid)
    grid["selected"] = grid["alpha_a3"].eq(selected_alpha).astype(int)
    grid["selection_sample_count"] = len(development)
    grid["selection_source"] = "strict_development_OOF_assignment_ensemble"

    assignment_long = pd.read_csv(assignment_path)
    assignment_long = assignment_long.loc[
        assignment_long["model_id"].isin([MODEL_A3, MODEL_T1])
    ].copy()
    assignment_rows = []
    for assignment_seed, frame in assignment_long.groupby("assignment_seed"):
        assignment = frame.pivot(
            index=identity, columns="model_id", values="probability"
        ).reset_index()
        assignment_grid = grid_metrics(assignment, alpha_grid)
        alpha = select_alpha(assignment_grid)
        selected = assignment_grid.loc[
            assignment_grid["alpha_a3"].eq(alpha)
        ].iloc[0]
        assignment_rows.append(
            {
                "assignment_seed": int(assignment_seed),
                "selected_alpha_a3": float(alpha),
                "selected_weight_t1": float(1.0 - alpha),
                "selected_pr_auc": float(selected["pr_auc"]),
                "selected_roc_auc": float(selected["roc_auc"]),
                "selected_brier": float(selected["brier"]),
                "selection_sample_count": int(len(assignment)),
            }
        )
    assignment_audit = pd.DataFrame(assignment_rows).sort_values(
        "assignment_seed"
    )

    validation_long = pd.read_csv(validation_path)
    validation_long = validation_long.loc[
        validation_long["model_id"].isin([MODEL_A3, MODEL_T1, PRESELECTED_BEST])
    ].copy()
    if not validation_long["model_seed_count"].eq(3).all():
        raise RuntimeError("Locked predictions do not average three model seeds")
    validation = validation_long.pivot(
        index=identity, columns="model_id", values="probability"
    ).reset_index()
    if len(validation) != 190 or validation["inchikey"].nunique() != 190:
        raise RuntimeError("Locked-validation fusion frame is not 190-molecule unique")
    validation["F0"] = (
        selected_alpha * validation[MODEL_A3]
        + (1.0 - selected_alpha) * validation[MODEL_T1]
    )
    validation["fusion_alpha_a3"] = selected_alpha
    validation["fusion_weight_t1"] = 1.0 - selected_alpha
    validation["alpha_selection_data"] = "development_strict_OOF_only"
    if not np.allclose(
        validation["F0"].to_numpy(float),
        selected_alpha * validation[MODEL_A3].to_numpy(float)
        + (1.0 - selected_alpha) * validation[MODEL_T1].to_numpy(float),
        atol=0.0,
        rtol=0.0,
    ):
        raise RuntimeError("Frozen F0 probabilities failed the fusion identity")

    metric_rows = []
    for model_id in [MODEL_A3, MODEL_T1, PRESELECTED_BEST, "F0"]:
        metric_rows.append(
            {
                "model_id": model_id,
                **classification_metrics(
                    validation["active_label"].to_numpy(int),
                    validation[model_id].to_numpy(float),
                    threshold,
                ),
            }
        )
    validation_metrics = pd.DataFrame(metric_rows)

    comparison_rows = []
    absolute_rows = []
    for metric_index, metric in enumerate(("pr_auc", "roc_auc", "brier")):
        comparison_rows.append(
            {
                "comparison": "F0_DevelopmentOOFFrozen_vs_PreselectedBEST_T3",
                "candidate_model_id": "F0",
                "reference_model_id": PRESELECTED_BEST,
                "metric": metric,
                **cluster_bootstrap_difference(
                    validation,
                    "F0",
                    PRESELECTED_BEST,
                    metric,
                    repetitions,
                    bootstrap_seed + 6000 + metric_index,
                ),
            }
        )
        absolute_rows.append(
            {
                "model_id": "F0",
                "metric": metric,
                **absolute_cluster_bootstrap(
                    validation,
                    "F0",
                    metric,
                    repetitions,
                    # Match the original T1 CI seed because F0 collapses to T1.
                    bootstrap_seed + 5000 + 4 * 10 + metric_index,
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    absolute = pd.DataFrame(absolute_rows)

    development_primary = pd.read_csv(development_primary_path)
    primary_pr = development_primary.loc[
        development_primary["metric"].eq("pr_auc")
    ].copy()
    validation_pr = comparison.loc[comparison["metric"].eq("pr_auc")].copy()
    primary_comparisons = pd.concat([primary_pr, validation_pr], ignore_index=True)

    disagreement = pd.read_csv(disagreement_path)
    if len(disagreement) != 190 or disagreement["sample_index"].nunique() != 190:
        raise RuntimeError("Interpretability molecule table is not 190-molecule unique")
    revised_probability = validation[["sample_index", "F0"]].rename(
        columns={"F0": "revised_f0_probability"}
    )
    disagreement = disagreement.drop(columns=["F0"]).merge(
        revised_probability,
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    if disagreement["revised_f0_probability"].isna().any():
        raise RuntimeError("Revised F0 does not cover the interpretability table")
    disagreement = disagreement.rename(
        columns={"revised_f0_probability": "F0"}
    )
    disagreement["fusion_alpha"] = selected_alpha
    disagreement["f0_absolute_error"] = (
        disagreement["active_label"].astype(int)
        - disagreement["F0"].astype(float)
    ).abs()
    disagreement["f0_misclassified"] = (
        disagreement["F0"].ge(threshold).astype(int)
        != disagreement["active_label"].astype(int)
    ).astype(int)
    strata_rows = []
    for similarity_stratum, frame in disagreement.groupby(
        "similarity_stratum", sort=False
    ):
        strata_rows.append(
            {
                "similarity_stratum": similarity_stratum,
                "molecule_count": int(len(frame)),
                "scaffold_count": int(frame["scaffold_group"].nunique()),
                "similarity_min": float(
                    frame["nearest_development_morgan_tanimoto"].min()
                ),
                "similarity_median": float(
                    frame["nearest_development_morgan_tanimoto"].median()
                ),
                "similarity_max": float(
                    frame["nearest_development_morgan_tanimoto"].max()
                ),
                "best_mean_absolute_error": float(
                    frame["best_absolute_error"].mean()
                ),
                "f0_mean_absolute_error": float(
                    frame["f0_absolute_error"].mean()
                ),
                "best_misclassification_rate": float(
                    frame["best_misclassified"].mean()
                ),
                "f0_misclassification_rate": float(
                    frame["f0_misclassified"].mean()
                ),
            }
        )
    similarity_strata = pd.DataFrame(strata_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "development_fusion_grid.csv", grid)
    atomic_csv(
        output_dir / "development_assignment_alpha_audit.csv", assignment_audit
    )
    atomic_csv(
        output_dir / "locked_validation_predictions_frozen_alpha.csv", validation
    )
    atomic_csv(output_dir / "locked_validation_metrics.csv", validation_metrics)
    atomic_csv(
        output_dir / "locked_validation_absolute_performance_ci.csv", absolute
    )
    atomic_csv(
        output_dir / "f0_vs_preselected_best_bootstrap.csv", comparison
    )
    atomic_csv(output_dir / "primary_pr_auc_comparisons.csv", primary_comparisons)
    atomic_csv(
        output_dir / "interpretability_disagreement_molecule_data_revised_f0.csv",
        disagreement,
    )
    atomic_csv(
        output_dir / "interpretability_similarity_error_strata_revised_f0.csv",
        similarity_strata,
    )

    f0_metrics = validation_metrics.loc[
        validation_metrics["model_id"].eq("F0")
    ].iloc[0]
    f0_pr_comparison = comparison.loc[
        comparison["metric"].eq("pr_auc")
    ].iloc[0]
    manifest = {
        "status": "COMPLETE",
        "analysis": "F0 coefficient selected on development strict OOF only",
        "original_locked_run_preserved": True,
        "development_count": int(len(development)),
        "locked_validation_count": int(len(validation)),
        "alpha_definition": "p_F0 = alpha_A3 * p_A3 + (1-alpha_A3) * p_T1",
        "selected_alpha_a3": float(selected_alpha),
        "selected_weight_t1": float(1.0 - selected_alpha),
        "alpha_grid": alpha_grid,
        "selection_order": [
            "PR-AUC descending",
            "Brier ascending",
            "distance to 0.5 ascending",
            "alpha ascending",
        ],
        "all_five_assignment_selected_alphas": assignment_audit[
            "selected_alpha_a3"
        ].astype(float).tolist(),
        "locked_f0_metrics": {
            "pr_auc": float(f0_metrics["pr_auc"]),
            "roc_auc": float(f0_metrics["roc_auc"]),
            "brier": float(f0_metrics["brier"]),
        },
        "f0_vs_preselected_best_t3_pr_auc": {
            "difference": float(
                f0_pr_comparison["difference_candidate_minus_reference"]
            ),
            "ci95_low": float(f0_pr_comparison["ci95_low"]),
            "ci95_high": float(f0_pr_comparison["ci95_high"]),
            "raw_p_value": float(
                f0_pr_comparison["bootstrap_p_value_two_sided"]
            ),
        },
        "similarity_strata": similarity_strata.to_dict(orient="records"),
        "input_sha256": {
            str(development_path): sha256_file(development_path),
            str(assignment_path): sha256_file(assignment_path),
            str(validation_path): sha256_file(validation_path),
            str(development_primary_path): sha256_file(development_primary_path),
            str(disagreement_path): sha256_file(disagreement_path),
            str(config_path): sha256_file(config_path),
        },
    }
    atomic_json(output_dir / "revision_manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
