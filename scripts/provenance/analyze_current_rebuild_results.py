#!/usr/bin/env python3
"""Analyze current-data OOF results with scaffold-cluster bootstrap inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
SELECTABLE_MODEL_IDS = ("A0", "A1", "A2", "A3", "T1", "T2", "T3", "T4")
EXPECTED_DEVELOPMENT_MODEL_IDS = {
    "A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8",
    "T1", "T2", "T3", "T4", "BEST",
}
DEVELOPMENT_PRIMARY_COMPARISONS = (
    ("A1", "A0", "A1_Pretrained_GINE_vs_A0_Scratch_GINE"),
    ("A3", "A2", "A3_GatedFusion_vs_A2_DirectConcat"),
)
SECONDARY_ABLATION_COMPARISONS = (
    ("A4", "A3", "A4_GINE_Zero_vs_A3"),
    ("A5", "A3", "A5_RDKit_Zero_vs_A3"),
    ("A6", "A3", "A6_Morgan_Zero_vs_A3"),
    ("A7", "A3", "A7_MACCS_Zero_vs_A3"),
    ("A8", "A3", "A8_Family_Augmented_vs_A3"),
)
METRICS = ("pr_auc", "roc_auc", "brier")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def metric_value(metric: str, y: np.ndarray, p: np.ndarray) -> float:
    if metric == "pr_auc":
        return float(average_precision_score(y, p))
    if metric == "roc_auc":
        return float(roc_auc_score(y, p))
    if metric == "brier":
        return float(brier_score_loss(y, p))
    raise ValueError(metric)




def average_model_seeds(oof: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "assignment_seed",
        "model_id",
        "model",
        "sample_index",
        "inchikey",
        "canonical_smiles",
        "scaffold_group",
        "active_label",
    ]
    return (
        oof.groupby(keys, as_index=False)
        .agg(
            probability=("probability", "mean"),
            model_seed_count=("model_seed", "nunique"),
        )
        .sort_values(["assignment_seed", "model_id", "sample_index"])
    )


def assignment_metrics(seed_averaged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, frame in seed_averaged.groupby(
        ["assignment_seed", "model_id", "model"], sort=True
    ):
        y = frame["active_label"].to_numpy(int)
        p = frame["probability"].to_numpy(float)
        rows.append(
            {
                "assignment_seed": int(keys[0]),
                "model_id": keys[1],
                "model": keys[2],
                **{metric: metric_value(metric, y, p) for metric in METRICS},
            }
        )
    return pd.DataFrame(rows)


def ensemble_predictions(seed_averaged: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "model_id",
        "model",
        "sample_index",
        "inchikey",
        "canonical_smiles",
        "scaffold_group",
        "active_label",
    ]
    return (
        seed_averaged.groupby(keys, as_index=False)
        .agg(
            probability=("probability", "mean"),
            assignment_count=("assignment_seed", "nunique"),
        )
        .sort_values(["model_id", "sample_index"])
    )


def cluster_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    keys = ["sample_index", "active_label", "scaffold_group"]
    merged = candidate[keys + ["probability"]].merge(
        reference[keys + ["probability"]],
        on=keys,
        suffixes=("_candidate", "_reference"),
        validate="one_to_one",
    )
    if len(merged) != len(candidate) or len(merged) != len(reference):
        raise RuntimeError("Primary-comparison sample alignment failed")
    groups = {
        key: values.index.to_numpy(int)
        for key, values in merged.groupby("scaffold_group", sort=True)
    }
    scaffold_keys = np.asarray(list(groups), dtype=object)
    y = merged["active_label"].to_numpy(int)
    p_candidate = merged["probability_candidate"].to_numpy(float)
    p_reference = merged["probability_reference"].to_numpy(float)
    observed = metric_value(metric, y, p_candidate) - metric_value(
        metric, y, p_reference
    )
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(
            scaffold_keys, size=len(scaffold_keys), replace=True
        )
        sampled_indices = np.concatenate([groups[key] for key in sampled_groups])
        sampled_y = y[sampled_indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        differences.append(
            metric_value(metric, sampled_y, p_candidate[sampled_indices])
            - metric_value(metric, sampled_y, p_reference[sampled_indices])
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
        "difference_candidate_minus_reference": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "bootstrap_p_value_two_sided": float(p_value),
        "valid_bootstrap_repetitions": int(len(values)),
        "scaffold_cluster_count": int(len(scaffold_keys)),
    }


def absolute_cluster_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    groups = {
        key: values.index.to_numpy(int)
        for key, values in frame.reset_index(drop=True).groupby(
            "scaffold_group", sort=True
        )
    }
    scaffold_keys = np.asarray(list(groups), dtype=object)
    y = frame["active_label"].to_numpy(int)
    probability = frame["probability"].to_numpy(float)
    observed = metric_value(metric, y, probability)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        sampled_groups = rng.choice(
            scaffold_keys, size=len(scaffold_keys), replace=True
        )
        sampled_indices = np.concatenate([groups[key] for key in sampled_groups])
        sampled_y = y[sampled_indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        values.append(
            metric_value(metric, sampled_y, probability[sampled_indices])
        )
    values = np.asarray(values, dtype=float)
    if len(values) < int(0.95 * repetitions):
        raise RuntimeError("Too many invalid absolute bootstrap replicates")
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "estimate": observed,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "valid_bootstrap_repetitions": int(len(values)),
        "scaffold_cluster_count": int(len(scaffold_keys)),
    }


def assignment_direction_stability(
    by_assignment: pd.DataFrame,
    candidate_id: str,
    reference_id: str,
    metric: str,
    ensemble_difference: float,
) -> dict[str, float]:
    candidate = by_assignment.loc[
        by_assignment["model_id"].eq(candidate_id),
        ["assignment_seed", metric],
    ].rename(columns={metric: "candidate"})
    reference = by_assignment.loc[
        by_assignment["model_id"].eq(reference_id),
        ["assignment_seed", metric],
    ].rename(columns={metric: "reference"})
    merged = candidate.merge(
        reference, on="assignment_seed", validate="one_to_one"
    )
    differences = merged["candidate"] - merged["reference"]
    target_sign = np.sign(ensemble_difference)
    same_direction = np.sign(differences.to_numpy(float)) == target_sign
    return {
        "assignment_count": int(len(merged)),
        "assignment_same_direction_count": int(same_direction.sum()),
        "assignment_same_direction_fraction": float(same_direction.mean()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--config", default="configs/current_rebuild_v1.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    completion = json.loads((run_dir / "RUN_COMPLETION.json").read_text(encoding="utf-8"))
    if completion.get("status") != "FORMAL_OOF_COMPLETE":
        raise RuntimeError("Formal OOF run is not complete")
    oof_path = run_dir / "all_oof_predictions.csv"
    oof = pd.read_csv(oof_path)
    observed_model_ids = set(oof["model_id"].astype(str))
    if observed_model_ids != EXPECTED_DEVELOPMENT_MODEL_IDS:
        raise RuntimeError(
            "Development OOF model coverage differs from the locked A0-A8, "
            f"T1-T4 and BEST set: {sorted(observed_model_ids)}"
        )
    if oof["sample_index"].nunique() != 765:
        raise RuntimeError("Development analysis must contain exactly 765 molecules")
    seed_averaged = average_model_seeds(oof)
    expected_model_seeds = len(config["model_seeds"])
    if not seed_averaged["model_seed_count"].eq(expected_model_seeds).all():
        raise RuntimeError("Model-seed averaging is incomplete")
    by_assignment = assignment_metrics(seed_averaged)
    ensemble = ensemble_predictions(seed_averaged)
    expected_assignments = len(config["assignment_seeds"])
    if not ensemble["assignment_count"].eq(expected_assignments).all():
        raise RuntimeError("Assignment averaging is incomplete")

    summary_rows = []
    for (model_id, model), frame in by_assignment.groupby(
        ["model_id", "model"], sort=True
    ):
        row = {
            "model_id": model_id,
            "model": model,
            "assignment_count": int(frame["assignment_seed"].nunique()),
        }
        for metric in METRICS:
            row[f"{metric}_mean_across_assignments"] = float(frame[metric].mean())
            row[f"{metric}_sd_across_assignments_descriptive"] = float(
                frame[metric].std(ddof=1)
            )
        final_frame = ensemble.loc[ensemble["model_id"].eq(model_id)]
        final_y = final_frame["active_label"].to_numpy(int)
        final_probability = final_frame["probability"].to_numpy(float)
        for metric in METRICS:
            row[f"{metric}_final_ensemble"] = metric_value(
                metric, final_y, final_probability
            )
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    selectable = summary.loc[
        summary["model_id"].isin(SELECTABLE_MODEL_IDS)
    ].copy()
    ranking = selectable.sort_values(
        ["pr_auc_final_ensemble", "brier_final_ensemble"],
        ascending=[False, True],
    ).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    selected = ranking.iloc[0]

    bootstrap_rows = []
    repetitions = int(config["statistics"]["cluster_bootstrap_repetitions"])
    bootstrap_seed = int(config["statistics"]["bootstrap_seed"])
    for comparison_index, (candidate_id, reference_id, name) in enumerate(
        DEVELOPMENT_PRIMARY_COMPARISONS
    ):
        candidate = ensemble.loc[ensemble["model_id"].eq(candidate_id)]
        reference = ensemble.loc[ensemble["model_id"].eq(reference_id)]
        for metric_index, metric in enumerate(METRICS):
            result = cluster_bootstrap(
                candidate,
                reference,
                metric,
                repetitions,
                bootstrap_seed + comparison_index * 100 + metric_index,
            )
            bootstrap_rows.append(
                {
                    "comparison": name,
                    "candidate_model_id": candidate_id,
                    "reference_model_id": reference_id,
                    "metric": metric,
                    **result,
                    **assignment_direction_stability(
                        by_assignment,
                        candidate_id,
                        reference_id,
                        metric,
                        result["difference_candidate_minus_reference"],
                    ),
                }
            )
    comparisons = pd.DataFrame(bootstrap_rows)
    primary_pr = comparisons["metric"].eq("pr_auc")
    comparisons["primary_family"] = primary_pr.astype(int)

    secondary_rows = []
    for comparison_index, (candidate_id, reference_id, name) in enumerate(
        SECONDARY_ABLATION_COMPARISONS
    ):
        candidate = ensemble.loc[ensemble["model_id"].eq(candidate_id)]
        reference = ensemble.loc[ensemble["model_id"].eq(reference_id)]
        for metric_index, metric in enumerate(METRICS):
            result = cluster_bootstrap(
                candidate,
                reference,
                metric,
                repetitions,
                bootstrap_seed + 1000 + comparison_index * 100 + metric_index,
            )
            secondary_rows.append(
                {
                    "comparison": name,
                    "candidate_model_id": candidate_id,
                    "reference_model_id": reference_id,
                    "metric": metric,
                    **result,
                    **assignment_direction_stability(
                        by_assignment,
                        candidate_id,
                        reference_id,
                        metric,
                        result["difference_candidate_minus_reference"],
                    ),
                }
            )
    secondary = pd.DataFrame(secondary_rows)

    absolute_rows = []
    selected_frame = ensemble.loc[
        ensemble["model_id"].eq(selected["model_id"])
    ]
    for metric_index, metric in enumerate(METRICS):
        absolute_rows.append(
            {
                "model_id": selected["model_id"],
                "model": selected["model"],
                "metric": metric,
                **absolute_cluster_bootstrap(
                    selected_frame,
                    metric,
                    repetitions,
                    bootstrap_seed + 2000 + metric_index,
                ),
            }
        )
    absolute = pd.DataFrame(absolute_rows)

    seed_averaged.to_csv(run_dir / "seed_averaged_oof_predictions.csv", index=False)
    by_assignment.to_csv(run_dir / "seed_averaged_assignment_metrics.csv", index=False)
    ensemble.to_csv(run_dir / "final_ensemble_oof_predictions.csv", index=False)
    summary.to_csv(run_dir / "model_summary.csv", index=False)
    ranking.to_csv(run_dir / "model_ranking.csv", index=False)
    comparisons.to_csv(
        run_dir / "primary_comparisons_scaffold_cluster_bootstrap.csv",
        index=False,
    )
    secondary.to_csv(
        run_dir / "secondary_ablations_scaffold_cluster_bootstrap.csv",
        index=False,
    )
    absolute.to_csv(
        run_dir / "selected_model_absolute_performance_ci.csv",
        index=False,
    )
    selected_payload = {
        "status": "MODEL_SELECTED_FROM_LOCKED_765_DEVELOPMENT_OOF_ONLY",
        "selected_model_id": selected["model_id"],
        "selected_model": selected["model"],
        "selection_rule": (
            "highest molecule-level probability-ensemble scaffold-OOF PR-AUC "
            "across the predeclared assignments; Brier breaks exact ties"
        ),
        "pr_auc_final_ensemble": float(
            selected["pr_auc_final_ensemble"]
        ),
        "brier_final_ensemble": float(
            selected["brier_final_ensemble"]
        ),
        "selection_candidate_ids": list(SELECTABLE_MODEL_IDS),
        "a4_a8_excluded_from_selection": True,
        "locked_validation_results_accessed": False,
        "external_or_prior_project_data_used_for_selection": False,
        "analysis_git_commit": current_git_commit(),
        "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        "oof_predictions_sha256": sha256_file(oof_path),
        "model_ranking_sha256": sha256_file(run_dir / "model_ranking.csv"),
        "primary_comparisons_sha256": sha256_file(
            run_dir / "primary_comparisons_scaffold_cluster_bootstrap.csv"
        ),
    }
    (run_dir / "SELECTED_MODEL_LOCK.json").write_text(
        json.dumps(selected_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "STATISTICAL_ANALYSIS_COMPLETE",
        "model_seed_treatment": "probabilities averaged within assignment",
        "assignment_treatment": (
            "assignment results are descriptive; inferential resampling uses "
            "one ensemble probability per molecule and Bemis-Murcko scaffold clusters"
        ),
        "multiple_comparison_family": "three predeclared PR-AUC comparisons",
        "selected_model_id": selected["model_id"],
        "input_hashes": {
            "oof_predictions_sha256": sha256_file(oof_path),
            "config_sha256": sha256_file(config_path),
            "analysis_source_sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    (run_dir / "STATISTICAL_ANALYSIS_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selected_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
