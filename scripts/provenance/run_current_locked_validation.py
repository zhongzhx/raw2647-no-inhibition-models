#!/usr/bin/env python3
"""Run the one-time locked 190-molecule validation after development lock.

The execution order is deliberately fail-closed:

1. verify the label-blind validation-unlock audit;
2. lock this script, inputs, model set, epoch-CV rule and fusion rule;
3. select neural epochs using only four-fold scaffold CV inside the 765
   development molecules;
4. refit the locked base models on all 765 development molecules;
5. predict the 190 locked molecules once;
6. calculate the pre-registered A3+T1 five-fold scaffold-cross-fitted fusion,
   metrics and scaffold-cluster confidence intervals.

Completed per-model/per-seed artifacts are resumable only when the validation
run lock is byte-for-byte identical. A completed validation cannot be rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts" / "provenance"
EXPERIMENTS = ROOT / "src" / "code" / "experiments"
for path in (SCRIPTS, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_current_rebuild_results as development_analysis  # noqa: E402
import run_current_rebuild_models as runner  # noqa: E402
from validation_fusion_core import (  # noqa: E402
    build_validation_folds,
    crossfit_validation_fusion,
)


NEURAL_MODEL_IDS = ("A0", "A1", "A2", "A3")
REQUIRED_BASE_MODEL_IDS = (*NEURAL_MODEL_IDS, "T1")
VALIDATION_EPOCH_CV_SEED = 20260727
VALIDATION_FUSION_FOLD_SEED = 20260727


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/current_rebuild_v3_locked_validation.json",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check locks and planned execution without reading validation labels.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only complete same-lock cached model/seed tasks.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=True,
    )


def planned_lock(
    config_path: Path,
    config: dict[str, Any],
    run_dir: Path,
    validation_dir: Path,
) -> dict[str, Any]:
    selected = json.loads(
        (run_dir / "SELECTED_MODEL_LOCK.json").read_text(encoding="utf-8")
    )
    unlock_path = run_dir / "VALIDATION_UNLOCK_AUDIT.json"
    unlock = json.loads(unlock_path.read_text(encoding="utf-8"))
    if (
        unlock.get("status") != "READY_FOR_ONE_TIME_LOCKED_VALIDATION"
        or unlock.get("all_required_passed") is not True
        or unlock.get("failed_required_gates")
        or unlock.get("validation_results_accessed") is not False
    ):
        raise RuntimeError("Validation unlock audit is not ready and fail-closed")
    selected_id = str(selected.get("selected_model_id"))
    trained_ids = list(REQUIRED_BASE_MODEL_IDS)
    if selected_id not in trained_ids:
        trained_ids.append(selected_id)
    if selected_id not in runner.BEST_CANDIDATE_IDS:
        raise RuntimeError("Selected model is outside the locked BEST candidates")
    fusion = config["fusion_policy"]
    expected_grid = [round(index * 0.05, 2) for index in range(21)]
    if (
        fusion.get("base_models") != ["A3", "T1"]
        or fusion.get("alpha_grid") != expected_grid
        or fusion.get("validation_crossfit_folds") != 5
        or fusion.get("validation_crossfit_group") != "scaffold_group"
    ):
        raise RuntimeError("Validation fusion procedure differs from the lock")
    source_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/run_current_rebuild_models.py",
        ROOT / "scripts/analyze_current_rebuild_results.py",
        ROOT / "src/code/experiments/formal_transfer_ablation_core.py",
        ROOT / "src/code/experiments/locked_validation_core.py",
        ROOT / "src/code/experiments/validation_fusion_core.py",
        ROOT / "src/code/models/gine_v2.py",
    ]
    return {
        "status": "LOCKED_BEFORE_ONE_TIME_VALIDATION",
        "protocol_name": config["protocol_name"],
        "development_run_dir": str(run_dir),
        "validation_output_dir": str(validation_dir),
        "selected_model_id": selected_id,
        "base_model_ids": trained_ids,
        "neural_model_ids": list(NEURAL_MODEL_IDS),
        "model_seeds": list(map(int, config["model_seeds"])),
        "development_count": 765,
        "validation_count": 190,
        "development_full_epoch_cv": {
            "folds": 4,
            "group": "scaffold_group",
            "random_state": VALIDATION_EPOCH_CV_SEED,
            "selection_metric": "PR-AUC",
            "final_epoch_rule": "rounded median of four fold-selected epochs per model seed",
        },
        "fusion_policy": fusion,
        "fusion_fold_random_state": VALIDATION_FUSION_FOLD_SEED,
        "classification_threshold": float(
            config["training"]["classification_threshold"]
        ),
        "config_sha256": sha256_file(config_path),
        "development_run_lock_sha256": sha256_file(run_dir / "RUN_LOCK.json"),
        "development_completion_sha256": sha256_file(
            run_dir / "RUN_COMPLETION.json"
        ),
        "selected_model_lock_sha256": sha256_file(
            run_dir / "SELECTED_MODEL_LOCK.json"
        ),
        "statistical_analysis_manifest_sha256": sha256_file(
            run_dir / "STATISTICAL_ANALYSIS_MANIFEST.json"
        ),
        "validation_unlock_audit_sha256": sha256_file(unlock_path),
        "git_commit": runner.current_git_commit(),
        "source_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in source_paths
        },
        "prior_project_artifacts_used": False,
    }


def check_or_create_validation_lock(
    lock_path: Path,
    payload: dict[str, Any],
    *,
    resume: bool,
) -> None:
    completion = lock_path.parent / "VALIDATION_COMPLETION.json"
    if completion.exists():
        raise RuntimeError(
            "Locked validation is already complete and cannot be executed again"
        )
    if lock_path.exists():
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if canonical_json(existing) != canonical_json(payload):
            raise RuntimeError("Existing validation run lock differs from current plan")
        if not resume:
            raise RuntimeError(
                "Partial validation artifacts exist; use --resume after lock audit"
            )
        return
    if resume:
        raise RuntimeError("--resume requested but no validation run lock exists")
    atomic_json(lock_path, payload)


def development_epoch_splits(
    labels: np.ndarray,
    review: pd.DataFrame,
    development_indices: np.ndarray,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    development_indices = np.asarray(development_indices, dtype=int)
    local = runner.grouped_folds(
        labels[development_indices],
        review.iloc[development_indices].reset_index(drop=True),
        "scaffold_group",
        VALIDATION_EPOCH_CV_SEED,
        4,
    )
    splits = tuple(
        (
            development_indices[np.asarray(train_local, dtype=int)],
            development_indices[np.asarray(valid_local, dtype=int)],
        )
        for train_local, valid_local in local
    )
    covered = np.concatenate([valid for _, valid in splits])
    if len(covered) != len(development_indices) or set(map(int, covered)) != set(
        map(int, development_indices)
    ):
        raise RuntimeError("Full-development epoch CV coverage is incomplete")
    return splits


def epoch_cv_membership(
    review: pd.DataFrame,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
) -> pd.DataFrame:
    rows = []
    for fold, (train_indices, validation_indices) in enumerate(splits):
        train_scaffolds = set(
            review.iloc[train_indices]["scaffold_group"].astype(str)
        )
        validation_scaffolds = set(
            review.iloc[validation_indices]["scaffold_group"].astype(str)
        )
        if train_scaffolds & validation_scaffolds:
            raise RuntimeError("Full-development epoch CV has scaffold leakage")
        for role, indices in (
            ("epoch_inner_training", train_indices),
            ("epoch_inner_validation", validation_indices),
        ):
            for index in indices:
                row = review.iloc[int(index)]
                rows.append(
                    {
                        "fold": fold,
                        "role": role,
                        "sample_index": int(index),
                        "inchikey": row["inchikey"],
                        "scaffold_group": row["scaffold_group"],
                        "active_label": int(row["active_label"]),
                    }
                )
    return pd.DataFrame(rows)


def save_state_dict(path: Path, model: torch.nn.Module) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    torch.save(state, temporary)
    os.replace(temporary, path)


def train_validation_neural_seed(
    model_id: str,
    model_seed: int,
    spec: Any,
    graphs: list,
    review: pd.DataFrame,
    splits: tuple[tuple[np.ndarray, np.ndarray], ...],
    development_indices: np.ndarray,
    validation_indices: np.ndarray,
    checkpoint: Path,
    args: Any,
    task_dir: Path,
    *,
    resume: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_path = task_dir / "predictions.csv"
    audit_path = task_dir / "audit.json"
    checkpoint_path = task_dir / "model_state_dict.pt"
    if (
        resume
        and prediction_path.exists()
        and audit_path.exists()
        and checkpoint_path.exists()
    ):
        return (
            pd.read_csv(prediction_path),
            json.loads(audit_path.read_text(encoding="utf-8")),
        )
    task_dir.mkdir(parents=True, exist_ok=True)
    selected_epochs: list[int] = []
    fold_audits: list[dict[str, Any]] = []
    history_frames: list[pd.DataFrame] = []
    for fold, (train_indices, validation_cv_indices) in enumerate(splits):
        fold_dir = task_dir / f"epoch_cv_fold_{fold}"
        fold_audit_path = fold_dir / "audit.json"
        fold_history_path = fold_dir / "history.csv"
        if resume and fold_audit_path.exists() and fold_history_path.exists():
            fold_audit = json.loads(fold_audit_path.read_text(encoding="utf-8"))
            selected_epoch = int(fold_audit["selected_epoch"])
            history = pd.read_csv(fold_history_path)
        else:
            shared_seed = runner.stable_seed(
                "locked-validation-neural-inner",
                model_seed,
                fold,
            )
            model, fold_graphs, raw_history, raw_audit, selected_epoch = (
                runner.train_neural(
                    spec,
                    graphs,
                    train_indices,
                    checkpoint,
                    args,
                    shared_seed,
                    args.epochs,
                    valid_idx=validation_cv_indices,
                )
            )
            history = pd.DataFrame(raw_history)
            fold_audit = {
                "model_id": model_id,
                "model_seed": model_seed,
                "fold": fold,
                "selected_epoch": int(selected_epoch),
                "training_count": int(len(train_indices)),
                "validation_count": int(len(validation_cv_indices)),
                "scaffold_overlap_count": int(
                    len(
                        set(review.iloc[train_indices]["scaffold_group"].astype(str))
                        & set(
                            review.iloc[validation_cv_indices][
                                "scaffold_group"
                            ].astype(str)
                        )
                    )
                ),
                "training_audit": runner.clean_audit(raw_audit),
            }
            atomic_csv(fold_history_path, history)
            atomic_json(fold_audit_path, fold_audit)
            del model, fold_graphs
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        if int(fold_audit.get("scaffold_overlap_count", -1)) != 0:
            raise RuntimeError("Neural epoch CV scaffold overlap detected")
        selected_epochs.append(int(selected_epoch))
        fold_audits.append(fold_audit)
        block = history.copy()
        block.insert(0, "fold", fold)
        history_frames.append(block)
    selected_epoch = int(round(float(np.median(selected_epochs))))
    final_seed = runner.stable_seed(
        "locked-validation-neural-final",
        model_seed,
    )
    model, fitted_graphs, final_history, final_audit, _ = runner.train_neural(
        spec,
        graphs,
        development_indices,
        checkpoint,
        args,
        final_seed,
        selected_epoch,
        valid_idx=None,
    )
    probability = runner.predict_model(
        model,
        fitted_graphs,
        validation_indices,
        runner.resolve_device(args.device),
        args.batch_size,
    )
    rows = review.iloc[validation_indices][
        [
            "sample_index",
            "inchikey",
            "canonical_smiles",
            "scaffold_group",
            "active_label",
        ]
    ].copy()
    rows.insert(0, "model_seed", model_seed)
    rows.insert(0, "model", runner.MODEL_NAMES[model_id])
    rows.insert(0, "model_id", model_id)
    rows["probability"] = probability
    audit = {
        "model_id": model_id,
        "model": runner.MODEL_NAMES[model_id],
        "model_seed": model_seed,
        "epoch_cv_fold_count": 4,
        "epoch_cv_selected_epochs": selected_epochs,
        "selected_final_epoch": selected_epoch,
        "epoch_cv_folds": fold_audits,
        "final_training_count": int(len(development_indices)),
        "prediction_count": int(len(validation_indices)),
        "validation_labels_used_for_training_or_epoch_selection": False,
        "final_training_audit": runner.clean_audit(final_audit),
    }
    atomic_csv(
        task_dir / "epoch_cv_training_history.csv",
        pd.concat(history_frames, ignore_index=True),
    )
    atomic_csv(task_dir / "final_training_history.csv", pd.DataFrame(final_history))
    save_state_dict(checkpoint_path, model)
    atomic_csv(prediction_path, rows)
    atomic_json(audit_path, audit)
    del model, fitted_graphs
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return rows, audit


def train_validation_traditional_seed(
    model_id: str,
    model_seed: int,
    matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
    review: pd.DataFrame,
    development_indices: np.ndarray,
    validation_indices: np.ndarray,
    config: dict[str, Any],
    task_dir: Path,
    *,
    resume: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_path = task_dir / "predictions.csv"
    audit_path = task_dir / "audit.json"
    model_path = task_dir / "model.joblib"
    if resume and prediction_path.exists() and audit_path.exists() and model_path.exists():
        return (
            pd.read_csv(prediction_path),
            json.loads(audit_path.read_text(encoding="utf-8")),
        )
    task_dir.mkdir(parents=True, exist_ok=True)
    matrix_name = "flat" if model_id == "T1" else "rdkit"
    matrix = matrices[matrix_name]
    seed = runner.stable_seed("locked-validation-traditional", model_id, model_seed)
    started = time.time()
    model = runner.fit_traditional_model(
        model_id,
        matrix,
        labels,
        weights,
        development_indices,
        config,
        seed,
    )
    probability = runner.traditional_scores(
        model_id, model, matrix[validation_indices]
    )
    rows = review.iloc[validation_indices][
        [
            "sample_index",
            "inchikey",
            "canonical_smiles",
            "scaffold_group",
            "active_label",
        ]
    ].copy()
    rows.insert(0, "model_seed", model_seed)
    rows.insert(0, "model", runner.MODEL_NAMES[model_id])
    rows.insert(0, "model_id", model_id)
    rows["probability"] = probability
    audit = {
        "model_id": model_id,
        "model": runner.MODEL_NAMES[model_id],
        "model_seed": model_seed,
        "feature_matrix": matrix_name,
        "training_count": int(len(development_indices)),
        "prediction_count": int(len(validation_indices)),
        "validation_labels_used_for_training": False,
        "elapsed_seconds": time.time() - started,
    }
    temporary = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump(model, temporary)
    os.replace(temporary, model_path)
    atomic_csv(prediction_path, rows)
    atomic_json(audit_path, audit)
    return rows, audit


def expanded_metrics(
    labels: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    result = runner.classification_metrics(labels, probability, threshold)
    predicted = (np.asarray(probability, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        np.asarray(labels, dtype=int), predicted, labels=[0, 1]
    ).ravel()
    result.update(
        {
            "precision": float(
                precision_score(labels, predicted, zero_division=0)
            ),
            "recall": float(recall_score(labels, predicted, zero_division=0)),
            "specificity": float(tn / max(tn + fp, 1)),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )
    return result


def assemble_validation_results(
    predictions: list[pd.DataFrame],
    selected_model_id: str,
    config: dict[str, Any],
    validation_dir: Path,
) -> dict[str, Any]:
    long = pd.concat(predictions, ignore_index=True)
    identity = [
        "sample_index",
        "inchikey",
        "canonical_smiles",
        "scaffold_group",
        "active_label",
    ]
    seed_averaged = (
        long.groupby([*identity, "model_id", "model"], as_index=False)
        .agg(
            probability=("probability", "mean"),
            model_seed_count=("model_seed", "nunique"),
        )
        .sort_values(["model_id", "sample_index"])
    )
    if not seed_averaged["model_seed_count"].eq(
        len(config["model_seeds"])
    ).all():
        raise RuntimeError("Validation model-seed averaging is incomplete")
    pivot = seed_averaged.pivot(
        index=identity,
        columns="model_id",
        values="probability",
    ).reset_index()
    required = set(REQUIRED_BASE_MODEL_IDS) | {selected_model_id}
    if required - set(pivot.columns):
        raise RuntimeError("Validation base predictions are incomplete")
    fusion_frame = pivot[
        ["sample_index", "inchikey", "scaffold_group", "active_label"]
    ].copy()
    fusion_frame["probability_a3"] = pivot["A3"].to_numpy(float)
    fusion_frame["probability_t1"] = pivot["T1"].to_numpy(float)
    fold_membership = build_validation_folds(
        fusion_frame,
        n_splits=5,
        random_state=VALIDATION_FUSION_FOLD_SEED,
    )
    fusion = crossfit_validation_fusion(
        fusion_frame,
        fold_membership,
        alpha_grid=config["fusion_policy"]["alpha_grid"],
    )
    pivot["BEST"] = pivot[selected_model_id].to_numpy(float)
    fusion_probability = fusion.predictions[
        ["sample_index", "probability_f0", "fusion_alpha", "validation_fold"]
    ]
    pivot = pivot.merge(
        fusion_probability,
        on="sample_index",
        how="left",
        validate="one_to_one",
    )
    pivot = pivot.rename(columns={"probability_f0": "F0"})
    if len(pivot) != 190 or pivot["inchikey"].nunique() != 190:
        raise RuntimeError("Locked validation prediction coverage is incomplete")
    model_order = [*REQUIRED_BASE_MODEL_IDS]
    if selected_model_id not in model_order:
        model_order.append(selected_model_id)
    model_order.extend(["BEST", "F0"])
    metric_rows = []
    ci_rows = []
    threshold = float(config["training"]["classification_threshold"])
    for model_id in model_order:
        frame = pivot[
            ["sample_index", "active_label", "scaffold_group", model_id]
        ].rename(columns={model_id: "probability"})
        metric_rows.append(
            {
                "model_id": model_id,
                **expanded_metrics(
                    frame["active_label"].to_numpy(int),
                    frame["probability"].to_numpy(float),
                    threshold,
                ),
            }
        )
        for metric_index, metric in enumerate(("pr_auc", "roc_auc", "brier")):
            ci_rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    **development_analysis.absolute_cluster_bootstrap(
                        frame,
                        metric,
                        int(config["statistics"]["cluster_bootstrap_repetitions"]),
                        int(config["statistics"]["bootstrap_seed"])
                        + 5000
                        + model_order.index(model_id) * 10
                        + metric_index,
                    ),
                }
            )
    candidate = pivot[
        ["sample_index", "active_label", "scaffold_group", "F0"]
    ].rename(columns={"F0": "probability"})
    reference = pivot[
        ["sample_index", "active_label", "scaffold_group", "BEST"]
    ].rename(columns={"BEST": "probability"})
    comparison_rows = []
    for metric_index, metric in enumerate(("pr_auc", "roc_auc", "brier")):
        comparison_rows.append(
            {
                "comparison": "F0_ValidationCrossFitFusion_vs_InnerSelectedBestSingle",
                "candidate_model_id": "F0",
                "reference_model_id": "BEST",
                "metric": metric,
                **development_analysis.cluster_bootstrap(
                    candidate,
                    reference,
                    metric,
                    int(config["statistics"]["cluster_bootstrap_repetitions"]),
                    int(config["statistics"]["bootstrap_seed"])
                    + 6000
                    + metric_index,
                ),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    development_primary = pd.read_csv(
        validation_dir.parent
        / "primary_comparisons_scaffold_cluster_bootstrap.csv"
    )
    primary_pr = development_primary.loc[
        development_primary["metric"].eq("pr_auc")
    ].copy()
    validation_pr = comparison.loc[comparison["metric"].eq("pr_auc")].copy()
    complete_family = pd.concat([primary_pr, validation_pr], ignore_index=True)
    atomic_csv(validation_dir / "validation_seed_predictions.csv", long)
    atomic_csv(
        validation_dir / "validation_seed_averaged_predictions.csv",
        seed_averaged,
    )
    atomic_csv(validation_dir / "validation_predictions.csv", pivot)
    atomic_csv(
        validation_dir / "validation_fusion_fold_membership.csv",
        fold_membership,
    )
    atomic_csv(
        validation_dir / "validation_fusion_crossfit.csv",
        fusion.predictions,
    )
    atomic_csv(validation_dir / "validation_fusion_grid.csv", fusion.grid)
    atomic_csv(
        validation_dir / "validation_fusion_fold_audit.csv",
        fusion.fold_audit,
    )
    atomic_csv(
        validation_dir / "validation_metrics.csv", pd.DataFrame(metric_rows)
    )
    atomic_csv(
        validation_dir / "validation_absolute_performance_ci.csv",
        pd.DataFrame(ci_rows),
    )
    atomic_csv(
        validation_dir / "validation_f0_vs_best_bootstrap.csv",
        comparison,
    )
    atomic_csv(
        validation_dir / "primary_pr_auc_comparisons.csv",
        complete_family,
    )
    return {
        "selected_model_id": selected_model_id,
        "base_model_ids": model_order,
        "validation_count": 190,
        "model_seed_count": len(config["model_seeds"]),
        "fusion_crossfit_folds": 5,
        "fusion_fold_selected_alphas": fusion.fold_audit[
            "selected_alpha"
        ].astype(float).tolist(),
        "fusion_alpha_median": float(
            fusion.fold_audit["selected_alpha"].median()
        ),
        "fusion_alpha_min": float(fusion.fold_audit["selected_alpha"].min()),
        "fusion_alpha_max": float(fusion.fold_audit["selected_alpha"].max()),
        "full_application_alpha": float(fusion.full_application_alpha),
    }


def main() -> None:
    cli = parse_args()
    config, paths = runner.resolve_config(cli.config)
    config_path = paths["config"]
    run_dir = paths["output_dir"]
    validation_dir = run_dir / "locked_validation_evaluation"
    plan = planned_lock(config_path, config, run_dir, validation_dir)
    if cli.preflight:
        if validation_dir.exists() and any(validation_dir.iterdir()):
            raise RuntimeError(
                "Validation output already contains artifacts; preflight is not clean"
            )
        print(
            json.dumps(
                {
                    "status": "ONE_TIME_VALIDATION_PREFLIGHT_PASS",
                    "validation_results_accessed": False,
                    "plan": plan,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    validation_dir.mkdir(parents=True, exist_ok=True)
    check_or_create_validation_lock(
        validation_dir / "VALIDATION_RUN_LOCK.json",
        plan,
        resume=cli.resume,
    )
    graphs, review, labels = runner.prepare_data(paths)
    locked = runner.build_locked_split(review, labels, config["locked_split"])
    if (
        len(locked.development_indices) != 765
        or len(locked.validation_indices) != 190
        or locked.audit.get("inchikey_overlap_count") != 0
        or locked.audit.get("scaffold_overlap_count") != 0
    ):
        raise RuntimeError("Locked 765/190 split failed during validation")
    splits = development_epoch_splits(
        labels, review, locked.development_indices
    )
    atomic_csv(
        validation_dir / "development_full_epoch_cv_membership.csv",
        epoch_cv_membership(review, splits),
    )
    args = runner.training_args(config, smoke=False)
    specs = runner.model_specs(config)
    matrices = runner.feature_matrices(graphs)
    weights = runner.review_sample_weights(review)
    selected_model_id = str(plan["selected_model_id"])
    trained_ids = list(plan["base_model_ids"])
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, Any]] = []
    expected_tasks = len(trained_ids) * len(config["model_seeds"])
    completed_tasks = 0
    for model_id in trained_ids:
        if model_id not in (*NEURAL_MODEL_IDS, "T1", "T3", "T4", "T2"):
            raise RuntimeError(f"Unsupported locked selected model: {model_id}")
        for model_seed in map(int, config["model_seeds"]):
            task_dir = validation_dir / "models" / model_id / f"seed_{model_seed}"
            if model_id in NEURAL_MODEL_IDS:
                rows, audit = train_validation_neural_seed(
                    model_id,
                    model_seed,
                    specs[model_id],
                    graphs,
                    review,
                    splits,
                    locked.development_indices,
                    locked.validation_indices,
                    paths["checkpoint"],
                    args,
                    task_dir,
                    resume=cli.resume,
                )
            else:
                if model_id == "T2":
                    raise RuntimeError(
                        "T2 was not selected; one-time full-development T2 "
                        "would require a separately locked Platt calibration step"
                    )
                rows, audit = train_validation_traditional_seed(
                    model_id,
                    model_seed,
                    matrices,
                    labels,
                    weights,
                    review,
                    locked.development_indices,
                    locked.validation_indices,
                    config,
                    task_dir,
                    resume=cli.resume,
                )
            predictions.append(rows)
            audits.append(audit)
            completed_tasks += 1
            print(
                json.dumps(
                    {
                        "completed": completed_tasks,
                        "total": expected_tasks,
                        "model_id": model_id,
                        "model_seed": model_seed,
                    }
                ),
                flush=True,
            )
    summary = assemble_validation_results(
        predictions,
        selected_model_id,
        config,
        validation_dir,
    )
    atomic_json(
        validation_dir / "VALIDATION_ACCESS_EVENT.json",
        {
            "status": "ONE_TIME_VALIDATION_RESULTS_ACCESSED_AFTER_UNLOCK",
            "unlock_audit_sha256": plan["validation_unlock_audit_sha256"],
            "base_models_locked_before_access": True,
            "epoch_rule_locked_before_access": True,
            "fusion_rule_locked_before_access": True,
            "validation_results_accessed": True,
        },
    )
    completion = {
        "status": "LOCKED_VALIDATION_COMPLETE",
        "completed_tasks": completed_tasks,
        "expected_tasks": expected_tasks,
        "failure_count": 0,
        **summary,
        "validation_run_lock_sha256": sha256_file(
            validation_dir / "VALIDATION_RUN_LOCK.json"
        ),
        "validation_predictions_sha256": sha256_file(
            validation_dir / "validation_predictions.csv"
        ),
        "validation_metrics_sha256": sha256_file(
            validation_dir / "validation_metrics.csv"
        ),
        "prior_project_artifacts_used": False,
    }
    atomic_json(validation_dir / "VALIDATION_COMPLETION.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
