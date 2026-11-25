#!/usr/bin/env python3
"""Run resumable, nested scaffold OOF models for the current 955 molecules."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ROOT = Path(__file__).resolve().parents[2]
for path in (
    ROOT / "src" / "code",
    ROOT / "src" / "code" / "models",
    ROOT / "src" / "code" / "utils",
    ROOT / "src" / "code" / "data_processing",
    ROOT / "src" / "code" / "experiments",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from formal_transfer_ablation_core import (  # noqa: E402
    ModelSpec,
    add_group_metadata,
    build_model,
    classification_metrics,
    graph_labels,
    grouped_folds,
    inner_grouped_split,
    leakage_record,
    predict_model,
    resolve_device,
    review_sample_weights,
    train_neural,
)
from locked_validation_core import build_locked_split  # noqa: E402
from pipeline_utils import load_graphs, set_seed  # noqa: E402


MODEL_NAMES = {
    "A0": "A0_Scratch_GINE",
    "A1": "A1_Pretrained_GINE",
    "A2": "A2_Pretrained_DirectConcat",
    "A3": "A3_Pretrained_GatedFusion",
    "A4": "A4_GatedFusion_GINE_Zero",
    "A5": "A5_GatedFusion_RDKit_Zero",
    "A6": "A6_GatedFusion_Morgan_Zero",
    "A7": "A7_GatedFusion_MACCS_Zero",
    "A8": "A8_GatedFusion_Family_Augmented",
    "T1": "T1_RF_Flat",
    "T2": "T2_RBF_SVM_Flat",
    "T3": "T3_RF_RDKit",
    "T4": "T4_ExtraTrees_RDKit",
    "F0": "F0_ValidationCrossFit_A3_T1",
    "BEST": "InnerSelectedBestSingle",
}
CORE_NEURAL_IDS = ("A0", "A1", "A2", "A3")
ABLATION_NEURAL_IDS = ("A4", "A5", "A6", "A7", "A8")
TRADITIONAL_IDS = ("T1", "T2", "T3", "T4")
BEST_CANDIDATE_IDS = (*CORE_NEURAL_IDS, *TRADITIONAL_IDS)
BEST_SIMPLICITY_ORDER = ("T3", "T4", "A0", "A1", "T1", "T2", "A2", "A3")
BEST_SIMPLICITY_RANK = {
    model_id: rank for rank, model_id in enumerate(BEST_SIMPLICITY_ORDER)
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    # ZIP archives have no Git metadata; source hashes remain the provenance anchor.
    return completed.stdout.strip() if completed.returncode == 0 else "unversioned_archive"


def locked_source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__).resolve(),
        ROOT / "src/code/experiments/formal_transfer_ablation_core.py",
        ROOT / "src/code/experiments/locked_validation_core.py",
        ROOT / "src/code/experiments/validation_fusion_core.py",
        ROOT / "src/code/models/gine_v2.py",
    )
    return {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in paths
    }


def stable_seed(*parts: Any) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:8], 16) % (2**31 - 1)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/current_rebuild_v1.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def resolve_config(
    path_value: str,
    allow_unapproved_checkpoint: bool = False,
) -> tuple[dict[str, Any], dict[str, Path]]:
    config_path = Path(path_value)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {"config": config_path.resolve()}
    for key in ("data", "review_csv", "checkpoint", "pretraining_summary", "output_dir"):
        value = Path(config[key])
        paths[key] = value.resolve() if value.is_absolute() else (ROOT / value).resolve()
    required_keys = ["data", "review_csv", "checkpoint"]
    if not allow_unapproved_checkpoint:
        required_keys.append("pretraining_summary")
    for key in required_keys:
        if not paths[key].exists():
            raise FileNotFoundError(paths[key])
    summary = (
        json.loads(paths["pretraining_summary"].read_text(encoding="utf-8"))
        if paths["pretraining_summary"].exists()
        else {}
    )
    if not summary.get("approved_checkpoint") and not allow_unapproved_checkpoint:
        raise RuntimeError("Pretraining checkpoint is not approved for downstream use")
    return config, paths


def training_args(config: dict[str, Any], smoke: bool) -> SimpleNamespace:
    values = config["training"]
    epochs = 4 if smoke else int(values["epochs"])
    return SimpleNamespace(
        epochs=epochs,
        freeze_epochs=0,
        min_finetune_epochs=2,
        patience=2 if smoke else int(values["patience"]),
        min_delta=float(values["min_delta"]),
        batch_size=int(values["batch_size"]),
        encoder_hidden_dim=int(values["encoder_hidden_dim"]),
        encoder_layers=int(values["encoder_layers"]),
        encoder_dropout=float(values["encoder_dropout"]),
        head_hidden_dim=int(values["graph_head_hidden_dim"]),
        head_dropout=float(values["head_dropout"]),
        branch_dim=int(values["hybrid_branch_dim"]),
        fusion="concat",
        head_lr=float(values["head_lr"]),
        pretrained_encoder_lr=float(values["common_encoder_lr"]),
        scratch_staged_encoder_lr=float(values["common_encoder_lr"]),
        scratch_full_encoder_lr=float(values["common_encoder_lr"]),
        lr_scheduler="cosine",
        warmup_epochs=int(values["warmup_epochs"]),
        min_lr_ratio=float(values["min_lr_ratio"]),
        weight_decay=float(values["weight_decay"]),
        grad_clip=float(values["grad_clip"]),
        classification_threshold=float(values["classification_threshold"]),
        device="cpu" if smoke else str(values["device"]),
        rf_estimators=50 if smoke else int(config["traditional_models"]["n_estimators"]),
    )


def model_specs(config: dict[str, Any]) -> dict[str, ModelSpec]:
    train = config["training"]
    v3_protocol = "model_feature_policy" in config
    hybrid_branches = ("gine", "rdkit", "morgan", "maccs")
    if not v3_protocol and "feature_policy" in config:
        hybrid_branches = (*hybrid_branches, "family")
    common_graph = {
        "branches": ("gine",),
        "schedule": "full",
        "encoder_lr_source": "scratch_full",
        "head_hidden_dim": int(train["graph_head_hidden_dim"]),
    }
    common_hybrid = {
        "branches": hybrid_branches,
        "initialization": "pretrained",
        "schedule": "full",
        "encoder_lr_source": "scratch_full",
        "branch_dim": int(train["hybrid_branch_dim"]),
    }
    specs = {
        "A0": ModelSpec(
            MODEL_NAMES["A0"], initialization="scratch", **common_graph
        ),
        "A1": ModelSpec(
            MODEL_NAMES["A1"], initialization="pretrained", **common_graph
        ),
        "A2": ModelSpec(
            MODEL_NAMES["A2"],
            fusion="concat",
            head_hidden_dim=int(train["direct_concat_head_hidden_dim"]),
            **common_hybrid,
        ),
        "A3": ModelSpec(
            MODEL_NAMES["A3"],
            fusion="gated",
            head_hidden_dim=int(train["gated_head_hidden_dim"]),
            **common_hybrid,
        ),
    }
    if v3_protocol:
        for model_id, branch in zip(
            ("A4", "A5", "A6", "A7"),
            hybrid_branches,
        ):
            specs[model_id] = ModelSpec(
                MODEL_NAMES[model_id],
                fusion="gated",
                head_hidden_dim=int(train["gated_head_hidden_dim"]),
                zeroed_branches=(branch,),
                **common_hybrid,
            )
        specs["A8"] = ModelSpec(
            MODEL_NAMES["A8"],
            branches=(*hybrid_branches, "family"),
            initialization="pretrained",
            schedule="full",
            encoder_lr_source="scratch_full",
            branch_dim=int(train["hybrid_branch_dim"]),
            fusion="gated",
            head_hidden_dim=int(train["gated_head_hidden_dim"]),
        )
    return specs


def prepare_data(paths: dict[str, Path]) -> tuple[list, pd.DataFrame, np.ndarray]:
    graphs = load_graphs(paths["data"])
    review = pd.read_csv(paths["review_csv"])
    if len(graphs) != 955 or len(review) != 955:
        raise RuntimeError("Current rebuild must use exactly 955 molecules")
    if review["inchikey"].nunique() != 955:
        raise RuntimeError("Review manifest is not InChIKey-unique")
    has_family = [hasattr(graph, "scaffold_family_fp") for graph in graphs]
    if any(has_family):
        if not all(has_family):
            raise RuntimeError("Scaffold-family features are only partially present")
        family_dims = {
            int(graph.scaffold_family_fp.detach().reshape(-1).shape[0])
            for graph in graphs
        }
        if len(family_dims) != 1 or next(iter(family_dims)) <= 0:
            raise RuntimeError("Scaffold-family feature dimensions are inconsistent")
        if any(
            not np.isclose(float(graph.scaffold_family_fp.sum().item()), 1.0)
            for graph in graphs
        ):
            raise RuntimeError("Scaffold-family features must be one-hot encoded")
    review = review.copy()
    review["sample_index"] = np.arange(len(review), dtype=int)
    review["source_graph_row_index"] = np.arange(len(review), dtype=int)
    review = add_group_metadata(review)
    labels = graph_labels(graphs)
    if not np.array_equal(labels, review["active_label"].to_numpy(int)):
        raise RuntimeError("Graph labels and review labels disagree")
    return graphs, review, labels


def build_fold_lookup(
    labels: np.ndarray,
    review: pd.DataFrame,
    development_indices: np.ndarray,
    assignment_seeds: list[int],
    outer_folds: int,
    inner_folds: int,
) -> tuple[
    dict[
        tuple[int, int],
        tuple[
            np.ndarray,
            np.ndarray,
            tuple[tuple[np.ndarray, np.ndarray], ...],
        ],
    ],
    pd.DataFrame,
    pd.DataFrame,
]:
    lookup = {}
    assignment_rows = []
    leakage_rows = []
    development_indices = np.asarray(development_indices, dtype=int)
    development_labels = labels[development_indices]
    development_review = review.iloc[development_indices].reset_index(drop=True)
    for assignment_seed in assignment_seeds:
        local_outer = grouped_folds(
            development_labels,
            development_review,
            "scaffold_group",
            assignment_seed,
            outer_folds,
        )
        for fold, (local_outer_train, local_outer_valid) in enumerate(local_outer):
            outer_train = development_indices[local_outer_train]
            outer_valid = development_indices[local_outer_valid]
            local_inner = grouped_folds(
                labels[outer_train],
                review.iloc[outer_train].reset_index(drop=True),
                "scaffold_group",
                stable_seed("inner", assignment_seed, fold),
                inner_folds,
            )
            inner_splits = tuple(
                (
                    outer_train[np.asarray(local_train, dtype=int)],
                    outer_train[np.asarray(local_valid, dtype=int)],
                )
                for local_train, local_valid in local_inner
            )
            lookup[(assignment_seed, fold)] = (
                outer_train,
                outer_valid,
                inner_splits,
            )
            outer_record = leakage_record(
                review,
                outer_train,
                outer_valid,
                "scaffold_group",
                assignment_seed,
                fold,
                "outer",
            )
            outer_record["inner_fold"] = -1
            leakage_rows.append(outer_record)
            inner_validation_union: set[int] = set()
            for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits):
                inner_record = leakage_record(
                    review,
                    inner_train,
                    inner_valid,
                    "scaffold_group",
                    assignment_seed,
                    fold,
                    "inner",
                )
                inner_record["inner_fold"] = inner_fold
                leakage_rows.append(inner_record)
                inner_validation_union.update(map(int, inner_valid))
            if inner_validation_union != set(map(int, outer_train)):
                raise RuntimeError(
                    "Complete inner CV does not cover every outer-training molecule once"
                )
            for sample_index in outer_valid:
                row = review.iloc[int(sample_index)]
                assignment_rows.append(
                    {
                        "assignment_seed": assignment_seed,
                        "outer_fold": fold,
                        "sample_index": int(sample_index),
                        "inchikey": row["inchikey"],
                        "scaffold_group": row["scaffold_group"],
                        "active_label": int(row["active_label"]),
                    }
                )
    assignments = pd.DataFrame(assignment_rows)
    leakage = pd.DataFrame(leakage_rows)
    if not leakage["target_group_check_pass"].eq(1).all():
        raise RuntimeError("Scaffold leakage detected")
    expected = len(development_indices) * len(assignment_seeds)
    if len(assignments) != expected:
        raise RuntimeError("Outer assignment coverage is incomplete")
    return lookup, assignments, leakage


def split_membership_table(
    review: pd.DataFrame,
    lookup: dict[
        tuple[int, int],
        tuple[
            np.ndarray,
            np.ndarray,
            tuple[tuple[np.ndarray, np.ndarray], ...],
        ],
    ],
    locked_validation_indices: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (assignment_seed, outer_fold), (
        outer_train,
        outer_evaluation,
        inner_splits,
    ) in sorted(lookup.items()):
        for inner_fold, (inner_train, inner_validation) in enumerate(inner_splits):
            role_by_index = {
                **{int(index): "inner_training" for index in inner_train},
                **{int(index): "inner_validation" for index in inner_validation},
                **{int(index): "outer_evaluation" for index in outer_evaluation},
                **{
                    int(index): "locked_internal_validation"
                    for index in locked_validation_indices
                },
            }
            if len(role_by_index) != len(review):
                raise RuntimeError("A split does not assign every current molecule once")
            if set(map(int, outer_train)) != set(map(int, inner_train)) | set(
                map(int, inner_validation)
            ):
                raise RuntimeError(
                    "Inner training and validation do not cover outer training"
                )
            for sample_index, role in sorted(role_by_index.items()):
                item = review.iloc[sample_index]
                rows.append(
                    {
                        "assignment_seed": assignment_seed,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "sample_index": sample_index,
                        "inchikey": item["inchikey"],
                        "scaffold_group": item["scaffold_group"],
                        "active_label": int(item["active_label"]),
                        "role": role,
                    }
                )
    return pd.DataFrame(rows)


def parameter_audit(
    specs: dict[str, ModelSpec],
    example_graph: Any,
    checkpoint: Path,
    args: SimpleNamespace,
) -> pd.DataFrame:
    rows = []
    for model_id, spec in specs.items():
        scratch_spec = copy.copy(spec)
        if scratch_spec.initialization == "pretrained":
            scratch_spec = ModelSpec(
                **{
                    **asdict(scratch_spec),
                    "initialization": "scratch",
                }
            )
        model, _, _, _ = build_model(
            scratch_spec, example_graph, checkpoint, args, initialization_seed=13
        )
        rows.append(
            {
                "model_id": model_id,
                "model": MODEL_NAMES[model_id],
                "total_parameters": sum(int(p.numel()) for p in model.parameters()),
                "trainable_parameters": sum(
                    int(p.numel()) for p in model.parameters() if p.requires_grad
                ),
                "fusion": spec.fusion,
                "branches": ",".join(spec.branches),
                "zeroed_branches": ",".join(spec.zeroed_branches),
                "branch_dim": spec.branch_dim,
                "head_hidden_dim": spec.head_hidden_dim,
            }
        )
    frame = pd.DataFrame(rows)
    indexed = frame.set_index("model_id")
    if indexed.loc["A0", "total_parameters"] != indexed.loc["A1", "total_parameters"]:
        raise RuntimeError("A0/A1 parameter counts differ")
    direct = float(indexed.loc["A2", "total_parameters"])
    gated = float(indexed.loc["A3", "total_parameters"])
    relative = abs(direct - gated) / max(direct, gated)
    frame["a2_a3_relative_parameter_difference"] = relative
    if relative > 0.05:
        raise RuntimeError(f"A2/A3 parameter mismatch is {relative:.2%}")
    if set(ABLATION_NEURAL_IDS).issubset(indexed.index):
        for model_id in ABLATION_NEURAL_IDS[:-1]:
            if (
                indexed.loc[model_id, "total_parameters"]
                != indexed.loc["A3", "total_parameters"]
            ):
                raise RuntimeError(f"{model_id}/A3 parameter counts differ")
        if (
            indexed.loc["A8", "total_parameters"]
            <= indexed.loc["A3", "total_parameters"]
        ):
            raise RuntimeError("A8 must add parameters only through the family branch")
    return frame


def clean_audit(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in payload.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned[key] = None
        elif isinstance(value, np.generic):
            cleaned[key] = value.item()
        else:
            cleaned[key] = value
    return cleaned


def neural_task(
    model_id: str,
    spec: ModelSpec,
    graphs: list,
    review: pd.DataFrame,
    indices: tuple[
        np.ndarray,
        np.ndarray,
        tuple[tuple[np.ndarray, np.ndarray], ...],
    ],
    assignment_seed: int,
    model_seed: int,
    fold: int,
    args: SimpleNamespace,
    checkpoint: Path,
    run_dir: Path,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_path = run_dir / "predictions.csv"
    audit_path = run_dir / "audit.json"
    if not force and prediction_path.exists() and audit_path.exists():
        return pd.read_csv(prediction_path), json.loads(
            audit_path.read_text(encoding="utf-8")
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    outer_train, outer_valid, inner_splits = indices
    inner_probability = np.full(len(review), np.nan, dtype=float)
    selected_epochs: list[int] = []
    inner_audits: list[dict[str, Any]] = []
    inner_histories: list[pd.DataFrame] = []
    inner_prediction_rows: list[pd.DataFrame] = []
    for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits):
        shared_seed = stable_seed(
            "neural-inner",
            assignment_seed,
            model_seed,
            fold,
            inner_fold,
        )
        inner_model, inner_graphs, inner_history, inner_audit, selected_epoch = (
            train_neural(
                spec,
                graphs,
                inner_train,
                checkpoint,
                args,
                shared_seed,
                args.epochs,
                valid_idx=inner_valid,
            )
        )
        fold_probability = predict_model(
            inner_model,
            inner_graphs,
            inner_valid,
            resolve_device(args.device),
            args.batch_size,
        )
        inner_probability[inner_valid] = fold_probability
        selected_epochs.append(int(selected_epoch))
        inner_audits.append(clean_audit(inner_audit))
        history_frame = pd.DataFrame(inner_history)
        history_frame.insert(0, "inner_fold", inner_fold)
        inner_histories.append(history_frame)
        inner_rows = review.iloc[inner_valid][
            ["sample_index", "inchikey", "scaffold_group", "active_label"]
        ].copy()
        inner_rows.insert(0, "inner_fold", inner_fold)
        inner_rows["probability"] = fold_probability
        inner_prediction_rows.append(inner_rows)
        del inner_model, inner_graphs
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    if np.isnan(inner_probability[outer_train]).any():
        raise RuntimeError("Neural inner OOF predictions are incomplete")
    selected_epoch = int(round(float(np.median(selected_epochs))))
    pooled_inner_pr = float(
        average_precision_score(
            review.iloc[outer_train]["active_label"].to_numpy(dtype=int),
            inner_probability[outer_train],
        )
    )
    pooled_inner_brier = float(
        brier_score_loss(
            review.iloc[outer_train]["active_label"].to_numpy(dtype=int),
            inner_probability[outer_train],
        )
    )
    atomic_csv(
        run_dir / "inner_training_history.csv",
        pd.concat(inner_histories, ignore_index=True),
    )
    atomic_csv(
        run_dir / "inner_oof_predictions.csv",
        pd.concat(inner_prediction_rows, ignore_index=True),
    )

    hash_fields = (
        "head_initial_sha256",
        "training_order_signature",
        "lr_trajectory_sha256",
    )
    inner_summary: dict[str, Any] = {
        "fold_count": len(inner_splits),
        "selected_epochs": selected_epochs,
        "selected_epoch_median": selected_epoch,
        "best_inner_pr_auc": pooled_inner_pr,
        "inner_brier": pooled_inner_brier,
        "folds": inner_audits,
    }
    for field in hash_fields:
        inner_summary[field] = hashlib.sha256(
            json.dumps(
                [audit[field] for audit in inner_audits],
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
    encoder_lrs = [float(audit["base_encoder_lr"]) for audit in inner_audits]
    if not all(
        math.isclose(value, encoder_lrs[0], rel_tol=0.0, abs_tol=0.0)
        for value in encoder_lrs
    ):
        raise RuntimeError("Inner folds used different base encoder learning rates")
    inner_summary["base_encoder_lr"] = encoder_lrs[0]

    final_seed = stable_seed("neural-final", assignment_seed, model_seed, fold)
    final_model, fold_graphs, final_history, final_audit, _ = train_neural(
        spec,
        graphs,
        outer_train,
        checkpoint,
        args,
        final_seed,
        selected_epoch,
        valid_idx=None,
    )
    probability = predict_model(
        final_model,
        fold_graphs,
        outer_valid,
        resolve_device(args.device),
        args.batch_size,
    )
    del final_model, fold_graphs
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    atomic_csv(run_dir / "final_training_history.csv", pd.DataFrame(final_history))
    rows = review.iloc[outer_valid][
        ["sample_index", "inchikey", "canonical_smiles", "scaffold_group", "active_label"]
    ].copy()
    rows.insert(0, "model", MODEL_NAMES[model_id])
    rows.insert(0, "model_id", model_id)
    rows.insert(0, "outer_fold", fold)
    rows.insert(0, "model_seed", model_seed)
    rows.insert(0, "assignment_seed", assignment_seed)
    rows["probability"] = probability
    audit = {
        "model_id": model_id,
        "model": MODEL_NAMES[model_id],
        "assignment_seed": assignment_seed,
        "model_seed": model_seed,
        "outer_fold": fold,
        "selected_epoch": selected_epoch,
        "inner": inner_summary,
        "final": clean_audit(final_audit),
        "outer_metrics": classification_metrics(
            rows["active_label"].to_numpy(int),
            probability,
            args.classification_threshold,
        ),
    }
    atomic_csv(prediction_path, rows)
    atomic_json(audit_path, audit)
    return rows, audit


def feature_matrices(graphs: list) -> dict[str, np.ndarray]:
    rdkit = np.vstack(
        [graph.global_feat.detach().cpu().float().reshape(-1).numpy() for graph in graphs]
    )
    morgan = np.vstack(
        [graph.morgan_fp.detach().cpu().float().reshape(-1).numpy() for graph in graphs]
    )
    maccs = np.vstack(
        [graph.maccs_fp.detach().cpu().float().reshape(-1).numpy() for graph in graphs]
    )
    family = (
        np.vstack(
            [
                graph.scaffold_family_fp.detach().cpu().float().reshape(-1).numpy()
                for graph in graphs
            ]
        )
        if hasattr(graphs[0], "scaffold_family_fp")
        else np.empty((len(graphs), 0), dtype=np.float32)
    )
    return {
        "rdkit": rdkit.astype(np.float32),
        "family": family.astype(np.float32),
        "rdkit_family": np.concatenate([rdkit, family], axis=1).astype(np.float32),
        "flat": np.concatenate([rdkit, morgan, maccs], axis=1).astype(
            np.float32
        ),
        "flat_family": np.concatenate([rdkit, morgan, maccs, family], axis=1).astype(
            np.float32
        ),
    }


def traditional_preprocessor(model_id: str, feature_count: int) -> Any:
    if model_id in {"T1", "T2"}:
        descriptor_steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if model_id == "T2":
            descriptor_steps.append(("scaler", StandardScaler()))
        return ColumnTransformer(
            [
                ("rdkit", Pipeline(descriptor_steps), slice(0, 171)),
                ("fingerprints", "passthrough", slice(171, feature_count)),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
    return SimpleImputer(strategy="median")


def make_traditional(
    model_id: str,
    config: dict[str, Any],
    seed: int,
    feature_count: int,
) -> Any:
    traditional = config["traditional_models"]
    common = {
        "n_estimators": int(traditional["n_estimators"]),
        "min_samples_leaf": int(traditional["min_samples_leaf"]),
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": seed,
        "n_jobs": -1,
    }
    if model_id in {"T1", "T3"}:
        estimator = RandomForestClassifier(**common)
    elif model_id == "T4":
        estimator = ExtraTreesClassifier(**common)
    if model_id == "T2":
        estimator = SVC(
            C=float(traditional["svm_C"]),
            kernel="rbf",
            gamma="scale",
            class_weight="balanced",
            probability=False,
            random_state=seed,
        )
    if model_id not in TRADITIONAL_IDS:
        raise ValueError(model_id)
    return Pipeline(
        [
            ("features", traditional_preprocessor(model_id, feature_count)),
            ("model", estimator),
        ]
    )


def fit_traditional_model(
    model_id: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    train_idx: np.ndarray,
    config: dict[str, Any],
    seed: int,
) -> Any:
    model = make_traditional(model_id, config, seed, matrix.shape[1])
    model.fit(
        matrix[train_idx],
        labels[train_idx],
        model__sample_weight=weights[train_idx],
    )
    return model


def traditional_scores(model_id: str, model: Any, matrix: np.ndarray) -> np.ndarray:
    if model_id == "T2":
        return np.asarray(model.decision_function(matrix), dtype=float)
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=float)


def fit_platt_calibrator(
    decision_scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    seed: int,
) -> LogisticRegression:
    calibrator = LogisticRegression(
        solver="lbfgs",
        random_state=seed,
        max_iter=1000,
    )
    calibrator.fit(
        np.asarray(decision_scores, dtype=float).reshape(-1, 1),
        np.asarray(labels, dtype=int),
        sample_weight=np.asarray(weights, dtype=float),
    )
    return calibrator


def platt_probability(
    calibrator: LogisticRegression,
    decision_scores: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        calibrator.predict_proba(
            np.asarray(decision_scores, dtype=float).reshape(-1, 1)
        )[:, 1],
        dtype=float,
    )


def traditional_task(
    model_id: str,
    matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    weights: np.ndarray,
    review: pd.DataFrame,
    indices: tuple[
        np.ndarray,
        np.ndarray,
        tuple[tuple[np.ndarray, np.ndarray], ...],
    ],
    assignment_seed: int,
    model_seed: int,
    fold: int,
    config: dict[str, Any],
    run_dir: Path,
    force: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    prediction_path = run_dir / "predictions.csv"
    audit_path = run_dir / "audit.json"
    if not force and prediction_path.exists() and audit_path.exists():
        return pd.read_csv(prediction_path), json.loads(
            audit_path.read_text(encoding="utf-8")
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    outer_train, outer_valid, inner_splits = indices
    matrix_name = "flat" if model_id in {"T1", "T2"} else "rdkit"
    matrix = matrices[matrix_name]
    seed = stable_seed("traditional", model_id, assignment_seed, model_seed, fold)
    started = time.time()
    inner_probability = np.full(len(labels), np.nan, dtype=float)
    inner_prediction_rows: list[pd.DataFrame] = []
    for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits):
        fold_model = fit_traditional_model(
            model_id,
            matrix,
            labels,
            weights,
            inner_train,
            config,
            stable_seed(seed, "inner", inner_fold),
        )
        fold_score = traditional_scores(model_id, fold_model, matrix[inner_valid])
        inner_probability[inner_valid] = fold_score
        fold_rows = review.iloc[inner_valid][
            ["sample_index", "inchikey", "scaffold_group", "active_label"]
        ].copy()
        fold_rows.insert(0, "inner_fold", inner_fold)
        fold_rows["raw_score"] = fold_score
        inner_prediction_rows.append(fold_rows)
    if np.isnan(inner_probability[outer_train]).any():
        raise RuntimeError("Traditional inner OOF predictions are incomplete")
    calibration_audit: dict[str, Any] = {"method": "not_required"}
    if model_id == "T2":
        calibrator = fit_platt_calibrator(
            inner_probability[outer_train],
            labels[outer_train],
            weights[outer_train],
            stable_seed(seed, "platt"),
        )
        inner_probability[outer_train] = platt_probability(
            calibrator, inner_probability[outer_train]
        )
        calibration_audit = {
            "method": "four_fold_scaffold_oof_platt",
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
            "oof_count": int(len(outer_train)),
        }
    for fold_rows in inner_prediction_rows:
        idx = fold_rows["sample_index"].to_numpy(dtype=int)
        fold_rows["probability"] = inner_probability[idx]
    atomic_csv(
        run_dir / "inner_oof_predictions.csv",
        pd.concat(inner_prediction_rows, ignore_index=True),
    )
    outer_model = fit_traditional_model(
        model_id,
        matrix,
        labels,
        weights,
        outer_train,
        config,
        seed + 1,
    )
    outer_score = traditional_scores(model_id, outer_model, matrix[outer_valid])
    outer_probability = (
        platt_probability(calibrator, outer_score)
        if model_id == "T2"
        else outer_score
    )
    rows = review.iloc[outer_valid][
        ["sample_index", "inchikey", "canonical_smiles", "scaffold_group", "active_label"]
    ].copy()
    rows.insert(0, "model", MODEL_NAMES[model_id])
    rows.insert(0, "model_id", model_id)
    rows.insert(0, "outer_fold", fold)
    rows.insert(0, "model_seed", model_seed)
    rows.insert(0, "assignment_seed", assignment_seed)
    rows["probability"] = outer_probability
    inner_pr = float(
        average_precision_score(labels[outer_train], inner_probability[outer_train])
    )
    inner_brier = float(
        brier_score_loss(labels[outer_train], inner_probability[outer_train])
    )
    audit = {
        "model_id": model_id,
        "model": MODEL_NAMES[model_id],
        "assignment_seed": assignment_seed,
        "model_seed": model_seed,
        "outer_fold": fold,
        "feature_matrix": matrix_name,
        "inner_best_pr_auc": inner_pr,
        "inner_brier": inner_brier,
        "inner_fold_count": len(inner_splits),
        "calibration": calibration_audit,
        "elapsed_seconds": time.time() - started,
        "outer_metrics": classification_metrics(
            labels[outer_valid],
            outer_probability,
            float(config["training"]["classification_threshold"]),
        ),
    }
    atomic_csv(prediction_path, rows)
    atomic_json(audit_path, audit)
    return rows, audit


def fairness_audit(audits: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "model_id": audit["model_id"],
                "assignment_seed": audit["assignment_seed"],
                "model_seed": audit["model_seed"],
                "outer_fold": audit["outer_fold"],
                "inner_head_initial_sha256": audit["inner"]["head_initial_sha256"],
                "final_head_initial_sha256": audit["final"]["head_initial_sha256"],
                "inner_training_order_signature": audit["inner"][
                    "training_order_signature"
                ],
                "final_training_order_signature": audit["final"][
                    "training_order_signature"
                ],
                "inner_lr_trajectory_sha256": audit["inner"][
                    "lr_trajectory_sha256"
                ],
                "final_lr_trajectory_sha256": audit["final"][
                    "lr_trajectory_sha256"
                ],
                "inner_base_encoder_lr": audit["inner"]["base_encoder_lr"],
                "final_base_encoder_lr": audit["final"]["base_encoder_lr"],
            }
            for audit in audits
            if audit["model_id"] in {"A0", "A1"}
        ]
    )
    rows = []
    for keys, group in frame.groupby(
        ["assignment_seed", "model_seed", "outer_fold"], sort=True
    ):
        indexed = group.set_index("model_id")
        if set(indexed.index) != {"A0", "A1"}:
            raise RuntimeError("A0/A1 fairness pair is incomplete")
        a0 = indexed.loc["A0"]
        a1 = indexed.loc["A1"]
        checks = {
            column: int(a0[column] == a1[column])
            for column in frame.columns
            if column
            not in {"model_id", "assignment_seed", "model_seed", "outer_fold"}
        }
        rows.append(
            {
                "assignment_seed": keys[0],
                "model_seed": keys[1],
                "outer_fold": keys[2],
                **checks,
                "pass": int(all(checks.values())),
            }
        )
    result = pd.DataFrame(rows)
    if not result["pass"].eq(1).all():
        raise RuntimeError("A0/A1 initialization-only fairness audit failed")
    return result


def assemble_outputs(
    predictions: list[pd.DataFrame],
    neural_audits: list[dict[str, Any]],
    traditional_audits: list[dict[str, Any]],
    review: pd.DataFrame,
    args: SimpleNamespace,
    output: Path,
    base_model_ids: tuple[str, ...],
) -> None:
    base = pd.concat(predictions, ignore_index=True)
    audit_lookup = {
        (
            item["assignment_seed"],
            item["model_seed"],
            item["outer_fold"],
            item["model_id"],
        ): item
        for item in [*neural_audits, *traditional_audits]
    }
    composite_rows = []
    selection_rows = []
    for keys, fold_frame in base.groupby(
        ["assignment_seed", "model_seed", "outer_fold"], sort=True
    ):
        assignment_seed, model_seed, outer_fold = map(int, keys)
        pivot = fold_frame.pivot(
            index=[
                "sample_index",
                "inchikey",
                "canonical_smiles",
                "scaffold_group",
                "active_label",
            ],
            columns="model_id",
            values="probability",
        ).reset_index()
        if any(model_id not in pivot for model_id in base_model_ids):
            raise RuntimeError("A fold is missing single-model predictions")
        inner_scores: dict[str, tuple[float, float]] = {}
        for model_id in BEST_CANDIDATE_IDS:
            audit = audit_lookup[
                (assignment_seed, model_seed, outer_fold, model_id)
            ]
            if model_id in CORE_NEURAL_IDS:
                inner_scores[model_id] = (
                    float(audit["inner"]["best_inner_pr_auc"]),
                    float(audit["inner"]["inner_brier"]),
                )
            else:
                inner_scores[model_id] = (
                    float(audit["inner_best_pr_auc"]),
                    float(audit["inner_brier"]),
                )
        selected = min(
            inner_scores,
            key=lambda model_id: (
                -inner_scores[model_id][0],
                inner_scores[model_id][1],
                BEST_SIMPLICITY_RANK[model_id],
            ),
        )
        for model_id, (inner_pr, inner_brier) in inner_scores.items():
            selection_rows.append(
                {
                    "assignment_seed": assignment_seed,
                    "model_seed": model_seed,
                    "outer_fold": outer_fold,
                    "candidate_model_id": model_id,
                    "inner_pr_auc": inner_pr,
                    "inner_brier": inner_brier,
                    "simplicity_rank": BEST_SIMPLICITY_RANK[model_id],
                    "selected": int(model_id == selected),
                }
            )
        for row in pivot.to_dict(orient="records"):
            common = {
                "assignment_seed": assignment_seed,
                "model_seed": model_seed,
                "outer_fold": outer_fold,
                "sample_index": int(row["sample_index"]),
                "inchikey": row["inchikey"],
                "canonical_smiles": row["canonical_smiles"],
                "scaffold_group": row["scaffold_group"],
                "active_label": int(row["active_label"]),
            }
            composite_rows.append(
                {
                    **common,
                    "model_id": "BEST",
                    "model": MODEL_NAMES["BEST"],
                    "probability": float(row[selected]),
                    "inner_selected_source_model": selected,
                }
            )
    all_oof = pd.concat([base, pd.DataFrame(composite_rows)], ignore_index=True)
    expected_models = len(base_model_ids) + 1
    expected_sample_count = int(base["sample_index"].nunique())
    coverage = all_oof.groupby(
        ["assignment_seed", "model_seed", "model_id"]
    )["sample_index"].nunique()
    if len(coverage) != (
        all_oof["assignment_seed"].nunique()
        * all_oof["model_seed"].nunique()
        * expected_models
    ) or not coverage.eq(expected_sample_count).all():
        raise RuntimeError("OOF coverage is incomplete")
    metric_rows = []
    for keys, frame in all_oof.groupby(
        ["assignment_seed", "model_seed", "model_id", "model"], sort=True
    ):
        metric_rows.append(
            {
                "assignment_seed": int(keys[0]),
                "model_seed": int(keys[1]),
                "model_id": keys[2],
                "model": keys[3],
                **classification_metrics(
                    frame["active_label"].to_numpy(int),
                    frame["probability"].to_numpy(float),
                    args.classification_threshold,
                ),
            }
        )
    atomic_csv(output / "all_oof_predictions.csv", all_oof)
    atomic_csv(output / "best_selection_audit.csv", pd.DataFrame(selection_rows))
    atomic_csv(output / "assignment_seed_metrics.csv", pd.DataFrame(metric_rows))
    atomic_csv(
        output / "neural_training_audit.csv", pd.json_normalize(neural_audits)
    )
    atomic_csv(
        output / "a0_a1_fairness_audit.csv", fairness_audit(neural_audits)
    )


def main() -> None:
    cli = parse_args()
    config, paths = resolve_config(
        cli.config,
        allow_unapproved_checkpoint=cli.smoke,
    )
    args = training_args(config, cli.smoke)
    output = paths["output_dir"]
    if cli.smoke:
        output = output.parent / f"{output.name}_smoke"
    output.mkdir(parents=True, exist_ok=True)
    graphs, review, labels = prepare_data(paths)
    if "feature_policy" in config and any(
        not hasattr(graph, "scaffold_family_fp") for graph in graphs
    ):
        raise RuntimeError("The family-feature protocol requires scaffold_family_fp")
    matrices = feature_matrices(graphs)
    weights = review_sample_weights(review)
    specs = model_specs(config)
    neural_ids = tuple(specs)
    if any("family" in spec.branches for spec in specs.values()) and any(
        not hasattr(graph, "scaffold_family_fp") for graph in graphs
    ):
        raise RuntimeError("A family ablation requires scaffold_family_fp")
    assignment_seeds = list(map(int, config["assignment_seeds"]))
    model_seeds = list(map(int, config["model_seeds"]))
    outer_folds = int(config["outer_folds"])
    inner_folds = int(config["inner_folds"])
    if cli.smoke:
        assignment_seeds = assignment_seeds[:1]
        model_seeds = model_seeds[:1]
        outer_folds = 2
        # The smoke run reduces assignments, seeds, outer folds, epochs and tree
        # count, but deliberately retains the protocol's complete four-fold
        # inner CV because that is a formal launch gate.
    if "locked_split" not in config:
        raise RuntimeError("Current v3 runner requires a locked development split")
    locked = build_locked_split(review, labels, config["locked_split"])
    atomic_csv(output / "current_split_membership.csv", locked.membership)
    atomic_json(output / "locked_split_audit.json", locked.audit)
    lookup, assignments, leakage = build_fold_lookup(
        labels,
        review,
        locked.development_indices,
        assignment_seeds,
        outer_folds,
        inner_folds,
    )
    atomic_csv(output / "fold_assignments.csv", assignments)
    atomic_csv(output / "leakage_checks.csv", leakage)
    atomic_csv(
        output / "split_membership.csv",
        split_membership_table(review, lookup, locked.validation_indices),
    )
    parameters = parameter_audit(specs, graphs[0], paths["checkpoint"], args)
    atomic_csv(output / "parameter_count_audit.csv", parameters)

    lock = {
        "status": "LOCKED_BEFORE_OOF_TRAINING",
        "protocol_name": config["protocol_name"],
        "data_sha256": sha256_file(paths["data"]),
        "review_sha256": sha256_file(paths["review_csv"]),
        "checkpoint_sha256": sha256_file(paths["checkpoint"]),
        "pretraining_summary_sha256": (
            sha256_file(paths["pretraining_summary"])
            if paths["pretraining_summary"].exists()
            else "SMOKE_NOT_AVAILABLE"
        ),
        "config_sha256": sha256_file(paths["config"]),
        "git_commit": current_git_commit(),
        "source_sha256": locked_source_hashes(),
        "assignment_seeds": assignment_seeds,
        "model_seeds": model_seeds,
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "locked_split_audit": locked.audit,
        "models": {key: asdict(value) for key, value in specs.items()},
        "parameter_count_audit": parameters.to_dict(orient="records"),
        "scratch_pretrained_difference": "encoder initialization only",
        "direct_gated_parameter_tolerance": 0.05,
        "best_candidates": list(BEST_CANDIDATE_IDS),
        "best_selection_order": [
            "inner_pr_auc_desc",
            "inner_brier_asc",
            "prelocked_simplicity_rank_asc",
        ],
        "best_simplicity_order": list(BEST_SIMPLICITY_ORDER),
        "svm_calibration": "four_fold_scaffold_oof_platt",
        "prior_project_artifacts_used": False,
        "scope_policy": config.get("scope_policy", {}),
        "smoke": cli.smoke,
    }
    lock_path = output / "RUN_LOCK.json"
    if lock_path.exists() and not cli.force:
        existing = json.loads(lock_path.read_text(encoding="utf-8"))
        if existing != lock:
            raise RuntimeError("Existing run lock differs from current protocol")
    else:
        atomic_json(lock_path, lock)

    predictions = []
    neural_audits = []
    traditional_audits = []
    total = (
        len(assignment_seeds)
        * len(model_seeds)
        * outer_folds
        * (len(neural_ids) + len(TRADITIONAL_IDS))
    )
    completed = 0
    failures = []
    for assignment_seed in assignment_seeds:
        for model_seed in model_seeds:
            for fold in range(outer_folds):
                indices = lookup[(assignment_seed, fold)]
                for model_id in neural_ids:
                    run_dir = (
                        output
                        / "runs"
                        / f"assignment_{assignment_seed}"
                        / f"model_seed_{model_seed}"
                        / f"fold_{fold}"
                        / model_id
                    )
                    try:
                        rows, audit = neural_task(
                            model_id,
                            specs[model_id],
                            graphs,
                            review,
                            indices,
                            assignment_seed,
                            model_seed,
                            fold,
                            args,
                            paths["checkpoint"],
                            run_dir,
                            cli.force,
                        )
                        predictions.append(rows)
                        neural_audits.append(audit)
                        completed += 1
                        print(
                            json.dumps(
                                {
                                    "completed": completed,
                                    "total": total,
                                    "assignment_seed": assignment_seed,
                                    "model_seed": model_seed,
                                    "fold": fold,
                                    "model_id": model_id,
                                }
                            ),
                            flush=True,
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "assignment_seed": assignment_seed,
                                "model_seed": model_seed,
                                "fold": fold,
                                "model_id": model_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        atomic_csv(output / "failed_runs.csv", pd.DataFrame(failures))
                        raise
                for model_id in TRADITIONAL_IDS:
                    run_dir = (
                        output
                        / "runs"
                        / f"assignment_{assignment_seed}"
                        / f"model_seed_{model_seed}"
                        / f"fold_{fold}"
                        / model_id
                    )
                    try:
                        rows, audit = traditional_task(
                            model_id,
                            matrices,
                            labels,
                            weights,
                            review,
                            indices,
                            assignment_seed,
                            model_seed,
                            fold,
                            config,
                            run_dir,
                            cli.force,
                        )
                        predictions.append(rows)
                        traditional_audits.append(audit)
                        completed += 1
                        print(
                            json.dumps(
                                {
                                    "completed": completed,
                                    "total": total,
                                    "assignment_seed": assignment_seed,
                                    "model_seed": model_seed,
                                    "fold": fold,
                                    "model_id": model_id,
                                }
                            ),
                            flush=True,
                        )
                    except Exception as exc:
                        failures.append(
                            {
                                "assignment_seed": assignment_seed,
                                "model_seed": model_seed,
                                "fold": fold,
                                "model_id": model_id,
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                        atomic_csv(output / "failed_runs.csv", pd.DataFrame(failures))
                        raise

    assemble_outputs(
        predictions,
        neural_audits,
        traditional_audits,
        review,
        args,
        output,
        (*neural_ids, *TRADITIONAL_IDS),
    )
    atomic_csv(
        output / "traditional_training_audit.csv",
        pd.DataFrame(traditional_audits),
    )
    completion = {
        "status": "SMOKE_COMPLETE" if cli.smoke else "FORMAL_OOF_COMPLETE",
        "completed_tasks": completed,
        "expected_tasks": total,
        "failure_count": len(failures),
        "structure_count": len(review),
        "assignment_count": len(assignment_seeds),
        "model_seed_count": len(model_seeds),
        "outer_folds": outer_folds,
        "inner_folds": inner_folds,
        "single_model_count": len(neural_ids) + len(TRADITIONAL_IDS),
        "composite_model_count": 1,
    }
    atomic_json(output / "RUN_COMPLETION.json", completion)
    print(json.dumps(completion, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
