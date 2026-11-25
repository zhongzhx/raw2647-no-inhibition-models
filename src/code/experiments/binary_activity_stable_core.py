"""Auditable utilities for the RAW264.7 pure-binary stable evaluation.

This module is deliberately independent of all previous evaluation outputs.
It consumes only the locked development graph/index inputs and the encoder-only
pretraining checkpoint declared by the new protocol configuration.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import random
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

from pharmafusion_evaluation_core import (
    ArchitectureSpec,
    build_model,
    cosine_factor,
    fit_descriptor_scaler,
    parameter_hash,
    predict_model,
    resolve_device,
)
from pharmafusion_gps_adapter import PharmaFusionConfig
from small_sample_ml_core import FoldSafePreprocessor


MODEL_NAMES = {
    "M0": "Pretrained GINE baseline",
    "M1": "ExtraTrees + RDKit",
    "M2": "Random Forest + Flat features",
    "M3": "Random Forest + GatedFusion embedding",
    "M4": "GatedFusion MLP",
    "M5": "Pretrained GINE + Adapter d=32",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def safe_path(path: str | Path, forbidden_tokens: Sequence[str]) -> Path:
    resolved = Path(path).expanduser().resolve()
    lowered = str(resolved).lower()
    if any(str(token).lower() in lowered for token in forbidden_tokens):
        raise RuntimeError(f"Forbidden locked/exposed test path: {resolved}")
    return resolved


def atomic_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, target)


def atomic_csv(path: str | Path, frame: pd.DataFrame) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, target)


def peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def set_all_seeds(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def normalize_raw_label(value: Any) -> str:
    return str(value).strip().replace("＞", ">").replace("μ", "u").upper()


def _censored_weight(threshold: float) -> tuple[float, str]:
    if threshold >= 50.0:
        return 1.00, "censored_ge50_inactive"
    if threshold >= 30.0:
        return 0.85, "censored_30_to_lt50_inactive"
    if threshold >= 20.0:
        return 0.75, "censored_20_to_lt30_inactive"
    if threshold >= 10.0:
        return 0.65, "censored_10_to_lt20_inactive"
    return 0.55, "censored_lt10_inactive"


def derive_binary_label_manifest(metadata: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the pre-locked four-source label rule without using old labels blindly."""
    required = {
        "development_index", "inchikey", "canonical_smiles", "ic50_raw_uM",
        "ic50_uM_numeric", "has_exact_pic50", "is_censored_gt50",
        "censored_threshold_uM", "scaffold_group",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Development index is missing columns: {sorted(missing)}")
    if metadata["inchikey"].astype(str).duplicated().any():
        raise RuntimeError("Fixed development data contain duplicate InChIKeys")

    rows: list[dict[str, Any]] = []
    for position, row in metadata.reset_index(drop=True).iterrows():
        raw = normalize_raw_label(row.ic50_raw_uM)
        exact = int(row.has_exact_pic50) == 1
        numeric = pd.to_numeric(pd.Series([row.ic50_uM_numeric]), errors="coerce").iloc[0]
        if exact and not np.isfinite(numeric) and "pic50" in metadata.columns:
            pic50 = pd.to_numeric(pd.Series([row.pic50]), errors="coerce").iloc[0]
            if np.isfinite(pic50):
                numeric = 10.0 ** (6.0 - float(pic50))
        threshold = pd.to_numeric(pd.Series([row.censored_threshold_uM]), errors="coerce").iloc[0]
        included = True
        reason = ""
        label_source = ""
        source_stratum = ""
        label = -1
        w1 = float("nan")

        if exact:
            if not np.isfinite(numeric):
                raise RuntimeError(f"Exact record lacks numeric IC50 at development_index={row.development_index}")
            label = int(float(numeric) <= 50.0)
            label_source = "exact_active" if label else "exact_inactive"
            source_stratum = label_source
            w1 = 1.0
        elif raw in {"NS", "ND"}:
            label = 0
            label_source = "NS"
            source_stratum = "NS_inactive"
            w1 = 1.0
        elif raw.startswith(">") or int(row.is_censored_gt50) == 1:
            if not np.isfinite(threshold):
                import re

                match = re.search(r">\s*([0-9]+(?:\.[0-9]+)?)", raw)
                if match:
                    threshold = float(match.group(1))
            if not np.isfinite(threshold):
                raise RuntimeError(f"Censored record lacks threshold at development_index={row.development_index}")
            label = 0
            label_source = "censored_gt"
            w1, source_stratum = _censored_weight(float(threshold))
        else:
            included = False
            reason = "unsupported label syntax"
            source_stratum = "unsupported_excluded"

        rows.append(
            {
                **row.to_dict(),
                "source_position": int(position),
                "normalized_raw_label": raw,
                "binary_label": int(label),
                "label_source": label_source,
                "label_source_stratum": source_stratum,
                "sample_weight_W1": float(w1),
                "sample_weight_W0": 1.0 if included else float("nan"),
                "eligible_for_modeling": int(included),
                "exclusion_reason": reason,
            }
        )
    manifest = pd.DataFrame(rows)
    eligible = manifest[manifest.eligible_for_modeling.eq(1)].copy()
    eligible["binary_index"] = np.arange(len(eligible), dtype=int)
    manifest = manifest.merge(
        eligible[["development_index", "binary_index"]], on="development_index", how="left", validate="one_to_one"
    )

    if not set(eligible.label_source).issubset({"exact_active", "exact_inactive", "NS", "censored_gt"}):
        raise RuntimeError("Unexpected label_source entered modeling")
    if not eligible.loc[eligible.label_source.eq("NS"), "binary_label"].eq(0).all():
        raise RuntimeError("NS label audit failed")
    if not eligible.loc[eligible.label_source.eq("censored_gt"), "binary_label"].eq(0).all():
        raise RuntimeError("Censored label audit failed")
    if not eligible.sample_weight_W1.between(0.55, 1.0).all():
        raise RuntimeError("Training weight audit failed")
    if eligible.inchikey.astype(str).duplicated().any():
        raise RuntimeError("Eligible structures are not unique by InChIKey")

    audit = {
        "fixed_development_records": int(len(manifest)),
        "eligible_modeling_structures": int(len(eligible)),
        "ns_inactive": int(eligible.label_source.eq("NS").sum()),
        "excluded_other_unsupported": int(manifest.label_source_stratum.eq("unsupported_excluded").sum()),
        "unique_inchikeys_fixed_development": int(manifest.inchikey.nunique()),
        "unique_inchikeys_eligible": int(eligible.inchikey.nunique()),
        "active": int(eligible.binary_label.sum()),
        "inactive": int((1 - eligible.binary_label).sum()),
        "label_source_counts": {str(k): int(v) for k, v in eligible.label_source.value_counts().items()},
        "label_source_stratum_counts": {str(k): int(v) for k, v in eligible.label_source_stratum.value_counts().items()},
        "weight_counts": {str(k): int(v) for k, v in eligible.sample_weight_W1.value_counts().sort_index().items()},
        "scaffolds": int(eligible.scaffold_group.fillna("NO_SCAFFOLD").nunique()),
        "evaluation_weighting": "all samples weight 1; sample_weight is training-only",
    }
    return manifest, audit


def _assignment_objective(
    stats: np.ndarray,
    targets: np.ndarray,
    objective_weights: Mapping[str, float],
) -> float:
    # columns: n, positive, scaffolds, weight; rows: folds
    scale = np.maximum(targets, 1e-8)
    normalized = (stats - targets[None, :]) / scale[None, :]
    weights = np.asarray(
        [
            objective_weights["sample_count"],
            objective_weights["positive_count"],
            objective_weights["scaffold_count"],
            objective_weights["training_weight_total"],
        ],
        dtype=float,
    )
    base = float(np.sum(normalized**2 * weights[None, :]))
    concentration = float(np.sum((np.maximum(normalized[:, 0], 0.0)) ** 4))
    return base + float(objective_weights["large_scaffold_concentration"]) * concentration


def balanced_scaffold_assignment(
    eligible: pd.DataFrame,
    folds: int,
    seed: int,
    objective_weights: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Assign whole scaffold groups with a deterministic multi-objective greedy pass."""
    data = eligible.reset_index(drop=True).copy()
    data["_group"] = data.scaffold_group.fillna("NO_SCAFFOLD").astype(str)
    grouped = (
        data.groupby("_group", sort=False)
        .agg(n=("binary_label", "size"), positive=("binary_label", "sum"), weight=("sample_weight_W1", "sum"))
        .reset_index()
    )
    grouped["scaffolds"] = 1.0
    grouped["difficulty"] = (
        grouped["n"].astype(float)
        + 0.75 * (grouped["positive"] - grouped["n"] * data.binary_label.mean()).abs()
    )
    rng = np.random.default_rng(int(seed))
    grouped["jitter"] = rng.random(len(grouped))
    grouped = grouped.sort_values(["difficulty", "n", "jitter"], ascending=[False, False, True]).reset_index(drop=True)

    totals = np.asarray(
        [len(data), float(data.binary_label.sum()), float(len(grouped)), float(data.sample_weight_W1.sum())], dtype=float
    )
    targets = totals / float(folds)
    stats = np.zeros((folds, 4), dtype=float)
    assignment: dict[str, int] = {}
    for _, group in grouped.iterrows():
        vector = np.asarray([group.n, group.positive, 1.0, group.weight], dtype=float)
        scores = []
        for fold in range(folds):
            candidate = stats.copy()
            candidate[fold] += vector
            # Seeded fold preference affects only exact objective ties.
            tie = ((fold + int(seed)) % folds) * 1e-12
            scores.append(_assignment_objective(candidate, targets, objective_weights) + tie)
        selected = int(np.argmin(scores))
        assignment[str(group._group)] = selected
        stats[selected] += vector

    fold_ids = data._group.map(assignment).to_numpy(dtype=int)
    for fold in range(folds):
        selected = fold_ids == fold
        if not selected.any() or data.loc[selected, "binary_label"].nunique() != 2:
            raise RuntimeError(f"Assignment seed {seed} fold {fold} lacks both classes")
    if any(data.loc[fold_ids == fold, "_group"].nunique() == 0 for fold in range(folds)):
        raise RuntimeError("Assignment contains an empty scaffold fold")

    diagnostics = {
        "assignment_seed": int(seed),
        "objective": _assignment_objective(stats, targets, objective_weights),
        "target_samples": float(targets[0]),
        "target_positives": float(targets[1]),
        "target_scaffolds": float(targets[2]),
        "target_training_weight": float(targets[3]),
        "fold_stats": [
            {
                "fold": fold,
                "samples": int(stats[fold, 0]),
                "active": int(stats[fold, 1]),
                "inactive": int(stats[fold, 0] - stats[fold, 1]),
                "active_fraction": float(stats[fold, 1] / max(stats[fold, 0], 1)),
                "scaffolds": int(stats[fold, 2]),
                "training_weight_total": float(stats[fold, 3]),
                "largest_scaffold": int(data.loc[fold_ids == fold].groupby("_group").size().max()),
            }
            for fold in range(folds)
        ],
    }
    return fold_ids, diagnostics


def build_outer_and_inner_manifests(
    eligible: pd.DataFrame,
    assignment_seeds: Sequence[int],
    outer_folds: int,
    inner_folds: int,
    objective_weights: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    outer_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    inner_rows: list[dict[str, Any]] = []
    prior_vectors: list[np.ndarray] = []
    for assignment_seed in assignment_seeds:
        folds, diagnostics = balanced_scaffold_assignment(
            eligible, outer_folds, int(assignment_seed), objective_weights
        )
        if any(np.array_equal(folds, prior) for prior in prior_vectors):
            raise RuntimeError(f"Assignment seed {assignment_seed} duplicated an earlier complete assignment")
        prior_vectors.append(folds.copy())
        for local, (_, row) in enumerate(eligible.reset_index(drop=True).iterrows()):
            outer_rows.append(
                {
                    "assignment_seed": int(assignment_seed),
                    "binary_index": int(local),
                    "development_index": int(row.development_index),
                    "inchikey": str(row.inchikey),
                    "scaffold_group": str(row.scaffold_group),
                    "binary_label": int(row.binary_label),
                    "sample_weight_W1": float(row.sample_weight_W1),
                    "outer_fold": int(folds[local]),
                }
            )
        for fold_stats in diagnostics["fold_stats"]:
            balance_rows.append({"assignment_seed": int(assignment_seed), "objective": diagnostics["objective"], **fold_stats})

        for outer_fold in range(outer_folds):
            outer_train = np.flatnonzero(folds != outer_fold)
            local = eligible.iloc[outer_train].reset_index(drop=True)
            inner_seed = int(assignment_seed) * 1000 + int(outer_fold) * 37 + 17
            inner_ids, inner_diagnostics = balanced_scaffold_assignment(
                local, inner_folds, inner_seed, objective_weights
            )
            for local_position, binary_index in enumerate(outer_train):
                row = eligible.iloc[int(binary_index)]
                inner_rows.append(
                    {
                        "assignment_seed": int(assignment_seed),
                        "outer_fold": int(outer_fold),
                        "inner_assignment_seed": inner_seed,
                        "binary_index": int(binary_index),
                        "development_index": int(row.development_index),
                        "scaffold_group": str(row.scaffold_group),
                        "inner_fold": int(inner_ids[local_position]),
                    }
                )
            if any(value["samples"] == 0 for value in inner_diagnostics["fold_stats"]):
                raise RuntimeError("Empty inner fold")

    outer = pd.DataFrame(outer_rows)
    balance = pd.DataFrame(balance_rows)
    inner = pd.DataFrame(inner_rows)
    key = ["assignment_seed", "binary_index"]
    if outer.duplicated(key).any() or len(outer) != len(eligible) * len(assignment_seeds):
        raise RuntimeError("Outer assignment coverage is invalid")
    for assignment_seed in assignment_seeds:
        subset = outer[outer.assignment_seed.eq(assignment_seed)]
        for outer_fold in range(outer_folds):
            train_scaffolds = set(subset.loc[subset.outer_fold.ne(outer_fold), "scaffold_group"])
            valid_scaffolds = set(subset.loc[subset.outer_fold.eq(outer_fold), "scaffold_group"])
            if train_scaffolds & valid_scaffolds:
                raise RuntimeError("Outer scaffold leakage")
    return outer, balance, inner


def expected_calibration_error(labels: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=int)
    probability = np.asarray(probability, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1]
        )
        if mask.any():
            result += float(mask.mean()) * abs(float(probability[mask].mean()) - float(labels[mask].mean()))
    return float(result)


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if len(labels) != len(probability) or not np.isfinite(probability).all():
        raise ValueError("Invalid probability vector")
    prevalence = float(labels.mean())
    order = np.argsort(-probability, kind="stable")
    top10_count = max(1, int(math.ceil(0.10 * len(labels))))
    top10 = labels[order[:top10_count]]
    return {
        "pr_auc": float(average_precision_score(labels, probability)),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "ece": expected_calibration_error(labels, probability),
        "ef_at_10pct": float(top10.mean() / max(prevalence, 1e-12)),
        "active_at_20": float(labels[order[: min(20, len(labels))]].sum()),
    }


def threshold_metrics(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    predicted = (np.asarray(probability, dtype=float) >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "npv": float(tn / max(tn + fn, 1)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
    }


def choose_thresholds(
    labels: np.ndarray,
    probability: np.ndarray,
    minimum: float = 0.10,
    maximum: float = 0.90,
    step: float = 0.01,
) -> dict[str, float]:
    rows = []
    for threshold in np.round(np.arange(minimum, maximum + step / 2, step), 10):
        rows.append(threshold_metrics(labels, probability, float(threshold)))
    frame = pd.DataFrame(rows)
    selected: dict[str, float] = {}
    for metric in ("balanced_accuracy", "mcc", "f1"):
        best = frame[metric].max()
        candidates = frame.loc[np.isclose(frame[metric], best), "threshold"].to_numpy(float)
        selected[metric] = float(candidates[np.argmin(np.abs(candidates - 0.5))])
    return selected


def graph_feature_matrices(graphs: Sequence[Any]) -> dict[str, np.ndarray]:
    def matrix(attribute: str) -> np.ndarray:
        output = np.concatenate(
            [getattr(graph, attribute).detach().cpu().reshape(1, -1).numpy() for graph in graphs], axis=0
        ).astype(np.float32, copy=False)
        if not np.isfinite(output).all():
            raise RuntimeError(f"Non-finite graph features: {attribute}")
        return output

    values = {
        "rdkit": matrix("global_feat"),
        "morgan": matrix("morgan_fp"),
        "maccs": matrix("maccs_fp"),
    }
    expected = {"rdkit": 171, "morgan": 2048, "maccs": 167}
    for name, dimension in expected.items():
        if values[name].shape[1] != dimension:
            raise RuntimeError(f"{name} feature dimension mismatch: {values[name].shape}")
    values["flat"] = np.concatenate([values["morgan"], values["maccs"], values["rdkit"]], axis=1)
    return values


def deterministic_balanced_candidates(
    search_space: Mapping[str, Sequence[Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    keys = [
        "n_estimators", "max_features", "min_samples_leaf", "max_depth", "class_weight"
    ]
    all_values = [list(search_space[key]) for key in keys]
    universe = [dict(zip(keys, values)) for values in itertools.product(*all_values)]
    if limit >= len(universe):
        return universe
    rng = np.random.default_rng(int(seed))
    tie_order = rng.permutation(len(universe))
    tie_rank = {int(value): index for index, value in enumerate(tie_order)}
    selected_indices = [0]
    while len(selected_indices) < limit:
        best_index = None
        best_score: tuple[float, float, float] | None = None
        for index, candidate in enumerate(universe):
            if index in selected_indices:
                continue
            coverage = 0.0
            for key in keys:
                observed = {universe[value][key] for value in selected_indices}
                coverage += float(candidate[key] not in observed)
            distance = min(
                sum(candidate[key] != universe[value][key] for key in keys) for value in selected_indices
            )
            score = (coverage, float(distance), -float(tie_rank[index]))
            if best_score is None or score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        selected_indices.append(best_index)
    return [universe[index] for index in selected_indices]


def tree_complexity(params: Mapping[str, Any]) -> float:
    depth = 50 if params["max_depth"] is None else float(params["max_depth"])
    return (
        float(params["n_estimators"]) / 1000.0
        + depth / 100.0
        + 1.0 / float(params["min_samples_leaf"])
        + (0.1 if params["class_weight"] is not None else 0.0)
    )


def fit_tree_probability(
    estimator_name: str,
    params: Mapping[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    valid_x: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float, int, int]:
    started = time.perf_counter()
    estimator_class = ExtraTreesClassifier if estimator_name == "ExtraTrees" else RandomForestClassifier
    estimator = estimator_class(**dict(params), random_state=int(seed), n_jobs=-1)
    estimator.fit(train_x, train_y, sample_weight=np.asarray(train_weight, dtype=float))
    probability = np.clip(estimator.predict_proba(valid_x)[:, 1], 0.0, 1.0)
    if not np.isfinite(probability).all():
        raise RuntimeError("Tree estimator produced non-finite probability")
    nodes = int(sum(tree.tree_.node_count for tree in estimator.estimators_))
    return probability, time.perf_counter() - started, peak_rss_bytes(), nodes


def fit_fold_preprocessor(train_x: np.ndarray, train_y: np.ndarray) -> FoldSafePreprocessor:
    return FoldSafePreprocessor(reducer="none", scale=True, variance_threshold=1e-8).fit(train_x, train_y)


@dataclass(frozen=True)
class DeepSpec:
    family: str
    architecture: ArchitectureSpec
    max_epochs: int
    candidate_epochs: tuple[int, ...]
    batch_size: int
    weight_decay: float
    gradient_clip: float
    warmup_epochs: int
    minimum_lr_ratio: float
    embedding_key: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["architecture"] = self.architecture.to_dict()
        return payload


def build_deep_specs(config: Mapping[str, Any]) -> dict[str, DeepSpec]:
    deep = config["deep_training"]
    epochs = tuple(int(value) for value in deep["candidate_epochs"])

    def make(
        family: str,
        section: str,
        model_config: PharmaFusionConfig,
        name: str,
        embedding_key: str,
    ) -> DeepSpec:
        values = deep[section]
        architecture = ArchitectureSpec(
            name=name,
            stage="binary_activity_stable",
            config=model_config,
            initialization="pretrained",
            tuning_mode=str(values["tuning_mode"]),
            backbone_lr=float(values["encoder_lr"]),
            new_module_lr=float(values["new_module_lr"]),
            freeze_epochs=int(values["freeze_epochs"]),
        )
        return DeepSpec(
            family=family,
            architecture=architecture,
            max_epochs=int(deep["max_epochs"]),
            candidate_epochs=epochs,
            batch_size=int(values["batch_size"]),
            weight_decay=float(values["weight_decay"]),
            gradient_clip=float(deep["gradient_clip"]),
            warmup_epochs=int(values["warmup_epochs"]),
            minimum_lr_ratio=float(deep["minimum_lr_ratio"]),
            embedding_key=embedding_key,
        )

    base0 = deep["M0"]
    m0_config = PharmaFusionConfig(
        use_jk=False,
        pooling="mean",
        use_adapter=False,
        use_gps=False,
        fusion="none",
        head_type="mlp",
        encoder_dropout=float(base0["encoder_dropout"]),
        head_dropout=float(base0["head_dropout"]),
    )
    fusion = deep["M3_M4"]
    fusion_config = PharmaFusionConfig(
        use_jk=False,
        pooling="mean",
        use_adapter=False,
        use_gps=False,
        fusion="gated",
        head_type="mlp",
        encoder_dropout=float(fusion["encoder_dropout"]),
        head_dropout=float(fusion["head_dropout"]),
    )
    adapter = deep["M5"]
    adapter_config = PharmaFusionConfig(
        use_jk=False,
        pooling="mean",
        use_adapter=True,
        adapter_dim=int(adapter["adapter_dim"]),
        adapter_scale=float(adapter["adapter_scale"]),
        use_gps=False,
        fusion="none",
        head_type="mlp",
        encoder_dropout=float(adapter["encoder_dropout"]),
        head_dropout=float(adapter["head_dropout"]),
    )
    return {
        "M0": make("m0", "M0", m0_config, MODEL_NAMES["M0"], "E0"),
        "GATED": make("gated", "M3_M4", fusion_config, "Shared GatedFusion representation", "E4"),
        "M5": make("m5", "M5", adapter_config, MODEL_NAMES["M5"], "E1"),
    }


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _changed_values(before: Mapping[str, torch.Tensor], module: torch.nn.Module) -> int:
    state = module.state_dict()
    return int(
        sum(
            int(torch.count_nonzero(value != state[name].detach().cpu()).item())
            for name, value in before.items()
            if name in state
        )
    )


def train_deep_snapshots(
    spec: DeepSpec,
    graphs: Sequence[Any],
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    sample_weights: np.ndarray,
    checkpoint: Path,
    seed: int,
    capture_train_embeddings: bool,
    device_name: str,
) -> tuple[dict[int, dict[str, np.ndarray]], pd.DataFrame, dict[str, Any]]:
    """Train once to max epoch and capture only pre-locked epoch snapshots."""
    device = resolve_device(device_name)
    set_all_seeds(seed)
    descriptor_scaler = None
    descriptor_audit: dict[str, Any] = {}
    if spec.architecture.config.fusion != "none":
        mean, scale, descriptor_audit = fit_descriptor_scaler(graphs, train_indices)
        descriptor_scaler = (mean, scale)
    model, construction = build_model(spec.architecture, checkpoint, int(seed), descriptor_scaler)
    model.to(device)
    initial_backbone = _clone_state(model.gine_encoder)
    backbone_parameters = list(model.gine_encoder.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    new_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in backbone_ids and parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [{"params": new_parameters, "lr": spec.architecture.new_module_lr, "group_name": "new", "weight_decay": spec.weight_decay}]
    )
    if spec.architecture.tuning_mode == "full":
        optimizer.add_param_group(
            {"params": backbone_parameters, "lr": spec.architecture.backbone_lr, "group_name": "backbone", "weight_decay": spec.weight_decay}
        )

    weights = np.asarray(sample_weights, dtype=np.float32)
    if weights.shape[0] != len(graphs):
        raise ValueError("Global sample-weight vector length mismatch")
    for index in np.unique(np.concatenate([train_indices, evaluation_indices])):
        graphs[int(index)].binary_sample_weight = torch.tensor([float(weights[int(index)])], dtype=torch.float32)

    snapshots: dict[int, dict[str, np.ndarray]] = {}
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    optimizer_step_time = 0.0
    nonzero_backbone_batches = 0
    backbone_gradient_batches = 0
    started = time.perf_counter()

    for epoch in range(1, spec.max_epochs + 1):
        if spec.architecture.tuning_mode == "staged" and epoch == spec.architecture.freeze_epochs + 1:
            model.set_backbone_mode("full")
            optimizer.add_param_group(
                {
                    "params": list(model.gine_encoder.parameters()),
                    "lr": spec.architecture.backbone_lr,
                    "group_name": "backbone",
                    "weight_decay": spec.weight_decay,
                }
            )
        new_factor, new_phase = cosine_factor(epoch, spec.max_epochs, spec.warmup_epochs, spec.minimum_lr_ratio)
        if spec.architecture.tuning_mode == "staged":
            local_epoch = epoch - spec.architecture.freeze_epochs
            if local_epoch <= 0:
                backbone_factor, backbone_phase = 0.0, "frozen"
            else:
                backbone_factor, backbone_phase = cosine_factor(
                    local_epoch,
                    max(spec.max_epochs - spec.architecture.freeze_epochs, 1),
                    spec.warmup_epochs,
                    spec.minimum_lr_ratio,
                )
        elif spec.architecture.tuning_mode in {"frozen", "adapter"}:
            backbone_factor, backbone_phase = 0.0, "frozen"
        else:
            backbone_factor, backbone_phase = new_factor, new_phase
        for group in optimizer.param_groups:
            group["lr"] = (
                spec.architecture.new_module_lr * new_factor
                if group["group_name"] == "new"
                else spec.architecture.backbone_lr * backbone_factor
            )

        set_all_seeds(int(seed) + epoch * 100003)
        rng = np.random.default_rng(int(seed) + epoch * 100003)
        ordered = np.asarray(train_indices, dtype=int)[rng.permutation(len(train_indices))]
        loader = DataLoader([graphs[int(index)] for index in ordered], batch_size=spec.batch_size, shuffle=False)
        model.train()
        if spec.architecture.tuning_mode in {"frozen", "adapter"} or (
            spec.architecture.tuning_mode == "staged" and epoch <= spec.architecture.freeze_epochs
        ):
            model.gine_encoder.eval()
        epoch_loss = 0.0
        epoch_n = 0
        epoch_backbone_norm = 0.0
        epoch_new_norm = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)["logits"]
            target = batch.y_cls.view(-1).float()
            batch_weight = batch.binary_sample_weight.view(-1).float()
            loss_values = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            loss = torch.mean(batch_weight * loss_values)
            loss.backward()
            backbone_norm_sq = sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in backbone_parameters
                if parameter.grad is not None
            )
            new_norm_sq = sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in new_parameters
                if parameter.grad is not None
            )
            if any(parameter.requires_grad for parameter in backbone_parameters):
                backbone_gradient_batches += 1
                nonzero_backbone_batches += int(backbone_norm_sq > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), spec.gradient_clip)
            step_started = time.perf_counter()
            optimizer.step()
            optimizer_step_time += time.perf_counter() - step_started
            optimizer_steps += 1
            epoch_loss += float(loss.detach().cpu()) * int(target.numel())
            epoch_n += int(target.numel())
            epoch_backbone_norm += math.sqrt(backbone_norm_sq)
            epoch_new_norm += math.sqrt(new_norm_sq)

        lrs = {str(group["group_name"]): float(group["lr"]) for group in optimizer.param_groups}
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss / max(epoch_n, 1),
                "new_module_lr": lrs.get("new", 0.0),
                "encoder_lr": lrs.get("backbone", 0.0),
                "new_scheduler_phase": new_phase,
                "encoder_scheduler_phase": backbone_phase,
                "encoder_frozen": int(not any(parameter.requires_grad for parameter in backbone_parameters)),
                "encoder_gradient_norm": epoch_backbone_norm / max(len(loader), 1),
                "new_module_gradient_norm": epoch_new_norm / max(len(loader), 1),
            }
        )
        if epoch in spec.candidate_epochs:
            evaluation = predict_model(model, graphs, evaluation_indices, spec.batch_size, device)
            snapshot = {
                "evaluation_probability": evaluation["probability"].astype(np.float64),
                "evaluation_embedding": evaluation["embeddings"][spec.embedding_key].astype(np.float32),
            }
            if capture_train_embeddings:
                training = predict_model(model, graphs, train_indices, spec.batch_size, device)
                snapshot["training_embedding"] = training["embeddings"][spec.embedding_key].astype(np.float32)
            snapshots[int(epoch)] = snapshot

    audit = {
        **construction,
        "deep_spec": spec.to_dict(),
        "training_sample_count": int(len(train_indices)),
        "evaluation_sample_count": int(len(evaluation_indices)),
        "training_indices_sha256": stable_hash(sorted(int(v) for v in train_indices)),
        "evaluation_indices_sha256": stable_hash(sorted(int(v) for v in evaluation_indices)),
        "evaluation_labels_used_for_gradient_updates": False,
        "weighted_bce_formula": "mean(sample_weight * BCEWithLogitsLoss(reduction=none))",
        "pos_weight_used": False,
        "optimizer_updates": int(optimizer_steps),
        "mean_optimizer_step_seconds": float(optimizer_step_time / max(optimizer_steps, 1)),
        "training_seconds": float(time.perf_counter() - started),
        "peak_memory_bytes": int(peak_rss_bytes()),
        "encoder_changed_values": _changed_values(initial_backbone, model.gine_encoder),
        "encoder_gradient_nonzero_fraction": float(nonzero_backbone_batches / max(backbone_gradient_batches, 1)),
        "final_model_state_sha256": parameter_hash(model.parameters()),
    }
    if spec.architecture.tuning_mode in {"frozen", "adapter"} and audit["encoder_changed_values"] != 0:
        raise RuntimeError("Frozen encoder parameters changed")
    if set(snapshots) != set(spec.candidate_epochs):
        raise RuntimeError("Not all pre-locked deep epochs were captured")
    return snapshots, pd.DataFrame(history), audit


def select_epoch(
    labels: np.ndarray,
    epoch_probability: Mapping[int, np.ndarray],
    tie_tolerance: float,
) -> tuple[int, list[dict[str, Any]]]:
    rows = [{"epoch": int(epoch), **probability_metrics(labels, probability)} for epoch, probability in epoch_probability.items()]
    best = max(row["pr_auc"] for row in rows)
    eligible = [row for row in rows if row["pr_auc"] >= best - float(tie_tolerance)]
    selected = min(eligible, key=lambda row: (row["epoch"], row["brier"]))
    return int(selected["epoch"]), rows


def validate_split_isolation(
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    eligible: pd.DataFrame,
) -> dict[str, int]:
    train = eligible.iloc[np.asarray(train_indices, dtype=int)]
    valid = eligible.iloc[np.asarray(valid_indices, dtype=int)]
    result = {
        "index_overlap": int(len(set(train_indices) & set(valid_indices))),
        "inchikey_overlap": int(len(set(train.inchikey.astype(str)) & set(valid.inchikey.astype(str)))),
        "scaffold_overlap": int(
            len(set(train.scaffold_group.fillna("NO_SCAFFOLD").astype(str)) & set(valid.scaffold_group.fillna("NO_SCAFFOLD").astype(str)))
        ),
    }
    if any(result.values()):
        raise RuntimeError(f"Split leakage: {result}")
    return result


def deep_forward_backward_reload_smoke(
    spec: DeepSpec,
    graphs: Sequence[Any],
    train_indices: np.ndarray,
    evaluation_indices: np.ndarray,
    sample_weights: np.ndarray,
    checkpoint: Path,
    seed: int,
    device_name: str,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """One weighted-BCE optimizer step followed by exact state reload QA."""
    device = resolve_device(device_name)
    set_all_seeds(seed)
    descriptor_scaler = None
    if spec.architecture.config.fusion != "none":
        mean, scale, _ = fit_descriptor_scaler(graphs, train_indices)
        descriptor_scaler = (mean, scale)
    model, construction = build_model(spec.architecture, checkpoint, seed, descriptor_scaler)
    model.to(device)
    initial_encoder = _clone_state(model.gine_encoder)
    for index in np.unique(np.concatenate([train_indices, evaluation_indices])):
        graphs[int(index)].binary_sample_weight = torch.tensor(
            [float(sample_weights[int(index)])], dtype=torch.float32
        )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=spec.architecture.new_module_lr, weight_decay=spec.weight_decay)
    batch = next(iter(DataLoader([graphs[int(index)] for index in train_indices], batch_size=len(train_indices), shuffle=False))).to(device)
    model.train()
    if spec.architecture.tuning_mode in {"frozen", "adapter", "staged"}:
        model.gine_encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    logits = model(batch)["logits"]
    targets = batch.y_cls.view(-1).float()
    weights = batch.binary_sample_weight.view(-1).float()
    per_sample = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = torch.mean(weights * per_sample)
    loss.backward()
    gradient_norm = math.sqrt(
        sum(float(parameter.grad.detach().pow(2).sum().item()) for parameter in parameters if parameter.grad is not None)
    )
    optimizer.step()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "spec": spec.to_dict()}, checkpoint_path)

    evaluation_batch = next(iter(DataLoader([graphs[int(index)] for index in evaluation_indices], batch_size=len(evaluation_indices), shuffle=False))).to(device)
    model.eval()
    with torch.no_grad():
        before = model(evaluation_batch)["logits"].detach().cpu()
    reloaded, _ = build_model(spec.architecture, checkpoint, seed, descriptor_scaler)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded.load_state_dict(payload["model_state_dict"], strict=True)
    reloaded.to(device).eval()
    with torch.no_grad():
        after = reloaded(evaluation_batch)["logits"].detach().cpu()
    tolerance = 1e-6 if device.type == "cpu" else 2e-5
    reload_max_abs_diff = float(torch.max(torch.abs(before - after)).item())
    encoder_changed = _changed_values(initial_encoder, model.gine_encoder)
    if spec.architecture.tuning_mode in {"frozen", "adapter", "staged"} and encoder_changed != 0:
        raise RuntimeError("Frozen smoke-test encoder changed")
    if not torch.isfinite(loss) or gradient_norm <= 0 or reload_max_abs_diff > tolerance:
        raise RuntimeError(
            f"Deep smoke failed: loss={loss.item()} grad={gradient_norm} reload_diff={reload_max_abs_diff}"
        )
    return {
        "family": spec.family,
        "device": str(device),
        "weighted_bce_loss": float(loss.detach().cpu()),
        "gradient_norm": float(gradient_norm),
        "encoder_changed_values": int(encoder_changed),
        "checkpoint_reload_max_abs_diff": reload_max_abs_diff,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "finite_probabilities": bool(torch.isfinite(torch.sigmoid(after)).all()),
        "probabilities_within_unit_interval": bool(((torch.sigmoid(after) >= 0) & (torch.sigmoid(after) <= 1)).all()),
        "parameter_audit": construction,
    }
