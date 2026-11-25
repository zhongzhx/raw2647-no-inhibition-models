"""Unified RAW264.7 pretraining-redesign downstream experiment runner.

The runner keeps the exposed legacy test set out of all model selection. It
creates development-only OOF predictions for three protocols, strict branch
ablations, linear probes, sample-efficiency runs, similarity strata, paired
bootstrap comparisons, and an explicitly exploratory legacy-test evaluation.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Scoring import Scoring
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "models", ROOT / "utils", ROOT / "data_processing", ROOT / "training"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gine_v2 import (
    GINEEncoder,
    GraphOnlyEncoder,
    IndependentHybridEncoder,
    MolecularClassifier,
    load_gine_checkpoint,
)
from pipeline_utils import load_graphs, set_seed


SEEDS = (13, 42, 2026)
PROTOCOLS = ("stratified", "publication", "scaffold")
MAIN_MODELS = (
    "RF",
    "RBF-SVM",
    "Logistic Regression",
    "Flat-only",
    "Scratch GINE",
    "Original-pretrained GINE",
    "Improved-pretrained GINE",
    "Scratch GINE-hybrid",
    "Original-pretrained GINE-hybrid",
    "Improved-pretrained GINE-hybrid",
)


@dataclass(frozen=True)
class NeuralSpec:
    name: str
    branches: tuple[str, ...]
    checkpoint: str = ""
    freeze_encoder: bool = False


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bemis_murcko(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "<invalid>"
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "<acyclic>"


def labels(graphs: list) -> np.ndarray:
    return np.asarray([int(graph.y_cls.view(-1)[0]) for graph in graphs], dtype=int)


def flat_matrix(graphs: list) -> np.ndarray:
    return np.vstack(
        [
            torch.cat(
                [graph.global_feat.view(-1), graph.morgan_fp.view(-1), graph.maccs_fp.view(-1)]
            ).float().numpy()
            for graph in graphs
        ]
    )


def split_metadata(review: pd.DataFrame) -> pd.DataFrame:
    meta = review.copy()
    meta["source_doi"] = meta["source_doi"].fillna("").astype(str).str.strip()
    meta.loc[meta["source_doi"].eq(""), "source_doi"] = [
        f"missing-publication-{idx}" for idx in meta.index[meta["source_doi"].eq("")]
    ]
    meta["bemis_murcko_scaffold"] = [bemis_murcko(s) for s in meta["canonical_smiles"].astype(str)]
    return meta


def fold_assignments(
    y: np.ndarray,
    meta: pd.DataFrame,
    protocol: str,
    seed: int,
    folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    dummy = np.zeros((len(y), 1))
    if protocol == "stratified":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        return list(splitter.split(dummy, y))
    group_column = "source_doi" if protocol == "publication" else "bemis_murcko_scaffold"
    groups = meta[group_column].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignments = list(splitter.split(dummy, y, groups))
    for train_idx, valid_idx in assignments:
        overlap = set(groups[train_idx]) & set(groups[valid_idx])
        if overlap:
            raise RuntimeError(f"{protocol} leakage: {len(overlap)} groups cross a fold")
    return assignments


def sample_weights(batch: Any) -> torch.Tensor:
    weights = torch.ones_like(batch.y_cls.view(-1).float())
    if hasattr(batch, "is_censored_gt50"):
        censored = batch.is_censored_gt50.view(-1).bool()
        threshold = batch.censored_threshold_uM.view(-1).float()
        weights = torch.where(censored & torch.isfinite(threshold) & (threshold >= 50), 0.8, weights)
        weights = torch.where(censored & (~torch.isfinite(threshold) | (threshold < 50)), 0.5, weights)
    return weights


def standardized_graphs(graphs: list, train_indices: Iterable[int]) -> list:
    train_values = torch.cat([graphs[idx].global_feat.float() for idx in train_indices], dim=0)
    mean = train_values.mean(0, keepdim=True)
    std = train_values.std(0, unbiased=False, keepdim=True).clamp(min=1e-6)
    result = []
    for graph in graphs:
        item = copy.copy(graph)
        item.global_feat = (graph.global_feat.float() - mean) / std
        result.append(item)
    return result


def make_neural_model(spec: NeuralSpec, graph: Any, args: argparse.Namespace) -> tuple[nn.Module, dict[str, Any] | None]:
    gine = GINEEncoder(args.hidden_dim, args.num_layers, args.dropout, use_layer_norm=True)
    audit = None
    if spec.checkpoint:
        audit = load_gine_checkpoint(gine, spec.checkpoint, min_load_ratio=0.95).to_dict()
    if spec.freeze_encoder:
        for parameter in gine.parameters():
            parameter.requires_grad = False
    if spec.branches == ("gine",):
        encoder = GraphOnlyEncoder(gine)
        output_dim = encoder.output_dim
    else:
        encoder = IndependentHybridEncoder(
            gine,
            global_dim=int(graph.global_feat.shape[-1]),
            morgan_dim=int(graph.morgan_fp.shape[-1]),
            maccs_dim=int(graph.maccs_fp.shape[-1]),
            branch_dim=args.branch_dim,
            dropout=args.dropout,
            enabled_branches=spec.branches,
            fusion=args.fusion,
        )
        output_dim = encoder.output_dim
    return MolecularClassifier(encoder, output_dim, args.hidden_dim, args.dropout), audit


@torch.no_grad()
def predict_neural(model: nn.Module, graphs: list, indices: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    loader = DataLoader([graphs[idx] for idx in indices], batch_size=batch_size, shuffle=False)
    values: list[float] = []
    for batch in loader:
        logits = model(batch.to(device))
        values.extend(torch.sigmoid(logits).detach().cpu().tolist())
    return np.asarray(values, dtype=float)


def train_neural_fold(
    spec: NeuralSpec,
    graphs: list,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    checkpoint_output: Path | None = None,
) -> tuple[np.ndarray, dict[str, Any], nn.Module]:
    set_seed(seed)
    device = resolve_device(args.device)
    fold_graphs = standardized_graphs(graphs, train_idx)
    model, transfer = make_neural_model(spec, fold_graphs[train_idx[0]], args)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    y_train = labels([fold_graphs[idx] for idx in train_idx])
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)],
        device=device,
        dtype=torch.float32,
    )
    loader = DataLoader([fold_graphs[idx] for idx in train_idx], batch_size=args.batch_size, shuffle=True)
    best_auc = -float("inf")
    best_state = None
    best_epoch = 0
    patience = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            y = batch.y_cls.view(-1).float()
            loss_values = F.binary_cross_entropy_with_logits(logits, y, reduction="none", pos_weight=pos_weight)
            weights = sample_weights(batch).to(device)
            loss = (loss_values * weights).sum() / weights.sum().clamp(min=1e-8)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        val_pred = predict_neural(model, fold_graphs, valid_idx, device, args.batch_size)
        val_y = labels([fold_graphs[idx] for idx in valid_idx])
        auc = roc_auc_score(val_y, val_pred) if len(np.unique(val_y)) == 2 else 0.5
        if auc > best_auc + 1e-6:
            best_auc = float(auc)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {spec.name}")
    model.load_state_dict(best_state)
    model = model.to(device)
    prediction = predict_neural(model, fold_graphs, valid_idx, device, args.batch_size)
    if checkpoint_output is not None:
        checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_state, "spec": spec.__dict__, "best_epoch": best_epoch}, checkpoint_output)
    return prediction, {
        "best_epoch": best_epoch,
        "best_valid_roc_auc": best_auc,
        "elapsed_seconds": time.time() - started,
        "transfer": transfer,
    }, model


def classical_models(seed: int) -> dict[str, Any]:
    return {
        "RF": RandomForestClassifier(
            n_estimators=500, max_features="sqrt", min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1,
        ),
        "RBF-SVM": make_pipeline(StandardScaler(), SVC(C=2.0, gamma="scale", probability=True, class_weight="balanced", random_state=seed)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced", random_state=seed)),
    }


def bedroc(y: np.ndarray, score: np.ndarray, alpha: float = 20.0) -> float:
    ranked = sorted(zip(score.tolist(), y.tolist()), key=lambda item: item[0], reverse=True)
    try:
        return float(Scoring.CalcBEDROC(ranked, 1, alpha))
    except Exception:
        return float("nan")


def ranking_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    order = np.argsort(-score)
    prevalence = float(y.mean())
    result: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, score)),
        "mcc": float(matthews_corrcoef(y, score >= 0.5)),
        "brier": float(brier_score_loss(y, score)),
        "bedroc": bedroc(y, score),
    }
    for count in (5, 10, 20):
        k = min(count, len(y))
        result[f"precision_at_{count}"] = float(y[order[:k]].mean())
        result[f"actives_at_{count}"] = float(y[order[:k]].sum())
        result[f"random_expected_actives_at_{count}"] = float(k * prevalence)
    k10 = max(1, int(math.ceil(0.10 * len(y))))
    result["precision_at_10pct"] = float(y[order[:k10]].mean())
    result["recall_at_10pct"] = float(y[order[:k10]].sum() / max(y.sum(), 1))
    for fraction in (0.05, 0.10, 0.20):
        k = max(1, int(math.ceil(fraction * len(y))))
        precision = float(y[order[:k]].mean())
        result[f"ef_at_{int(fraction * 100)}pct"] = precision / max(prevalence, 1e-12)
    return result


def morgan_fps(smiles_values: Iterable[str]) -> list:
    return [AllChem.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles_values]


def max_train_similarity(fps: list, train_idx: np.ndarray, valid_idx: np.ndarray) -> dict[int, float]:
    train_fps = [fps[idx] for idx in train_idx]
    return {
        int(idx): float(max(DataStructs.BulkTanimotoSimilarity(fps[idx], train_fps)))
        for idx in valid_idx
    }


def bootstrap_delta(y: np.ndarray, a: np.ndarray, b: np.ndarray, metric: str, samples: int, seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(samples):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        ma = ranking_metrics(y[idx], a[idx])[metric]
        mb = ranking_metrics(y[idx], b[idx])[metric]
        values.append(ma - mb)
    array = np.asarray(values)
    observed = ranking_metrics(y, a)[metric] - ranking_metrics(y, b)[metric]
    return {
        "delta": float(observed),
        "ci_low": float(np.quantile(array, 0.025)),
        "ci_high": float(np.quantile(array, 0.975)),
        "bootstrap_samples_used": int(len(array)),
    }


def run_oof(
    graphs: list,
    meta: pd.DataFrame,
    args: argparse.Namespace,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = labels(graphs)
    x = flat_matrix(graphs)
    fps = morgan_fps(meta["canonical_smiles"].astype(str))
    improved = args.improved_checkpoint
    specs = [
        NeuralSpec("Flat-only", ("rdkit", "morgan", "maccs")),
        NeuralSpec("Scratch GINE", ("gine",)),
        NeuralSpec("Original-pretrained GINE", ("gine",), args.original_checkpoint),
        NeuralSpec("Improved-pretrained GINE", ("gine",), improved),
        NeuralSpec("Scratch GINE-hybrid", ("gine", "rdkit", "morgan", "maccs")),
        NeuralSpec("Original-pretrained GINE-hybrid", ("gine", "rdkit", "morgan", "maccs"), args.original_checkpoint),
        NeuralSpec("Improved-pretrained GINE-hybrid", ("gine", "rdkit", "morgan", "maccs"), improved),
    ]
    oof_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    protocols = ("stratified",) if args.smoke else PROTOCOLS
    seeds = (13,) if args.smoke else SEEDS
    folds = 2 if args.smoke else 5
    for protocol in protocols:
        for seed in seeds:
            assignments = fold_assignments(y, meta, protocol, seed, folds)
            for fold, (train_idx, valid_idx) in enumerate(assignments):
                similarities = max_train_similarity(fps, train_idx, valid_idx)
                for idx in valid_idx:
                    assignment_rows.append({
                        "protocol": protocol, "seed": seed, "fold": fold,
                        "sample_index": int(idx), "partition": "validation",
                        "source_doi": meta.iloc[idx]["source_doi"],
                        "bemis_murcko_scaffold": meta.iloc[idx]["bemis_murcko_scaffold"],
                    })
                for name, estimator in classical_models(seed).items():
                    fitted = clone(estimator).fit(x[train_idx], y[train_idx])
                    pred = fitted.predict_proba(x[valid_idx])[:, 1]
                    metrics = ranking_metrics(y[valid_idx], pred)
                    fold_rows.append({"protocol": protocol, "seed": seed, "fold": fold, "model": name, **metrics})
                    for idx, score in zip(valid_idx, pred):
                        oof_rows.append({"protocol": protocol, "seed": seed, "fold": fold, "model": name, "sample_index": int(idx), "label": int(y[idx]), "score": float(score), "max_train_tanimoto": similarities[int(idx)]})
                for model_index, spec in enumerate(specs):
                    pred, training, _ = train_neural_fold(
                        spec, graphs, train_idx, valid_idx,
                        seed + fold * 100 + model_index * 1000, args,
                    )
                    metrics = ranking_metrics(y[valid_idx], pred)
                    fold_rows.append({"protocol": protocol, "seed": seed, "fold": fold, "model": spec.name, **metrics, "best_epoch": training["best_epoch"], "elapsed_seconds": training["elapsed_seconds"], "transfer_load_ratio": (training["transfer"] or {}).get("load_ratio", np.nan)})
                    for idx, score in zip(valid_idx, pred):
                        oof_rows.append({"protocol": protocol, "seed": seed, "fold": fold, "model": spec.name, "sample_index": int(idx), "label": int(y[idx]), "score": float(score), "max_train_tanimoto": similarities[int(idx)]})
                print(f"completed {protocol} seed={seed} fold={fold}", flush=True)
    oof = pd.DataFrame(oof_rows)
    folds_df = pd.DataFrame(fold_rows)
    assignments_df = pd.DataFrame(assignment_rows)
    oof.to_csv(output / "oof_predictions_all.csv", index=False)
    for protocol, frame in oof.groupby("protocol"):
        frame.to_csv(output / f"oof_predictions_{protocol}.csv", index=False)
    folds_df.to_csv(output / "fold_metrics.csv", index=False)
    assignments_df.to_csv(output / "fold_assignments.csv", index=False)
    return oof, folds_df, assignments_df


def summarize_oof(oof: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    for (protocol, seed, model), frame in oof.groupby(["protocol", "seed", "model"]):
        metrics = ranking_metrics(frame["label"].to_numpy(), frame["score"].to_numpy())
        rows.append({"protocol": protocol, "seed": seed, "model": model, **metrics})
    per_seed = pd.DataFrame(rows)
    metric_columns = [column for column in per_seed.columns if column not in {"protocol", "seed", "model"}]
    summary = per_seed.groupby(["protocol", "model"])[metric_columns].agg(["mean", "std"]).reset_index()
    summary.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    per_seed.to_csv(output / "model_comparison_per_seed.csv", index=False)
    summary.to_csv(output / "model_comparison.csv", index=False)
    return summary


def paired_comparisons(oof: pd.DataFrame, output: Path, bootstrap_samples: int) -> pd.DataFrame:
    comparisons = [
        ("Original-pretrained GINE", "Scratch GINE", "original_pretrained_vs_scratch_gine"),
        ("Improved-pretrained GINE", "Scratch GINE", "improved_pretrained_vs_scratch_gine"),
        ("Improved-pretrained GINE-hybrid", "Scratch GINE-hybrid", "improved_pretrained_vs_scratch_hybrid"),
        ("Improved-pretrained GINE-hybrid", "Flat-only", "full_hybrid_vs_flat_only"),
        ("Improved-pretrained GINE", "Original-pretrained GINE", "improved_vs_original_pretraining_graph"),
        ("Improved-pretrained GINE-hybrid", "Original-pretrained GINE-hybrid", "improved_vs_original_pretraining_hybrid"),
    ]
    rows = []
    for protocol in oof["protocol"].unique():
        protocol_frame = oof[oof["protocol"] == protocol]
        averaged = protocol_frame.groupby(["model", "sample_index", "label"], as_index=False)["score"].mean()
        for model_a, model_b, name in comparisons:
            a = averaged[averaged.model == model_a].sort_values("sample_index")
            b = averaged[averaged.model == model_b].sort_values("sample_index")
            if len(a) == 0 or len(b) == 0 or not np.array_equal(a.sample_index.to_numpy(), b.sample_index.to_numpy()):
                continue
            y = a.label.to_numpy()
            for metric in ("roc_auc", "pr_auc", "precision_at_10pct", "ef_at_10pct"):
                result = bootstrap_delta(y, a.score.to_numpy(), b.score.to_numpy(), metric, bootstrap_samples, 2026)
                rows.append({"protocol": protocol, "comparison": name, "model_a": model_a, "model_b": model_b, "metric": metric, **result})
    result = pd.DataFrame(rows)
    result.to_csv(output / "bootstrap_comparison.csv", index=False)
    result[result.comparison.str.contains("pretrained")].to_csv(output / "pretraining_effect.csv", index=False)
    return result


def branch_ablation(graphs: list, meta: pd.DataFrame, args: argparse.Namespace, output: Path) -> pd.DataFrame:
    y = labels(graphs)
    rows = []
    predictions = []
    specs = [
        NeuralSpec("full hybrid", ("gine", "rdkit", "morgan", "maccs"), args.improved_checkpoint),
        NeuralSpec("remove GINE", ("rdkit", "morgan", "maccs"), args.improved_checkpoint),
        NeuralSpec("remove RDKit", ("gine", "morgan", "maccs"), args.improved_checkpoint),
        NeuralSpec("remove Morgan", ("gine", "rdkit", "maccs"), args.improved_checkpoint),
        NeuralSpec("remove MACCS", ("gine", "rdkit", "morgan"), args.improved_checkpoint),
        NeuralSpec("graph-only", ("gine",), args.improved_checkpoint),
        NeuralSpec("flat-only", ("rdkit", "morgan", "maccs"), args.improved_checkpoint),
    ]
    seeds = (13,) if args.smoke else SEEDS
    folds = 2 if args.smoke else 5
    for seed in seeds:
        for fold, (train_idx, valid_idx) in enumerate(fold_assignments(y, meta, "stratified", seed, folds)):
            for spec_idx, spec in enumerate(specs):
                pred, training, _ = train_neural_fold(spec, graphs, train_idx, valid_idx, seed + fold * 100 + spec_idx * 1000, args)
                for idx, score in zip(valid_idx, pred):
                    predictions.append({"seed": seed, "fold": fold, "model": spec.name, "sample_index": int(idx), "label": int(y[idx]), "score": float(score)})
    pred_df = pd.DataFrame(predictions)
    for (seed, model), frame in pred_df.groupby(["seed", "model"]):
        rows.append({"seed": seed, "model": model, **ranking_metrics(frame.label.to_numpy(), frame.score.to_numpy())})
    result = pd.DataFrame(rows)
    result.to_csv(output / "branch_ablation.csv", index=False)
    pred_df.to_csv(output / "oof_predictions_branch_ablation.csv", index=False)
    return result


def frozen_probe(graphs: list, meta: pd.DataFrame, args: argparse.Namespace, output: Path) -> pd.DataFrame:
    y = labels(graphs)
    specs = [
        NeuralSpec("random frozen GINE", ("gine",), "", True),
        NeuralSpec("original pretrained frozen GINE", ("gine",), args.original_checkpoint, True),
        NeuralSpec("improved pretrained frozen GINE", ("gine",), args.improved_checkpoint, True),
    ]
    rows = []
    seeds = (13,) if args.smoke else SEEDS
    folds = 2 if args.smoke else 5
    for seed in seeds:
        predictions = {spec.name: np.zeros(len(y)) for spec in specs}
        for fold, (train_idx, valid_idx) in enumerate(fold_assignments(y, meta, "stratified", seed, folds)):
            for spec_idx, spec in enumerate(specs):
                pred, _, _ = train_neural_fold(spec, graphs, train_idx, valid_idx, seed + fold * 100 + spec_idx * 1000, args)
                predictions[spec.name][valid_idx] = pred
        for name, pred in predictions.items():
            rows.append({"seed": seed, "model": name, **ranking_metrics(y, pred)})
    result = pd.DataFrame(rows)
    result.to_csv(output / "frozen_linear_probe.csv", index=False)
    return result


def sample_efficiency(graphs: list, meta: pd.DataFrame, args: argparse.Namespace, output: Path) -> pd.DataFrame:
    y = labels(graphs)
    rows = []
    fractions = (0.10, 0.25) if args.smoke else (0.10, 0.25, 0.50, 0.75, 1.0)
    seeds = (13,) if args.smoke else SEEDS
    for seed in seeds:
        train_pool, valid_idx = fold_assignments(y, meta, "stratified", seed, 5)[0]
        for fraction in fractions:
            if fraction < 1:
                splitter = StratifiedShuffleSplit(n_splits=1, train_size=max(2, int(len(train_pool) * fraction)), random_state=seed)
                relative, _ = next(splitter.split(np.zeros((len(train_pool), 1)), y[train_pool]))
                train_idx = train_pool[relative]
            else:
                train_idx = train_pool
            for offset, spec in enumerate((
                NeuralSpec("Scratch GINE", ("gine",)),
                NeuralSpec("Improved-pretrained GINE", ("gine",), args.improved_checkpoint),
            )):
                pred, training, _ = train_neural_fold(spec, graphs, train_idx, valid_idx, seed + int(fraction * 1000) + offset * 10000, args)
                rows.append({"seed": seed, "fraction": fraction, "train_count": len(train_idx), "model": spec.name, "best_epoch": training["best_epoch"], **ranking_metrics(y[valid_idx], pred)})
    result = pd.DataFrame(rows)
    result.to_csv(output / "sample_efficiency.csv", index=False)
    return result


def similarity_stratification(oof: pd.DataFrame, output: Path) -> pd.DataFrame:
    frame = oof[(oof.protocol == "stratified") & oof.model.isin(["Scratch GINE", "Improved-pretrained GINE"])]
    bins = [-np.inf, 0.4, 0.6, 0.8, np.inf]
    names = ["<0.4", "0.4-0.6", "0.6-0.8", ">=0.8"]
    frame = frame.copy()
    frame["similarity_bin"] = pd.cut(frame.max_train_tanimoto, bins=bins, labels=names, right=False)
    rows = []
    for (model, similarity_bin), group in frame.groupby(["model", "similarity_bin"], observed=True):
        metrics = ranking_metrics(group.label.to_numpy(), group.score.to_numpy())
        rows.append({"model": model, "similarity_bin": similarity_bin, "n": len(group), **metrics})
    result = pd.DataFrame(rows)
    result.to_csv(output / "similarity_stratification.csv", index=False)
    return result


def calibration_table(oof: pd.DataFrame, output: Path) -> pd.DataFrame:
    rows = []
    frame = oof[oof.protocol == "stratified"].groupby(["model", "sample_index", "label"], as_index=False).score.mean()
    for model, group in frame.groupby("model"):
        frac, mean = calibration_curve(group.label, group.score, n_bins=8, strategy="quantile")
        rows.extend({"model": model, "bin": idx, "mean_predicted": mp, "fraction_positive": fp} for idx, (mp, fp) in enumerate(zip(mean, frac)))
    result = pd.DataFrame(rows)
    result.to_csv(output / "calibration_curve.csv", index=False)
    return result


def make_figures(summary: pd.DataFrame, oof: pd.DataFrame, ablation: pd.DataFrame, efficiency: pd.DataFrame, similarity: pd.DataFrame, calibration: pd.DataFrame, output: Path) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors
    avg = oof[oof.protocol == "stratified"].groupby(["model", "sample_index", "label"], as_index=False).score.mean()
    for curve_name in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(8, 6))
        for color, (model, frame) in zip(colors, avg.groupby("model")):
            if curve_name == "roc":
                x, yv, _ = roc_curve(frame.label, frame.score)
                label = f"{model} ({roc_auc_score(frame.label, frame.score):.3f})"
                ax.set(xlabel="False positive rate", ylabel="True positive rate")
            else:
                yv, x, _ = precision_recall_curve(frame.label, frame.score)
                label = f"{model} ({average_precision_score(frame.label, frame.score):.3f})"
                ax.set(xlabel="Recall", ylabel="Precision")
            ax.plot(x, yv, label=label, color=color, linewidth=1.6)
        ax.legend(fontsize=7, frameon=False)
        ax.set_title(f"Development OOF {curve_name.upper()} curves")
        fig.tight_layout(); fig.savefig(figure_dir / f"{curve_name}_curves.png", dpi=180); plt.close(fig)
    if not ablation.empty:
        plot = ablation.groupby("model", as_index=False).roc_auc.mean().sort_values("roc_auc")
        fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(plot.model, plot.roc_auc, color="#2563EB"); ax.set_xlabel("ROC-AUC"); ax.set_title("Branch ablation"); fig.tight_layout(); fig.savefig(figure_dir / "branch_ablation.png", dpi=180); plt.close(fig)
    if not efficiency.empty:
        plot = efficiency.groupby(["model", "fraction"], as_index=False).agg(roc_auc=("roc_auc", "mean"), std=("roc_auc", "std"))
        fig, ax = plt.subplots(figsize=(7, 5))
        for model, frame in plot.groupby("model"):
            ax.plot(frame.fraction, frame.roc_auc, marker="o", label=model)
            if frame["std"].notna().any(): ax.fill_between(frame.fraction, frame.roc_auc-frame["std"].fillna(0), frame.roc_auc+frame["std"].fillna(0), alpha=.15)
        ax.set(xlabel="Fraction of fold training labels", ylabel="ROC-AUC", title="Sample efficiency"); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "sample_efficiency.png", dpi=180); plt.close(fig)
    if not similarity.empty:
        pivot = similarity.pivot(index="similarity_bin", columns="model", values="roc_auc")
        ax = pivot.plot(kind="bar", figsize=(8, 5)); ax.set(ylabel="ROC-AUC", title="Similarity stratification"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(figure_dir / "similarity_stratification.png", dpi=180); plt.close()
    if not calibration.empty:
        fig, ax = plt.subplots(figsize=(6, 6)); ax.plot([0,1],[0,1],"--",color="gray")
        for model, frame in calibration.groupby("model"): ax.plot(frame.mean_predicted, frame.fraction_positive, marker="o", label=model)
        ax.set(xlabel="Mean predicted probability", ylabel="Observed active fraction", title="Calibration"); ax.legend(fontsize=6, frameon=False); fig.tight_layout(); fig.savefig(figure_dir / "calibration_curve.png", dpi=180); plt.close(fig)
    enrich = summary[summary.protocol == "stratified"].sort_values("ef_at_10pct_mean")
    if not enrich.empty:
        fig, ax = plt.subplots(figsize=(8, 5)); ax.barh(enrich.model, enrich.ef_at_10pct_mean, color="#059669"); ax.axvline(1,color="gray",linestyle="--"); ax.set(xlabel="EF@10%", title="Early enrichment"); fig.tight_layout(); fig.savefig(figure_dir / "enrichment.png", dpi=180); plt.close(fig)


def exploratory_legacy_test(graphs: list, meta: pd.DataFrame, dev_idx: np.ndarray, test_idx: np.ndarray, args: argparse.Namespace, output: Path) -> pd.DataFrame:
    lock = {
        "model": "Improved-pretrained GINE-hybrid",
        "branches": ["gine", "rdkit", "morgan", "maccs"],
        "checkpoint": args.improved_checkpoint,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "branch_dim": args.branch_dim,
        "fusion": args.fusion,
        "epochs": args.epochs,
        "patience": args.patience,
        "seeds": list(SEEDS),
        "selection_data": "420-compound development partition only",
        "legacy_test_label": "post-redesign exploratory evaluation on an already exposed legacy test set",
    }
    lock_text = json.dumps(lock, sort_keys=True)
    lock["preprocessing_and_config_sha256"] = hashlib.sha256(lock_text.encode()).hexdigest()
    (output / "locked_v2_configuration.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    predictions = []
    seeds = (13,) if args.smoke else SEEDS
    for seed in seeds:
        spec = NeuralSpec("Improved-pretrained GINE-hybrid", ("gine", "rdkit", "morgan", "maccs"), args.improved_checkpoint)
        pred, training, _ = train_neural_fold(spec, graphs, dev_idx, test_idx, seed, args, output / "locked_models" / f"seed_{seed}.pt")
        for idx, score in zip(test_idx, pred):
            predictions.append({"seed": seed, "sample_index": int(idx), "label": int(graphs[idx].y_cls), "score": float(score), "evaluation_label": lock["legacy_test_label"]})
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(output / "legacy_test_predictions_exploratory.csv", index=False)
    rows = []
    for seed, frame in pred_df.groupby("seed"): rows.append({"seed": seed, **ranking_metrics(frame.label.to_numpy(), frame.score.to_numpy()), "evaluation_label": lock["legacy_test_label"]})
    result = pd.DataFrame(rows); result.to_csv(output / "legacy_test_metrics_exploratory.csv", index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="processed_smiles_graphs_multitask_ns_inactive/processed_graphs.pt")
    parser.add_argument("--review_csv", default="processed_smiles_graphs_multitask_ns_inactive/processed_index.csv")
    parser.add_argument("--legacy_assignments", default="outputs_pretraining_redesign_v2/inputs/legacy_split_assignments.csv")
    parser.add_argument("--original_checkpoint", default="outputs_pretraining_redesign_v2/inputs/original_pretrained_gine_checkpoint.pt")
    parser.add_argument("--improved_checkpoint", required=True)
    parser.add_argument("--output_dir", default="outputs_pretraining_redesign_v2")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--branch_dim", type=int, default=64)
    parser.add_argument("--fusion", choices=["concat", "gated"], default="concat")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=3e-3)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip_legacy_test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RDLogger.DisableLog("rdApp.*")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    graphs = load_graphs(Path(args.data))
    review = pd.read_csv(args.review_csv)
    if len(graphs) != 526 or len(review) != 526:
        raise RuntimeError(f"Expected locked 526-compound dataset, got graphs={len(graphs)}, rows={len(review)}")
    meta = split_metadata(review)
    legacy = pd.read_csv(args.legacy_assignments).sort_values("sample_index")
    if len(legacy) != 526:
        raise RuntimeError("Legacy split assignment does not cover all 526 compounds")
    dev_idx = legacy.loc[legacy.partition == "development", "sample_index"].to_numpy(dtype=int)
    test_idx = legacy.loc[legacy.partition == "locked_test", "sample_index"].to_numpy(dtype=int)
    if len(dev_idx) != 420 or len(test_idx) != 106 or set(dev_idx) & set(test_idx):
        raise RuntimeError("Expected disjoint 420/106 development/legacy-test split")
    dev_graphs = [graphs[idx] for idx in dev_idx]
    dev_meta = meta.iloc[dev_idx].reset_index(drop=True)
    oof, fold_metrics, assignments = run_oof(dev_graphs, dev_meta, args, output)
    summary = summarize_oof(oof, output)
    bootstrap = paired_comparisons(oof, output, 100 if args.smoke else args.bootstrap_samples)
    ablation = branch_ablation(dev_graphs, dev_meta, args, output)
    probe = frozen_probe(dev_graphs, dev_meta, args, output)
    efficiency = sample_efficiency(dev_graphs, dev_meta, args, output)
    similarity = similarity_stratification(oof, output)
    calibration = calibration_table(oof, output)
    make_figures(summary, oof, ablation, efficiency, similarity, calibration, output)
    legacy_result = pd.DataFrame()
    if not args.skip_legacy_test:
        legacy_result = exploratory_legacy_test(graphs, meta, dev_idx, test_idx, args, output)
    manifest = {
        "mode": "smoke" if args.smoke else "formal",
        "dataset_count": len(graphs), "development_count": len(dev_idx), "legacy_test_count": len(test_idx),
        "seeds": [13] if args.smoke else list(SEEDS), "protocols": ["stratified"] if args.smoke else list(PROTOCOLS),
        "main_models": list(MAIN_MODELS), "device": str(resolve_device(args.device)),
        "oof_rows": len(oof), "fold_metric_rows": len(fold_metrics), "bootstrap_rows": len(bootstrap),
        "branch_ablation_rows": len(ablation), "probe_rows": len(probe), "sample_efficiency_rows": len(efficiency),
        "legacy_test_rows": len(legacy_result),
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
