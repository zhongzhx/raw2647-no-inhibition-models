"""Core utilities for strict RAW264.7 GINE transfer ablation.

The exposed legacy test set is deliberately outside this module.  Every model
selection decision is made by an inner grouped split of the outer training
fold, followed by a clean refit on the complete outer training fold.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.loader import DataLoader

from gine_v2 import (
    GINEEncoder,
    GraphOnlyEncoder,
    IndependentHybridEncoder,
    MolecularClassifier,
    load_gine_checkpoint,
)
from pipeline_utils import set_seed


SEEDS = (13, 42, 2026)
PROTOCOLS = ("scaffold_group",)
METRIC_NAMES = (
    "roc_auc",
    "pr_auc",
    "mcc",
    "f1",
    "balanced_accuracy",
    "brier",
    "ece",
    "precision_at_10pct",
    "precision_at_20pct",
    "ef_at_10pct",
    "active_at_10",
    "active_at_20",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    branches: tuple[str, ...] = ("gine",)
    zeroed_branches: tuple[str, ...] = ()
    initialization: str = "scratch"
    schedule: str = "staged"
    encoder_lr_source: str = "pretrained"
    is_flat: bool = False
    fusion: str = "concat"
    branch_dim: int | None = None
    head_hidden_dim: int | None = None


def encoder_should_update(spec: ModelSpec) -> bool:
    """Return whether the GINE encoder contributes to this model's prediction."""

    return spec.schedule != "frozen" and "gine" not in spec.zeroed_branches


MODEL_SPECS = (
    ModelSpec(
        "Scratch GINE full",
        initialization="scratch",
        schedule="full",
        encoder_lr_source="scratch_full",
    ),
    ModelSpec(
        "Scratch GINE staged matched-LR",
        initialization="scratch",
        schedule="staged",
        encoder_lr_source="pretrained",
    ),
    ModelSpec(
        "Scratch GINE staged tuned-LR",
        initialization="scratch",
        schedule="staged",
        encoder_lr_source="scratch_staged",
    ),
    ModelSpec("Pretrained GINE frozen", initialization="pretrained", schedule="frozen"),
    ModelSpec("Pretrained GINE staged", initialization="pretrained", schedule="staged"),
    ModelSpec(
        "Flat-only baseline",
        branches=(),
        initialization="flat",
        schedule="locked_rf",
        encoder_lr_source="none",
        is_flat=True,
    ),
    ModelSpec(
        "Scratch GINE-hybrid staged matched-LR",
        branches=("gine", "rdkit", "morgan", "maccs"),
        initialization="scratch",
        schedule="staged",
        encoder_lr_source="pretrained",
    ),
    ModelSpec(
        "Pretrained GINE-hybrid staged",
        branches=("gine", "rdkit", "morgan", "maccs"),
        initialization="pretrained",
        schedule="staged",
    ),
)


def base_encoder_lr(spec: ModelSpec, args: Any) -> float:
    """Resolve the auditable encoder base LR assigned to one model specification."""

    sources = {
        "pretrained": "pretrained_encoder_lr",
        "scratch_staged": "scratch_staged_encoder_lr",
        "scratch_full": "scratch_full_encoder_lr",
    }
    if spec.encoder_lr_source == "none":
        return 0.0
    try:
        argument = sources[spec.encoder_lr_source]
    except KeyError as error:
        raise ValueError(f"Unknown encoder LR source: {spec.encoder_lr_source}") from error
    return float(getattr(args, argument))


def _warmup_factor(warmup_step: int, warmup_epochs: int) -> float:
    if warmup_epochs <= 0:
        raise ValueError("warmup_epochs must be positive")
    if warmup_epochs == 1:
        return 1.0
    return 0.1 + 0.9 * (warmup_step - 1) / (warmup_epochs - 1)


def _cosine_factor(decay_step: int, decay_steps: int, min_lr_ratio: float) -> float:
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in (0, 1]")
    if decay_steps <= 0:
        return 1.0
    progress = min(max(decay_step / decay_steps, 0.0), 1.0)
    return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def learning_rate_state(
    spec: ModelSpec,
    args: Any,
    epoch: int,
    scheduler_total_epochs: int | None = None,
) -> dict[str, Any]:
    """Return the head/encoder LR used for an epoch without mutable scheduler state.

    The horizon defaults to the locked maximum epoch budget rather than the selected
    refit length.  Consequently matched-LR scratch and pretrained runs always use
    identical LR prefixes even when inner validation selects different refit epochs.
    """

    if epoch < 1:
        raise ValueError("epoch must be at least 1")
    if args.lr_scheduler != "cosine":
        raise ValueError(f"Unsupported lr_scheduler: {args.lr_scheduler}")
    total_epochs = int(scheduler_total_epochs or args.epochs)
    if total_epochs < 1:
        raise ValueError("scheduler_total_epochs must be positive")

    base_head = float(args.head_lr)
    base_encoder = base_encoder_lr(spec, args)
    warmup_epochs = int(args.warmup_epochs)
    min_lr_ratio = float(args.min_lr_ratio)

    if spec.schedule == "staged" and epoch <= int(args.freeze_epochs):
        return {
            "head_lr": base_head,
            "encoder_lr": 0.0,
            "scheduler_phase": "head_only_frozen",
            "encoder_frozen": 1,
        }

    if spec.schedule == "staged":
        schedule_offset = int(args.freeze_epochs)
        schedule_epoch = epoch - schedule_offset
    elif spec.schedule in {"full", "frozen"}:
        schedule_offset = 0
        schedule_epoch = epoch
    else:
        raise ValueError(f"No neural LR schedule for {spec.schedule}")

    if schedule_epoch <= warmup_epochs:
        factor = _warmup_factor(schedule_epoch, warmup_epochs)
        # A staged head has already trained at its base LR.  Keep it at that LR
        # while the newly unfrozen encoder alone ramps from 10% to 100%.
        head_factor = 1.0 if spec.schedule == "staged" else factor
        phase = "warmup"
    else:
        decay_steps = max(total_epochs - schedule_offset - warmup_epochs, 0)
        factor = _cosine_factor(
            schedule_epoch - warmup_epochs,
            decay_steps,
            min_lr_ratio,
        )
        head_factor = factor
        phase = "cosine"

    encoder_frozen = int(not encoder_should_update(spec))
    return {
        "head_lr": base_head * head_factor,
        "encoder_lr": 0.0 if encoder_frozen else base_encoder * factor,
        "scheduler_phase": phase,
        "encoder_frozen": encoder_frozen,
    }


def apply_learning_rate_state(
    optimizer: torch.optim.Optimizer,
    state: dict[str, Any],
) -> dict[str, float]:
    """Apply epoch LRs to named parameter groups, including dynamically added ones."""

    applied: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("group_name", ""))
        if name not in {"head", "encoder"}:
            raise RuntimeError(f"Optimizer parameter group lacks a valid group_name: {name!r}")
        if name in applied:
            raise RuntimeError(f"Duplicate optimizer parameter group: {name}")
        lr = float(state[f"{name}_lr"])
        group["lr"] = lr
        applied[name] = lr
    if "head" not in applied:
        raise RuntimeError("Head optimizer parameter group is missing")
    if state["encoder_frozen"] == 0 and float(state["encoder_lr"]) > 0 and "encoder" not in applied:
        raise RuntimeError("Trainable encoder optimizer parameter group is missing")
    return applied


def learning_rate_trajectory(spec: ModelSpec, args: Any) -> list[dict[str, Any]]:
    return [
        {"epoch": epoch, **learning_rate_state(spec, args, epoch)}
        for epoch in range(1, int(args.epochs) + 1)
    ]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def stable_text_hash(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def bemis_murcko_scaffold(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"Invalid canonical SMILES during scaffold assignment: {smiles!r}")
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return "<acyclic>"
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def add_group_metadata(review: pd.DataFrame) -> pd.DataFrame:
    meta = review.copy()
    meta["scaffold_group"] = [bemis_murcko_scaffold(value) for value in meta.canonical_smiles]
    meta["label_type"] = np.select(
        [
            meta.has_exact_pic50.astype(int).eq(1),
            meta.is_censored_gt50.astype(int).eq(1),
        ],
        ["exact_ic50", "right_censored"],
        default="no_activity_token",
    )
    return meta


def graph_labels(graphs: list) -> np.ndarray:
    return np.asarray([int(graph.y_cls.view(-1)[0]) for graph in graphs], dtype=int)


def review_sample_weights(review: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(review), dtype=np.float32)
    censored = review.is_censored_gt50.astype(int).eq(1).to_numpy()
    thresholds = pd.to_numeric(review.censored_threshold_uM, errors="coerce").to_numpy(float)
    # Explicit right-censoring at or above the locked 50 μM activity
    # threshold is an unambiguous inactive label.  Lower or unavailable
    # censoring thresholds remain inactive but receive the protocol's
    # reduced confidence weight.
    weights[censored & np.isfinite(thresholds) & (thresholds >= 50.0)] = 1.0
    weights[censored & (~np.isfinite(thresholds) | (thresholds < 50.0))] = 0.8
    return weights


def batch_sample_weights(batch: Any) -> torch.Tensor:
    weights = torch.ones_like(batch.y_cls.view(-1).float())
    censored = batch.is_censored_gt50.view(-1).bool()
    thresholds = batch.censored_threshold_uM.view(-1).float()
    weights = torch.where(
        censored & torch.isfinite(thresholds) & (thresholds >= 50.0),
        torch.full_like(weights, 1.0),
        weights,
    )
    weights = torch.where(
        censored & (~torch.isfinite(thresholds) | (thresholds < 50.0)),
        torch.full_like(weights, 0.8),
        weights,
    )
    return weights


def standardized_graphs(graphs: list, train_indices: Iterable[int]) -> list:
    train_indices = [int(value) for value in train_indices]
    train_values = torch.cat([graphs[idx].global_feat.float() for idx in train_indices], dim=0)
    mean = train_values.mean(dim=0, keepdim=True)
    std = train_values.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-6)
    standardized = []
    for graph in graphs:
        item = copy.copy(graph)
        item.global_feat = (graph.global_feat.float() - mean) / std
        standardized.append(item)
    return standardized


def expected_calibration_error(y: np.ndarray, score: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    value = 0.0
    for idx in range(bins):
        if idx == bins - 1:
            mask = (score >= edges[idx]) & (score <= edges[idx + 1])
        else:
            mask = (score >= edges[idx]) & (score < edges[idx + 1])
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(float(y[mask].mean()) - float(score[mask].mean()))
    return float(value) if total else float("nan")


def classification_metrics(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    predicted = (score >= threshold).astype(int)
    order = np.argsort(-score)
    prevalence = float(y.mean())
    k10pct = max(1, int(math.ceil(0.10 * len(y))))
    k20pct = max(1, int(math.ceil(0.20 * len(y))))
    k10 = min(10, len(y))
    k20 = min(20, len(y))
    precision10pct = float(y[order[:k10pct]].mean())
    return {
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
        "pr_auc": float(average_precision_score(y, score)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "brier": float(brier_score_loss(y, score)),
        "ece": expected_calibration_error(y, score, bins=10),
        "precision_at_10pct": precision10pct,
        "precision_at_20pct": float(y[order[:k20pct]].mean()),
        "ef_at_10pct": precision10pct / max(prevalence, 1e-12),
        "active_at_10": float(y[order[:k10]].sum()),
        "active_at_20": float(y[order[:k20]].sum()),
    }


def grouped_folds(
    y: np.ndarray,
    meta: pd.DataFrame,
    protocol: str,
    seed: int,
    folds: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if protocol not in PROTOCOLS:
        raise ValueError(protocol)
    groups = meta[protocol].astype(str).to_numpy()
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignments = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    for train_idx, valid_idx in assignments:
        if set(groups[train_idx]) & set(groups[valid_idx]):
            raise RuntimeError(f"{protocol} group leakage")
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[valid_idx])) < 2:
            raise RuntimeError(f"{protocol} fold lacks both classes")
    return assignments


def inner_grouped_split(
    outer_train_idx: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    protocol: str,
    seed: int,
    folds: int,
) -> tuple[np.ndarray, np.ndarray]:
    local_y = y[outer_train_idx]
    local_meta = meta.iloc[outer_train_idx].reset_index(drop=True)
    local_train, local_valid = grouped_folds(local_y, local_meta, protocol, seed, folds)[0]
    inner_train = outer_train_idx[local_train]
    inner_valid = outer_train_idx[local_valid]
    return inner_train, inner_valid


def leakage_record(
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    protocol: str,
    seed: int,
    fold: int,
    level: str,
) -> dict[str, Any]:
    structure_overlap = set(meta.iloc[train_idx].inchikey) & set(meta.iloc[valid_idx].inchikey)
    scaffold_overlap = set(meta.iloc[train_idx].scaffold_group) & set(meta.iloc[valid_idx].scaffold_group)
    target_overlap = scaffold_overlap
    return {
        "level": level,
        "protocol": protocol,
        "seed": seed,
        "fold": fold,
        "train_count": len(train_idx),
        "validation_count": len(valid_idx),
        "train_active": int(meta.iloc[train_idx].active_label.sum()),
        "validation_active": int(meta.iloc[valid_idx].active_label.sum()),
        "structure_overlap_count": len(structure_overlap),
        "scaffold_overlap_count": len(scaffold_overlap),
        "publication_overlap_count": float("nan"),
        "publication_check_status": "not performed per user instruction",
        "target_group_overlap_count": len(target_overlap),
        "target_group_check_pass": int(len(target_overlap) == 0),
    }


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def changed_value_count(initial: dict[str, torch.Tensor], current: dict[str, torch.Tensor]) -> int:
    changed = 0
    for name, before in initial.items():
        after = current[name].detach().cpu()
        changed += int(torch.count_nonzero(before != after).item())
    return changed


def parameter_hash(parameters: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def training_order_signature(indices: np.ndarray, training_seed: int) -> str:
    payload = f"epoch_rng=seed+epoch*100003|seed={training_seed}|indices=" + ",".join(
        str(int(value)) for value in indices
    )
    return stable_text_hash(payload)


def epoch_order(indices: np.ndarray, training_seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(training_seed + epoch * 100003)
    return indices[rng.permutation(len(indices))]


def build_model(
    spec: ModelSpec,
    example_graph: Any,
    checkpoint: Path,
    args: Any,
    initialization_seed: int,
) -> tuple[nn.Module, GINEEncoder, dict[str, Any], str]:
    set_seed(initialization_seed)
    gine = GINEEncoder(
        hidden_dim=args.encoder_hidden_dim,
        num_layers=args.encoder_layers,
        dropout=args.encoder_dropout,
        use_layer_norm=True,
    )
    transfer: dict[str, Any] = {
        "checkpoint_path": "",
        "loaded_parameter_numel": 0,
        "target_parameter_numel": sum(value.numel() for value in gine.state_dict().values()),
        "load_ratio": float("nan"),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "exact_value_match_after_load": False,
    }
    if spec.initialization == "pretrained":
        transfer = load_gine_checkpoint(gine, str(checkpoint), min_load_ratio=0.95).to_dict()
        if (
            transfer["load_ratio"] != 1.0
            or transfer["missing_keys"]
            or transfer["unexpected_keys"]
            or transfer["shape_mismatches"]
            or not transfer["exact_value_match_after_load"]
        ):
            raise RuntimeError(f"Inexact checkpoint transfer for {spec.name}: {transfer}")

    if spec.branches == ("gine",):
        encoder = GraphOnlyEncoder(gine)
    else:
        encoder = IndependentHybridEncoder(
            gine,
            global_dim=int(example_graph.global_feat.shape[-1]),
            morgan_dim=int(example_graph.morgan_fp.shape[-1]),
            maccs_dim=int(example_graph.maccs_fp.shape[-1]),
            family_dim=(
                int(example_graph.scaffold_family_fp.shape[-1])
                if hasattr(example_graph, "scaffold_family_fp")
                else 0
            ),
            branch_dim=spec.branch_dim or args.branch_dim,
            dropout=args.head_dropout,
            enabled_branches=spec.branches,
            zeroed_branches=spec.zeroed_branches,
            fusion=spec.fusion,
        )
    model = MolecularClassifier(
        encoder,
        encoder.output_dim,
        spec.head_hidden_dim or args.head_hidden_dim,
        args.head_dropout,
    )
    encoder_ids = {id(value) for value in gine.parameters()}
    non_encoder_parameters = [value for value in model.parameters() if id(value) not in encoder_ids]
    return model, gine, transfer, parameter_hash(non_encoder_parameters)


@torch.no_grad()
def predict_model(
    model: nn.Module,
    graphs: list,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader([graphs[int(idx)] for idx in indices], batch_size=batch_size, shuffle=False)
    scores: list[float] = []
    for batch in loader:
        scores.extend(torch.sigmoid(model(batch.to(device))).detach().cpu().tolist())
    return np.asarray(scores, dtype=float)


def train_neural(
    spec: ModelSpec,
    graphs: list,
    train_idx: np.ndarray,
    checkpoint: Path,
    args: Any,
    training_seed: int,
    epochs: int,
    valid_idx: np.ndarray | None = None,
) -> tuple[nn.Module, list, list[dict[str, Any]], dict[str, Any], int]:
    """Train one model either for inner selection or a fixed final epoch count."""

    if spec.is_flat:
        raise ValueError("Flat model must use fit_flat_baseline")
    device = resolve_device(args.device)
    fold_graphs = standardized_graphs(graphs, train_idx)
    model, gine, transfer, head_initial_hash = build_model(
        spec, fold_graphs[int(train_idx[0])], checkpoint, args, training_seed
    )
    initial_encoder = clone_state(gine)
    encoder_params = list(gine.parameters())
    encoder_ids = {id(value) for value in encoder_params}
    downstream_params = [value for value in model.parameters() if id(value) not in encoder_ids]
    encoder_update_expected = encoder_should_update(spec)

    if spec.schedule in {"staged", "frozen"} or not encoder_update_expected:
        for parameter in encoder_params:
            parameter.requires_grad = False
    model = model.to(device)
    base_encoder = base_encoder_lr(spec, args)

    if spec.schedule == "full" and encoder_update_expected:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": downstream_params,
                    "lr": args.head_lr,
                    "base_lr": args.head_lr,
                    "group_name": "head",
                },
                {
                    "params": encoder_params,
                    "lr": base_encoder,
                    "base_lr": base_encoder,
                    "group_name": "encoder",
                },
            ],
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": downstream_params,
                    "lr": args.head_lr,
                    "base_lr": args.head_lr,
                    "group_name": "head",
                }
            ],
            weight_decay=args.weight_decay,
        )

    y_train = graph_labels([fold_graphs[int(idx)] for idx in train_idx])
    pos_weight = torch.tensor(
        [(len(y_train) - int(y_train.sum())) / max(int(y_train.sum()), 1)],
        dtype=torch.float32,
        device=device,
    )
    best_epoch = 0
    best_pr = -float("inf")
    best_model_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    gradient_batches_total = 0
    gradient_norm_max = 0.0
    started = time.time()

    for epoch in range(1, epochs + 1):
        if spec.schedule == "staged" and epoch == args.freeze_epochs + 1:
            for parameter in encoder_params:
                parameter.requires_grad = True
            optimizer.add_param_group(
                {
                    "params": encoder_params,
                    "lr": base_encoder,
                    "base_lr": base_encoder,
                    "group_name": "encoder",
                    "weight_decay": args.weight_decay,
                }
            )

        lr_state = learning_rate_state(spec, args, epoch)
        applied_lrs = apply_learning_rate_state(optimizer, lr_state)
        if not math.isclose(applied_lrs["head"], float(lr_state["head_lr"]), rel_tol=0, abs_tol=0):
            raise RuntimeError("Head learning rate was not applied exactly")
        if "encoder" in applied_lrs and not math.isclose(
            applied_lrs["encoder"], float(lr_state["encoder_lr"]), rel_tol=0, abs_tol=0
        ):
            raise RuntimeError("Encoder learning rate was not applied exactly")

        torch.manual_seed(training_seed + epoch * 1009)
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(training_seed + epoch * 1009)
        model.train()
        if not any(parameter.requires_grad for parameter in encoder_params):
            gine.eval()
        ordered_idx = epoch_order(train_idx, training_seed, epoch)
        loader = DataLoader(
            [fold_graphs[int(idx)] for idx in ordered_idx],
            batch_size=args.batch_size,
            shuffle=False,
        )
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        epoch_gradient_batches = 0
        epoch_gradient_norm_sum = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            target = batch.y_cls.view(-1).float()
            loss_values = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none", pos_weight=pos_weight
            )
            weights = batch_sample_weights(batch)
            loss = (loss_values * weights).sum() / weights.sum().clamp(min=1e-8)
            loss.backward()
            encoder_grad_sq = 0.0
            for parameter in encoder_params:
                if parameter.grad is not None:
                    encoder_grad_sq += float(parameter.grad.detach().pow(2).sum().item())
            encoder_grad_norm = math.sqrt(encoder_grad_sq)
            if encoder_grad_norm > 0:
                epoch_gradient_batches += 1
                epoch_gradient_norm_sum += encoder_grad_norm
                gradient_norm_max = max(gradient_norm_max, encoder_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            weighted_loss_sum += float((loss_values * weights).sum().detach().cpu())
            weight_sum += float(weights.sum().detach().cpu())

        gradient_batches_total += epoch_gradient_batches
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_weighted_bce": weighted_loss_sum / max(weight_sum, 1e-12),
            "head_lr": float(lr_state["head_lr"]),
            "encoder_lr": float(lr_state["encoder_lr"]),
            "scheduler_phase": str(lr_state["scheduler_phase"]),
            "encoder_frozen": int(not any(value.requires_grad for value in encoder_params)),
            "encoder_gradient_batches": epoch_gradient_batches,
            "encoder_gradient_norm_mean": (
                epoch_gradient_norm_sum / epoch_gradient_batches if epoch_gradient_batches else 0.0
            ),
        }
        if valid_idx is not None:
            valid_y = graph_labels([fold_graphs[int(idx)] for idx in valid_idx])
            valid_score = predict_model(model, fold_graphs, valid_idx, device, args.batch_size)
            row.update(classification_metrics(valid_y, valid_score))
            selection_start = (
                args.freeze_epochs + args.warmup_epochs
                if spec.schedule == "staged"
                else args.warmup_epochs
            )
            selection_allowed = epoch >= selection_start
            if selection_allowed and row["pr_auc"] > best_pr + args.min_delta:
                best_pr = float(row["pr_auc"])
                best_epoch = epoch
                best_model_state = clone_state(model)
                stale = 0
            elif selection_allowed:
                stale += 1
            minimum_stop = (
                max(args.freeze_epochs + args.min_finetune_epochs, selection_start)
                if spec.schedule == "staged"
                else selection_start
            )
            history.append(row)
            if selection_allowed and epoch >= minimum_stop and stale >= args.patience:
                break
        else:
            history.append(row)

    if valid_idx is None:
        best_epoch = epochs
    elif best_epoch == 0:
        raise RuntimeError(f"No inner epoch selected for {spec.name}")
    elif best_model_state is None:
        raise RuntimeError(f"No best model state captured for {spec.name}")
    else:
        model.load_state_dict(best_model_state)
    final_encoder = clone_state(gine)
    encoder_changed = changed_value_count(initial_encoder, final_encoder)
    if not encoder_update_expected:
        if gradient_batches_total != 0 or encoder_changed != 0:
            raise RuntimeError(f"Inactive encoder changed for {spec.name}")
    else:
        if gradient_batches_total == 0 or encoder_changed == 0:
            raise RuntimeError(f"Encoder did not train for {spec.name}")
    theoretical_trajectory = learning_rate_trajectory(spec, args)
    trajectory_hash = stable_text_hash(
        json.dumps(theoretical_trajectory, sort_keys=True, separators=(",", ":"))
    )
    audit = {
        "model": spec.name,
        "schedule": spec.schedule,
        "initialization": spec.initialization,
        "encoder_lr_source": spec.encoder_lr_source,
        "base_head_lr": float(args.head_lr),
        "base_encoder_lr": base_encoder,
        "scheduler": args.lr_scheduler,
        "warmup_epochs": int(args.warmup_epochs),
        "min_lr_ratio": float(args.min_lr_ratio),
        "scheduler_horizon_epochs": int(args.epochs),
        "lr_trajectory_sha256": trajectory_hash,
        "max_head_lr_observed": max(float(row["head_lr"]) for row in history),
        "min_head_lr_observed": min(float(row["head_lr"]) for row in history),
        "final_head_lr": float(history[-1]["head_lr"]),
        "max_encoder_lr_observed": max(float(row["encoder_lr"]) for row in history),
        "min_encoder_lr_observed": min(float(row["encoder_lr"]) for row in history),
        "final_encoder_lr": float(history[-1]["encoder_lr"]),
        "epochs_ran": len(history),
        "selected_epoch": best_epoch,
        "best_inner_pr_auc": best_pr if valid_idx is not None else float("nan"),
        "encoder_gradient_batches": gradient_batches_total,
        "encoder_gradient_norm_max": gradient_norm_max,
        "encoder_changed_values": encoder_changed,
        "encoder_update_expected": int(encoder_update_expected),
        "head_initial_sha256": head_initial_hash,
        "training_order_signature": training_order_signature(train_idx, training_seed),
        "elapsed_seconds": time.time() - started,
        "total_parameter_count": sum(int(value.numel()) for value in model.parameters()),
        "trainable_parameter_count": sum(
            int(value.numel()) for value in model.parameters() if value.requires_grad
        ),
        "fusion": spec.fusion,
        "branch_dim": spec.branch_dim or args.branch_dim,
        "head_hidden_dim": spec.head_hidden_dim or args.head_hidden_dim,
        "checkpoint_load_ratio": transfer["load_ratio"],
        "checkpoint_loaded_numel": transfer["loaded_parameter_numel"],
        "checkpoint_target_numel": transfer["target_parameter_numel"],
        "checkpoint_missing_keys": len(transfer["missing_keys"]),
        "checkpoint_unexpected_keys": len(transfer["unexpected_keys"]),
        "checkpoint_shape_mismatches": len(transfer["shape_mismatches"]),
        "checkpoint_exact_match": transfer["exact_value_match_after_load"],
    }
    return model, fold_graphs, history, audit, best_epoch


def flat_matrix(graphs: list, train_idx: np.ndarray) -> np.ndarray:
    global_values = np.vstack([graph.global_feat.view(-1).float().numpy() for graph in graphs])
    mean = global_values[train_idx].mean(axis=0, keepdims=True)
    std = global_values[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    global_values = (global_values - mean) / std
    morgan = np.vstack([graph.morgan_fp.view(-1).float().numpy() for graph in graphs])
    maccs = np.vstack([graph.maccs_fp.view(-1).float().numpy() for graph in graphs])
    return np.concatenate([global_values, morgan, maccs], axis=1)


def fit_flat_baseline(
    graphs: list,
    review: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int,
    args: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    x = flat_matrix(graphs, train_idx)
    y = graph_labels(graphs)
    weights = review_sample_weights(review)
    model = RandomForestClassifier(
        n_estimators=args.rf_estimators,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    started = time.time()
    model.fit(x[train_idx], y[train_idx], sample_weight=weights[train_idx])
    score = model.predict_proba(x[valid_idx])[:, 1]
    return score, {
        "model": "Flat-only baseline",
        "schedule": "locked_rf",
        "initialization": "flat",
        "encoder_lr_source": "none",
        "base_head_lr": 0.0,
        "base_encoder_lr": 0.0,
        "scheduler": "locked_rf",
        "warmup_epochs": 0,
        "min_lr_ratio": float("nan"),
        "scheduler_horizon_epochs": 0,
        "lr_trajectory_sha256": "",
        "max_head_lr_observed": 0.0,
        "min_head_lr_observed": 0.0,
        "final_head_lr": 0.0,
        "max_encoder_lr_observed": 0.0,
        "min_encoder_lr_observed": 0.0,
        "final_encoder_lr": 0.0,
        "epochs_ran": 0,
        "selected_epoch": 0,
        "best_inner_pr_auc": float("nan"),
        "encoder_gradient_batches": 0,
        "encoder_gradient_norm_max": 0.0,
        "encoder_changed_values": 0,
        "head_initial_sha256": "",
        "training_order_signature": training_order_signature(train_idx, seed),
        "elapsed_seconds": time.time() - started,
        "checkpoint_load_ratio": float("nan"),
        "checkpoint_loaded_numel": 0,
        "checkpoint_target_numel": 0,
        "checkpoint_missing_keys": 0,
        "checkpoint_unexpected_keys": 0,
        "checkpoint_shape_mismatches": 0,
        "checkpoint_exact_match": False,
    }


def spec_dicts() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in MODEL_SPECS]
