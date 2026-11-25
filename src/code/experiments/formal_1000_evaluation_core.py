"""Development-only utilities for the formal RAW264.7 capability evaluation.

The module contains no locked-test path and cannot load a legacy exposed test set.
All model selection accepts explicit development indices and scaffold groups.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import pearsonr, spearmanr
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_max_pool, global_mean_pool

from gine_v2 import GINEEncoder, load_gine_checkpoint
from pipeline_utils import set_seed


MODEL_NAMES = (
    "Flat-only Random Forest",
    "Flat-only best locked ML",
    "Scratch GINE full",
    "Scratch GINE staged matched-LR",
    "Scratch GINE staged tuned-LR",
    "Pretrained GINE frozen true linear probe",
    "Pretrained GINE frozen MLP probe",
    "Pretrained GINE staged fine-tune",
    "Pretrained GINE full fine-tune",
    "Pretrained GINE multitask",
    "Scratch GINE multitask",
)


@dataclass(frozen=True)
class NeuralSpec:
    name: str
    initialization: str
    schedule: str
    head_type: str = "mlp"
    multitask: bool = False


NEURAL_SPECS = {
    spec.name: spec
    for spec in (
        NeuralSpec("Scratch GINE full", "scratch", "full"),
        NeuralSpec("Scratch GINE staged matched-LR", "scratch", "staged"),
        NeuralSpec("Scratch GINE staged tuned-LR", "scratch", "staged"),
        NeuralSpec("Pretrained GINE frozen true linear probe", "pretrained", "frozen", "linear"),
        NeuralSpec("Pretrained GINE frozen MLP probe", "pretrained", "frozen", "mlp"),
        NeuralSpec("Pretrained GINE staged fine-tune", "pretrained", "staged"),
        NeuralSpec("Pretrained GINE full fine-tune", "pretrained", "full"),
        NeuralSpec("Pretrained GINE multitask", "pretrained", "staged", "mlp", True),
        NeuralSpec("Scratch GINE multitask", "scratch", "full", "mlp", True),
    )
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_all_seeds(seed: int) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def labels_from_graphs(graphs: Sequence[Any]) -> np.ndarray:
    return np.asarray([int(graph.y_cls.view(-1)[0]) for graph in graphs], dtype=int)


def exact_targets_from_graphs(graphs: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    exact = np.asarray([int(graph.has_exact_pic50.view(-1)[0]) for graph in graphs], dtype=bool)
    values = np.asarray([float(graph.y_reg.view(-1)[0]) for graph in graphs], dtype=float)
    return exact, values


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(y, dtype=float)
    probability = np.asarray(probability, dtype=float)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (probability >= edges[index]) & (probability <= edges[index + 1])
        else:
            mask = (probability >= edges[index]) & (probability < edges[index + 1])
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return float(ece)


def classification_metrics(
    y: np.ndarray, probability: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    probability = np.asarray(probability, dtype=float)
    predicted = (probability >= threshold).astype(int)
    order = np.argsort(-probability, kind="stable")
    k10_percent = max(1, int(math.ceil(0.10 * len(y))))
    k20_percent = max(1, int(math.ceil(0.20 * len(y))))
    k10 = min(10, len(y))
    k20 = min(20, len(y))
    prevalence = float(y.mean()) if len(y) else float("nan")
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    precision10 = float(y[order[:k10_percent]].mean())
    return {
        "pr_auc": float(average_precision_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
        "ef_at_10pct": precision10 / max(prevalence, 1e-12),
        "active_at_10": float(y[order[:k10]].sum()),
        "active_at_20": float(y[order[:k20]].sum()),
        "precision_at_10pct": precision10,
        "precision_at_20pct": float(y[order[:k20_percent]].mean()),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "brier": float(brier_score_loss(y, probability)),
        "ece": expected_calibration_error(y, probability),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
    }


def regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(prediction)
    y, prediction = y[valid], prediction[valid]
    if len(y) < 2:
        return {key: float("nan") for key in ("rmse", "mae", "r2", "pearson_r", "spearman_rho")}
    return {
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
        "r2": float(r2_score(y, prediction)),
        "pearson_r": float(pearsonr(y, prediction).statistic),
        "spearman_rho": float(spearmanr(y, prediction).statistic),
    }


def grouped_folds(
    y: np.ndarray, groups: np.ndarray, folds: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result = list(splitter.split(np.zeros((len(y), 1)), y, groups))
    for train, valid in result:
        if set(groups[train]) & set(groups[valid]):
            raise RuntimeError("Scaffold leakage detected")
        if len(np.unique(y[train])) != 2 or len(np.unique(y[valid])) != 2:
            raise RuntimeError("Grouped fold lacks both classes")
    return result


def training_order(indices: np.ndarray, seed: int, epoch: int) -> np.ndarray:
    rng = np.random.default_rng(seed + epoch * 100003)
    return indices[rng.permutation(len(indices))]


def graph_sample_weights(batch: Any) -> torch.Tensor:
    weights = torch.ones_like(batch.y_cls.view(-1).float())
    censored = batch.is_censored_gt50.view(-1).bool()
    thresholds = batch.censored_threshold_uM.view(-1).float()
    weights = torch.where(
        censored & torch.isfinite(thresholds) & (thresholds >= 50.0),
        torch.full_like(weights, 0.8),
        weights,
    )
    weights = torch.where(
        censored & (~torch.isfinite(thresholds) | (thresholds < 50.0)),
        torch.full_like(weights, 0.5),
        weights,
    )
    return weights


def _cosine_factor(step: int, total_steps: int, min_ratio: float) -> float:
    if total_steps <= 0:
        return 1.0
    progress = min(max(step / total_steps, 0.0), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def learning_rates(
    schedule: str,
    epoch: int,
    max_epochs: int,
    freeze_epochs: int,
    warmup_epochs: int,
    base_head_lr: float,
    base_encoder_lr: float,
    min_lr_ratio: float,
) -> dict[str, Any]:
    if schedule == "staged" and epoch <= freeze_epochs:
        return {"head_lr": base_head_lr, "encoder_lr": 0.0, "scheduler_phase": "head_only_frozen", "encoder_frozen": 1}
    if schedule == "staged":
        local_epoch = epoch - freeze_epochs
    else:
        local_epoch = epoch
    if local_epoch <= warmup_epochs:
        if warmup_epochs <= 1:
            factor = 1.0
        else:
            factor = 0.1 + 0.9 * (local_epoch - 1) / (warmup_epochs - 1)
        head_factor = 1.0 if schedule == "staged" else factor
        phase = "warmup"
    else:
        available = max(max_epochs - (freeze_epochs if schedule == "staged" else 0) - warmup_epochs, 0)
        factor = _cosine_factor(local_epoch - warmup_epochs, available, min_lr_ratio)
        head_factor = factor
        phase = "cosine"
    frozen = int(schedule == "frozen")
    return {
        "head_lr": base_head_lr * head_factor,
        "encoder_lr": 0.0 if frozen else base_encoder_lr * factor,
        "scheduler_phase": phase,
        "encoder_frozen": frozen,
    }


class FormalGINEModel(nn.Module):
    def __init__(
        self,
        encoder: GINEEncoder,
        pooling: str,
        head_type: str,
        head_hidden_dim: int,
        head_dropout: float,
        multitask: bool,
    ) -> None:
        super().__init__()
        if pooling not in {"mean", "mean_plus_max"}:
            raise ValueError(pooling)
        self.encoder = encoder
        self.pooling = pooling
        output_dim = encoder.hidden_dim * (2 if pooling == "mean_plus_max" else 1)
        if head_type == "linear":
            self.classification_head = nn.Linear(output_dim, 1)
        elif head_type == "mlp":
            self.classification_head = nn.Sequential(
                nn.Linear(output_dim, head_hidden_dim),
                nn.LayerNorm(head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1),
            )
        else:
            raise ValueError(head_type)
        self.regression_head = (
            nn.Sequential(
                nn.Linear(output_dim, head_hidden_dim),
                nn.LayerNorm(head_hidden_dim),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(head_hidden_dim, 1),
            )
            if multitask
            else None
        )

    def embed(self, data: Any) -> torch.Tensor:
        nodes = self.encoder.forward_nodes(data)
        mean = global_mean_pool(nodes, data.batch)
        if self.pooling == "mean":
            return mean
        return torch.cat([mean, global_max_pool(nodes, data.batch)], dim=-1)

    def forward(self, data: Any) -> dict[str, torch.Tensor | None]:
        embedding = self.embed(data)
        regression = None if self.regression_head is None else self.regression_head(embedding).view(-1)
        return {
            "logits": self.classification_head(embedding).view(-1),
            "regression": regression,
            "embedding": embedding,
        }


def parameter_hash(parameters: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_neural_model(
    spec: NeuralSpec,
    params: dict[str, Any],
    checkpoint: Path,
    seed: int,
) -> tuple[FormalGINEModel, dict[str, Any]]:
    set_all_seeds(seed)
    encoder = GINEEncoder(
        hidden_dim=256,
        num_layers=5,
        dropout=float(params["encoder_dropout"]),
        use_layer_norm=True,
    )
    transfer: dict[str, Any] = {
        "load_ratio": float("nan"),
        "exact_value_match_after_load": False,
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
    }
    if spec.initialization == "pretrained":
        transfer = load_gine_checkpoint(encoder, str(checkpoint), min_load_ratio=1.0).to_dict()
        if (
            transfer["load_ratio"] != 1.0
            or transfer["missing_keys"]
            or transfer["unexpected_keys"]
            or transfer["shape_mismatches"]
            or not transfer["exact_value_match_after_load"]
        ):
            raise RuntimeError("Pretraining checkpoint did not load exactly")
    model = FormalGINEModel(
        encoder=encoder,
        pooling=str(params["pooling"]),
        head_type=spec.head_type,
        head_hidden_dim=int(params.get("head_hidden_dim", 256)),
        head_dropout=float(params["head_dropout"]),
        multitask=spec.multitask,
    )
    encoder_ids = {id(parameter) for parameter in encoder.parameters()}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    audit = {
        "transfer": transfer,
        "encoder_parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
        "head_parameter_count": sum(parameter.numel() for parameter in head_parameters),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "head_initial_sha256": parameter_hash(head_parameters),
    }
    return model, audit


@torch.no_grad()
def predict_neural(
    model: FormalGINEModel,
    graphs: Sequence[Any],
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    loader = DataLoader([graphs[int(index)] for index in indices], batch_size=batch_size, shuffle=False)
    probabilities: list[float] = []
    regressions: list[float] = []
    embeddings: list[np.ndarray] = []
    for batch in loader:
        output = model(batch.to(device))
        probabilities.extend(torch.sigmoid(output["logits"]).detach().cpu().tolist())
        if output["regression"] is None:
            regressions.extend([float("nan")] * int(output["logits"].shape[0]))
        else:
            regressions.extend(output["regression"].detach().cpu().tolist())
        embeddings.append(output["embedding"].detach().cpu().numpy())
    return np.asarray(probabilities), np.asarray(regressions), np.concatenate(embeddings, axis=0)


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _changed_values(initial: dict[str, torch.Tensor], module: nn.Module) -> int:
    current = module.state_dict()
    return sum(int(torch.count_nonzero(before != current[name].detach().cpu()).item()) for name, before in initial.items())


def train_neural(
    spec: NeuralSpec,
    params: dict[str, Any],
    graphs: Sequence[Any],
    train_idx: np.ndarray,
    checkpoint: Path,
    seed: int,
    max_epochs: int,
    patience: int,
    min_delta: float,
    grad_clip: float,
    min_lr_ratio: float,
    device_name: str,
    valid_idx: np.ndarray | None = None,
    fixed_epochs: int | None = None,
) -> tuple[FormalGINEModel, list[dict[str, Any]], dict[str, Any], int]:
    device = resolve_device(device_name)
    model, construction = build_neural_model(spec, params, checkpoint, seed)
    encoder_parameters = list(model.encoder.parameters())
    encoder_ids = {id(parameter) for parameter in encoder_parameters}
    head_parameters = [parameter for parameter in model.parameters() if id(parameter) not in encoder_ids]
    initial_encoder = _clone_state(model.encoder)
    freeze_epochs = int(params.get("freeze_epochs", 0))
    warmup_epochs = int(params["warmup_epochs"])
    base_head_lr = float(params["head_lr"])
    base_encoder_lr = float(params.get("encoder_lr", 0.0))
    weight_decay = float(params["weight_decay"])
    batch_size = int(params["batch_size"])
    alpha = float(params.get("alpha", 0.0))

    if spec.schedule in {"staged", "frozen"}:
        for parameter in encoder_parameters:
            parameter.requires_grad = False
    optimizer_groups: list[dict[str, Any]] = [
        {"params": head_parameters, "lr": base_head_lr, "group_name": "head", "weight_decay": weight_decay}
    ]
    if spec.schedule == "full":
        optimizer_groups.append(
            {"params": encoder_parameters, "lr": base_encoder_lr, "group_name": "encoder", "weight_decay": weight_decay}
        )
    optimizer = torch.optim.AdamW(optimizer_groups)
    model.to(device)

    train_y = labels_from_graphs([graphs[int(index)] for index in train_idx])
    positive_weight = torch.tensor(
        [(len(train_y) - int(train_y.sum())) / max(int(train_y.sum()), 1)],
        dtype=torch.float32,
        device=device,
    )
    exact_mask_train, regression_train = exact_targets_from_graphs([graphs[int(index)] for index in train_idx])
    regression_mean = float(np.nanmean(regression_train[exact_mask_train])) if exact_mask_train.any() else 0.0
    regression_std = float(np.nanstd(regression_train[exact_mask_train])) if exact_mask_train.any() else 1.0
    regression_std = max(regression_std, 1e-6)

    history: list[dict[str, Any]] = []
    best_pr_auc = -float("inf")
    best_epoch = 0
    stale = 0
    encoder_gradient_batches = 0
    encoder_gradient_nonzero = 0
    optimizer_steps = 0
    optimizer_step_seconds = 0.0
    peak_memory_bytes = 0
    started = time.perf_counter()
    epoch_limit = int(fixed_epochs or max_epochs)

    for epoch in range(1, epoch_limit + 1):
        if spec.schedule == "staged" and epoch == freeze_epochs + 1:
            for parameter in encoder_parameters:
                parameter.requires_grad = True
            optimizer.add_param_group(
                {"params": encoder_parameters, "lr": base_encoder_lr, "group_name": "encoder", "weight_decay": weight_decay}
            )
        lr_state = learning_rates(
            spec.schedule,
            epoch,
            max_epochs,
            freeze_epochs,
            warmup_epochs,
            base_head_lr,
            base_encoder_lr,
            min_lr_ratio,
        )
        seen_groups: set[str] = set()
        for group in optimizer.param_groups:
            name = str(group["group_name"])
            if name in seen_groups:
                raise RuntimeError("Duplicate optimizer parameter group")
            seen_groups.add(name)
            group["lr"] = float(lr_state[f"{name}_lr"])
        if "head" not in seen_groups or (not lr_state["encoder_frozen"] and lr_state["encoder_lr"] > 0 and "encoder" not in seen_groups):
            raise RuntimeError("Learning-rate parameter group missing")

        set_all_seeds(seed + epoch * 1009)
        model.train()
        if not any(parameter.requires_grad for parameter in encoder_parameters):
            model.encoder.eval()
        ordered = training_order(train_idx, seed, epoch)
        loader = DataLoader([graphs[int(index)] for index in ordered], batch_size=batch_size, shuffle=False)
        classification_loss_sum = regression_loss_sum = total_loss_sum = 0.0
        sample_count = exact_count = 0
        epoch_encoder_norm = epoch_head_norm = 0.0
        epoch_gradient_batches = 0
        epoch_started = time.perf_counter()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            target = batch.y_cls.view(-1).float()
            class_values = F.binary_cross_entropy_with_logits(
                output["logits"], target, reduction="none", pos_weight=positive_weight
            )
            weights = graph_sample_weights(batch)
            classification_loss = (class_values * weights).sum() / weights.sum().clamp(min=1e-8)
            exact = batch.has_exact_pic50.view(-1).bool()
            if spec.multitask and bool(exact.any()):
                normalized_target = (batch.y_reg.view(-1).float()[exact] - regression_mean) / regression_std
                normalized_prediction = (output["regression"][exact] - regression_mean) / regression_std
                regression_loss = F.mse_loss(normalized_prediction, normalized_target, reduction="mean")
            else:
                regression_loss = classification_loss * 0.0
            loss = classification_loss + alpha * regression_loss
            loss.backward()
            encoder_norm_sq = sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in encoder_parameters
                if parameter.grad is not None
            )
            head_norm_sq = sum(
                float(parameter.grad.detach().pow(2).sum().item())
                for parameter in head_parameters
                if parameter.grad is not None
            )
            encoder_norm = math.sqrt(encoder_norm_sq)
            head_norm = math.sqrt(head_norm_sq)
            if any(parameter.requires_grad for parameter in encoder_parameters):
                encoder_gradient_batches += 1
                epoch_gradient_batches += 1
                if encoder_norm > 0:
                    encoder_gradient_nonzero += 1
            epoch_encoder_norm += encoder_norm
            epoch_head_norm += head_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            step_started = time.perf_counter()
            optimizer.step()
            optimizer_step_seconds += time.perf_counter() - step_started
            optimizer_steps += 1
            batch_count = int(target.numel())
            sample_count += batch_count
            exact_count += int(exact.sum().item())
            classification_loss_sum += float(classification_loss.detach().cpu()) * batch_count
            regression_loss_sum += float(regression_loss.detach().cpu()) * int(exact.sum().item())
            total_loss_sum += float(loss.detach().cpu()) * batch_count
            if device.type == "mps":
                peak_memory_bytes = max(peak_memory_bytes, int(torch.mps.current_allocated_memory()))
        row: dict[str, Any] = {
            "epoch": epoch,
            "classification_loss": classification_loss_sum / max(sample_count, 1),
            "regression_loss": regression_loss_sum / max(exact_count, 1),
            "total_loss": total_loss_sum / max(sample_count, 1),
            "head_lr": float(lr_state["head_lr"]),
            "encoder_lr": float(lr_state["encoder_lr"]),
            "scheduler_phase": str(lr_state["scheduler_phase"]),
            "encoder_frozen": int(not any(parameter.requires_grad for parameter in encoder_parameters)),
            "encoder_gradient_norm": epoch_encoder_norm / max(len(loader), 1),
            "head_gradient_norm": epoch_head_norm / max(len(loader), 1),
            "gradient_nonzero_batch_ratio": (
                epoch_gradient_batches / max(len(loader), 1) if any(parameter.requires_grad for parameter in encoder_parameters) else 0.0
            ),
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        if valid_idx is not None:
            probabilities, regression_predictions, _ = predict_neural(model, graphs, valid_idx, device, batch_size)
            valid_y = labels_from_graphs([graphs[int(index)] for index in valid_idx])
            row.update({f"valid_{key}": value for key, value in classification_metrics(valid_y, probabilities).items()})
            exact_valid, regression_valid = exact_targets_from_graphs([graphs[int(index)] for index in valid_idx])
            if spec.multitask:
                row.update({f"valid_pic50_{key}": value for key, value in regression_metrics(regression_valid[exact_valid], regression_predictions[exact_valid]).items()})
            selection_start = (freeze_epochs if spec.schedule == "staged" else 0) + warmup_epochs
            if epoch >= max(selection_start, 1) and row["valid_pr_auc"] > best_pr_auc + min_delta:
                best_pr_auc = float(row["valid_pr_auc"])
                best_epoch = epoch
                stale = 0
            elif epoch >= max(selection_start, 1):
                stale += 1
        history.append(row)
        if valid_idx is not None and stale >= patience and epoch >= max((freeze_epochs if spec.schedule == "staged" else 0) + warmup_epochs, 1):
            break

    if valid_idx is None:
        best_epoch = epoch_limit
    if best_epoch <= 0:
        raise RuntimeError(f"No epoch selected for {spec.name}")
    encoder_changed = _changed_values(initial_encoder, model.encoder)
    if spec.schedule == "frozen" and (encoder_changed != 0 or encoder_gradient_nonzero != 0):
        raise RuntimeError("Frozen encoder changed or received a nonzero gradient")
    if spec.schedule != "frozen" and encoder_changed == 0:
        raise RuntimeError("Trainable encoder did not change")
    audit = {
        "model": spec.name,
        "initialization": spec.initialization,
        "schedule": spec.schedule,
        "head_type": spec.head_type,
        "multitask": spec.multitask,
        "alpha": alpha,
        "base_head_lr": base_head_lr,
        "base_encoder_lr": base_encoder_lr,
        "scheduler": "warmup_cosine",
        "warmup_epochs": warmup_epochs,
        "freeze_epochs": freeze_epochs,
        "min_lr_ratio": min_lr_ratio,
        "scheduler_horizon_epochs": max_epochs,
        "selected_epoch": best_epoch,
        "best_inner_pr_auc": best_pr_auc if valid_idx is not None else float("nan"),
        "epochs_ran": len(history),
        "encoder_changed_values": encoder_changed,
        "encoder_gradient_batches": encoder_gradient_batches,
        "encoder_gradient_nonzero_batches": encoder_gradient_nonzero,
        "optimizer_steps": optimizer_steps,
        "mean_optimizer_step_seconds": optimizer_step_seconds / max(optimizer_steps, 1),
        "elapsed_seconds": time.perf_counter() - started,
        "peak_memory_bytes": peak_memory_bytes,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys_platform_is_macos() else 1024),
        "lr_trajectory_sha256": stable_hash([
            learning_rates(spec.schedule, epoch, max_epochs, freeze_epochs, warmup_epochs, base_head_lr, base_encoder_lr, min_lr_ratio)
            for epoch in range(1, max_epochs + 1)
        ]),
        "training_order_sha256": stable_hash({"indices": [int(x) for x in train_idx], "seed": seed, "formula": "seed+epoch*100003"}),
        "regression_target_mean": regression_mean,
        "regression_target_std": regression_std,
        **construction,
    }
    return model, history, audit, best_epoch


def sys_platform_is_macos() -> bool:
    return os.uname().sysname == "Darwin"


def flat_matrix(graphs: Sequence[Any], train_idx: np.ndarray) -> np.ndarray:
    descriptors = np.vstack([graph.global_feat.view(-1).float().numpy() for graph in graphs])
    mean = descriptors[train_idx].mean(axis=0, keepdims=True)
    std = descriptors[train_idx].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    descriptors = (descriptors - mean) / std
    morgan = np.vstack([graph.morgan_fp.view(-1).float().numpy() for graph in graphs])
    maccs = np.vstack([graph.maccs_fp.view(-1).float().numpy() for graph in graphs])
    return np.concatenate([descriptors, morgan, maccs], axis=1)


def sample_weights_from_meta(meta: pd.DataFrame) -> np.ndarray:
    weights = np.ones(len(meta), dtype=float)
    censored = meta["is_censored_gt50"].astype(int).eq(1).to_numpy()
    thresholds = pd.to_numeric(meta["censored_threshold_uM"], errors="coerce").to_numpy(float)
    weights[censored & np.isfinite(thresholds) & (thresholds >= 50)] = 0.8
    weights[censored & (~np.isfinite(thresholds) | (thresholds < 50))] = 0.5
    return weights


def make_flat_estimator(kind: str, params: dict[str, Any], seed: int) -> Any:
    if kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 700)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=int(params.get("n_estimators", 700)),
            max_features=params.get("max_features", "sqrt"),
            min_samples_leaf=int(params.get("min_samples_leaf", 2)),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "logistic_regression":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(params.get("C", 0.5)), max_iter=5000, class_weight="balanced", random_state=seed
            ),
        )
    if kind == "rbf_svm":
        return make_pipeline(
            StandardScaler(),
            SVC(
                C=float(params.get("C", 2.0)), gamma=params.get("gamma", "scale"), probability=True,
                class_weight="balanced", random_state=seed,
            ),
        )
    if kind == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 15)),
            l2_regularization=float(params.get("l2_regularization", 1.0)),
            random_state=seed,
        )
    raise ValueError(kind)


def fit_predict_flat(
    kind: str,
    params: dict[str, Any],
    graphs: Sequence[Any],
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, Any, dict[str, Any]]:
    matrix = flat_matrix(graphs, train_idx)
    y = labels_from_graphs(graphs)
    weights = sample_weights_from_meta(meta)
    estimator = make_flat_estimator(kind, params, seed)
    started = time.perf_counter()
    if kind in {"logistic_regression", "rbf_svm"}:
        estimator.fit(matrix[train_idx], y[train_idx], **{"svc__sample_weight" if kind == "rbf_svm" else "logisticregression__sample_weight": weights[train_idx]})
    else:
        estimator.fit(matrix[train_idx], y[train_idx], sample_weight=weights[train_idx])
    probability = estimator.predict_proba(matrix[valid_idx])[:, 1]
    audit = {
        "estimator": kind,
        "parameters": params,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_process_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * (1 if sys_platform_is_macos() else 1024),
    }
    return probability, estimator, audit


def embedding_diagnostics(embeddings: np.ndarray) -> dict[str, float]:
    embeddings = np.asarray(embeddings, dtype=float)
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    feature_std = centered.std(axis=0)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    variance = singular**2
    ratio = variance / max(variance.sum(), 1e-12)
    entropy = -np.sum(ratio[ratio > 0] * np.log(ratio[ratio > 0]))
    effective_rank = float(np.exp(entropy))
    normalized = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[np.triu_indices(len(cosine), k=1)]
    return {
        "embedding_median_feature_std": float(np.median(feature_std)),
        "effective_rank": effective_rank,
        "top1_singular_value_explained_ratio": float(ratio[0]) if len(ratio) else float("nan"),
        "mean_pairwise_cosine_similarity": float(off_diagonal.mean()) if len(off_diagonal) else float("nan"),
    }


def check_checkpoint_reload(
    model: FormalGINEModel,
    spec: NeuralSpec,
    params: dict[str, Any],
    checkpoint: Path,
    downstream_checkpoint: Path,
    fixed_graphs: Sequence[Any],
    device_name: str,
    seed: int,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    indices = np.arange(min(len(fixed_graphs), 8), dtype=int)
    before = predict_neural(model, fixed_graphs, indices, device, min(8, int(params["batch_size"])))[:2]
    payload = {
        "model_state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "spec": asdict(spec),
        "params": params,
        "seed": seed,
    }
    temporary = downstream_checkpoint.with_suffix(downstream_checkpoint.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, downstream_checkpoint)
    reloaded, _ = build_neural_model(spec, params, checkpoint, seed)
    reloaded.load_state_dict(torch.load(downstream_checkpoint, map_location="cpu", weights_only=False)["model_state_dict"], strict=True)
    reloaded.to(device)
    after = predict_neural(reloaded, fixed_graphs, indices, device, min(8, int(params["batch_size"])))[:2]
    exact = bool(np.array_equal(before[0], after[0], equal_nan=True) and np.array_equal(before[1], after[1], equal_nan=True))
    if not exact:
        raise RuntimeError("Downstream checkpoint reload changed fixed-batch outputs")
    return {"checkpoint_sha256": sha256_file(downstream_checkpoint), "exact_reload_output_match": exact}
