"""Development-only evaluation core for PharmaFusion-GPS-Adapter.

This module intentionally refuses any path containing ``locked_test``.  Model
selection uses explicit development indices and scaffold groups only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch_geometric.loader import DataLoader

from formal_1000_evaluation_core import (
    classification_metrics,
    fit_predict_flat,
    graph_sample_weights,
    grouped_folds,
    labels_from_graphs,
    set_all_seeds,
    training_order,
)
from gine_v2 import GINEEncoder, load_gine_checkpoint
from pharmafusion_gps_adapter import (
    PharmaFusionConfig,
    PharmaFusionGPSAdapter,
    attach_2d_structural_encodings,
)


FORBIDDEN_PATH_TOKEN = "locked_test"


def reject_forbidden_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if FORBIDDEN_PATH_TOKEN in str(resolved).lower():
        raise RuntimeError(f"Locked test access is forbidden in architecture development: {resolved}")
    return resolved


def sha256_file(path: str | Path) -> str:
    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def cosine_factor(epoch: int, total_epochs: int, warmup_epochs: int, min_ratio: float) -> tuple[float, str]:
    if epoch <= warmup_epochs:
        if warmup_epochs <= 1:
            return 1.0, "warmup"
        return 0.1 + 0.9 * (epoch - 1) / (warmup_epochs - 1), "warmup"
    decay_epochs = max(total_epochs - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / decay_epochs, 0.0), 1.0)
    factor = min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return factor, "cosine"


def validation_selection_start_epoch(spec: "ArchitectureSpec", training: "TrainingConfig") -> int:
    """Earliest epoch eligible for validation-based model selection.

    Staged models must complete both the frozen phase and the encoder warmup;
    all other trainable configurations must at least complete their warmup.
    """
    if spec.tuning_mode == "staged":
        return max(1, int(spec.freeze_epochs + training.warmup_epochs))
    return max(1, int(training.warmup_epochs))


@dataclass(frozen=True)
class ArchitectureSpec:
    name: str
    stage: str
    config: PharmaFusionConfig
    initialization: str = "pretrained"
    tuning_mode: str = "staged"
    backbone_lr: float = 5e-5
    new_module_lr: float = 5e-4
    freeze_epochs: int = 5

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config"] = self.config.to_dict()
        return payload


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 40
    patience: int = 7
    min_delta: float = 1e-4
    batch_size: int = 32
    weight_decay: float = 3e-3
    gradient_clip: float = 2.0
    warmup_epochs: int = 2
    min_lr_ratio: float = 0.05
    device: str = "auto"


def architecture_catalog(base: PharmaFusionConfig | None = None) -> dict[str, ArchitectureSpec]:
    base = base or PharmaFusionConfig(
        use_jk=False,
        pooling="mean",
        use_adapter=False,
        use_gps=False,
        fusion="none",
        head_type="mlp",
    )

    def spec(name: str, stage: str, **changes: Any) -> ArchitectureSpec:
        config_changes = changes.pop("config_changes", {})
        return ArchitectureSpec(name, stage, replace(base, **config_changes), **changes)

    catalog = {
        "A0": spec("A0 Current Pretrained GINE", "A", config_changes={"use_jk": False, "pooling": "mean"}),
        "A1": spec("A1 GINE + Weighted JK", "A", config_changes={"use_jk": True, "pooling": "mean"}),
        "A2": spec("A2 GINE + Attention Pool", "A", config_changes={"use_jk": False, "pooling": "attention"}),
        "A3": spec("A3 GINE + JK + Attention Pool", "A", config_changes={"use_jk": True, "pooling": "attention"}),
        "A4": spec("A4 GINE + JK + Multi-Pooling", "A", config_changes={"use_jk": True, "pooling": "multi"}),
    }
    return catalog


def stage_b_catalog(best_config: PharmaFusionConfig) -> dict[str, ArchitectureSpec]:
    result = {
        "B0": ArchitectureSpec(
            "B0 Frozen encoder linear probe", "B", replace(best_config, use_adapter=False, head_type="linear"), tuning_mode="frozen"
        ),
        "B1": ArchitectureSpec(
            "B1 Frozen encoder + MLP", "B", replace(best_config, use_adapter=False, head_type="mlp"), tuning_mode="frozen"
        ),
        "B2": ArchitectureSpec(
            "B2 Full fine-tune", "B", replace(best_config, use_adapter=False), tuning_mode="full"
        ),
        "B3": ArchitectureSpec(
            "B3 Staged fine-tune", "B", replace(best_config, use_adapter=False), tuning_mode="staged"
        ),
    }
    for adapter_dim in (8, 16, 32):
        result[f"B4_d{adapter_dim}"] = ArchitectureSpec(
            f"B4 Frozen GINE + Adapter d={adapter_dim}",
            "B",
            replace(best_config, use_adapter=True, adapter_dim=adapter_dim),
            tuning_mode="adapter",
        )
        result[f"B5_d{adapter_dim}"] = ArchitectureSpec(
            f"B5 Partial unfreeze + Adapter d={adapter_dim}",
            "B",
            replace(best_config, use_adapter=True, adapter_dim=adapter_dim),
            tuning_mode="partial_adapter",
            backbone_lr=1.5e-5,
        )
    return result


def stage_c_catalog(best: ArchitectureSpec) -> dict[str, ArchitectureSpec]:
    base = replace(best.config, use_gps=False, gps_use_rwse=False, gps_use_spd=False)
    entries = {
        "C0": ("C0 No GPS", False, False, False),
        "C1": ("C1 GPS only", True, False, False),
        "C2": ("C2 GPS + RWSE", True, True, False),
        "C3": ("C3 GPS + shortest-path bias", True, False, True),
        "C4": ("C4 GPS + RWSE + shortest-path bias", True, True, True),
    }
    return {
        key: replace(
            best,
            name=name,
            stage="C",
            config=replace(base, use_gps=gps, gps_use_rwse=rwse, gps_use_spd=spd),
        )
        for key, (name, gps, rwse, spd) in entries.items()
    }


def stage_d_catalog(best: ArchitectureSpec) -> dict[str, ArchitectureSpec]:
    entries = {
        "D0": ("D0 Pure graph", "none", 0.0),
        "D1": ("D1 Graph + RDKit direct concat", "concat", 0.0),
        "D2": ("D2 Graph + RDKit gated residual", "gated", 0.0),
        "D3": ("D3 Graph + RDKit gated residual + modality dropout", "gated_dropout", 0.15),
    }
    return {
        key: replace(best, name=name, stage="D", config=replace(best.config, fusion=fusion, modality_dropout=dropout))
        for key, (name, fusion, dropout) in entries.items()
    }


def prepare_development_graphs(graphs: Sequence[Any]) -> list[Any]:
    prepared: list[Any] = []
    for graph in graphs:
        item = copy.deepcopy(graph)
        attach_2d_structural_encodings(item, rwse_steps=16, spd_clip=10)
        prepared.append(item)
    return prepared


def fit_descriptor_scaler(graphs: Sequence[Any], train_indices: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    values = torch.cat([graphs[int(index)].global_feat.float().view(1, -1) for index in train_indices], dim=0)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Non-finite RDKit descriptor encountered in development training fold")
    mean = values.mean(dim=0)
    scale = values.std(dim=0, unbiased=False).clamp(min=1e-8)
    audit = {
        "fit_sample_count": int(len(train_indices)),
        "fit_index_sha256": stable_hash(sorted(int(value) for value in train_indices)),
        "mean_sha256": hashlib.sha256(mean.numpy().tobytes()).hexdigest(),
        "scale_sha256": hashlib.sha256(scale.numpy().tobytes()).hexdigest(),
    }
    return mean, scale, audit


def parameter_hash(parameters: Iterable[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        value = parameter.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def build_model(
    spec: ArchitectureSpec,
    checkpoint: Path,
    seed: int,
    descriptor_scaler: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[PharmaFusionGPSAdapter, dict[str, Any]]:
    set_all_seeds(seed)
    encoder = GINEEncoder(
        hidden_dim=256,
        num_layers=5,
        dropout=spec.config.encoder_dropout,
        use_layer_norm=True,
    )
    model = PharmaFusionGPSAdapter(encoder, spec.config)
    backbone_keys_before = tuple(encoder.state_dict())
    transfer: dict[str, Any] = {
        "load_ratio": float("nan"),
        "missing_keys": [],
        "unexpected_keys": [],
        "shape_mismatches": [],
        "exact_value_match_after_load": False,
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
            raise RuntimeError(f"Inexact checkpoint transfer for {spec.name}: {transfer}")
    if tuple(encoder.state_dict()) != backbone_keys_before:
        raise RuntimeError("GINE state-dict key names changed during PharmaFusion construction")
    if spec.config.fusion != "none":
        if descriptor_scaler is None:
            raise ValueError("Descriptor fusion requires a training-fold-only scaler")
        model.set_descriptor_scaler(*descriptor_scaler)

    if spec.tuning_mode in {"frozen", "adapter"}:
        model.set_backbone_mode("frozen")
    elif spec.tuning_mode == "partial_adapter":
        model.set_backbone_mode("partial")
    elif spec.tuning_mode == "full":
        model.set_backbone_mode("full")
    elif spec.tuning_mode == "staged":
        model.set_backbone_mode("frozen")
    else:
        raise ValueError(spec.tuning_mode)

    backbone_ids = {id(parameter) for parameter in model.gine_encoder.parameters()}
    new_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    audit = {
        "spec": spec.to_dict(),
        "transfer": transfer,
        **model.parameter_audit(),
        "new_module_initial_sha256": parameter_hash(new_parameters),
    }
    return model, audit


def _clone_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


def _changed_values(before: dict[str, torch.Tensor], module: nn.Module) -> int:
    current = module.state_dict()
    return sum(
        int(torch.count_nonzero(value != current[name].detach().cpu()).item())
        for name, value in before.items()
        if name in current
    )


@torch.no_grad()
def predict_model(
    model: PharmaFusionGPSAdapter,
    graphs: Sequence[Any],
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    export_details: bool = False,
) -> dict[str, Any]:
    model.eval()
    probabilities: list[float] = []
    embeddings = {f"E{index}": [] for index in range(5)}
    attention_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    cursor = 0
    loader = DataLoader([graphs[int(index)] for index in indices], batch_size=batch_size, shuffle=False)
    for batch_values in loader:
        batch_values = batch_values.to(device)
        output = model(batch_values, return_stages=True)
        probabilities.extend(torch.sigmoid(output["logits"]).detach().cpu().tolist())
        for key, values in output["stage_embeddings"].items():
            embeddings[key].append(values.detach().cpu().numpy())
        if export_details:
            weights = output["atom_attention"].detach().cpu()
            graph_batch = batch_values.batch.detach().cpu()
            gates = output["fusion_gate"].detach().cpu()
            for local_graph in range(int(batch_values.num_graphs)):
                development_index = int(indices[cursor + local_graph])
                atom_values = weights[graph_batch == local_graph]
                for atom_index, weight in enumerate(atom_values.tolist()):
                    attention_rows.append(
                        {"development_index": development_index, "atom_index": atom_index, "attention_weight": float(weight)}
                    )
                gate_rows.append(
                    {
                        "development_index": development_index,
                        "gate_mean": float(gates[local_graph].mean()),
                        "gate_std": float(gates[local_graph].std(unbiased=False)),
                    }
                )
            cursor += int(batch_values.num_graphs)
    return {
        "probability": np.asarray(probabilities, dtype=float),
        "embeddings": {key: np.concatenate(value, axis=0) for key, value in embeddings.items()},
        "attention_rows": attention_rows,
        "gate_rows": gate_rows,
        "jk_weights": model.jk.normalized_weights().detach().cpu().numpy() if model.jk is not None else np.array([0, 0, 0, 0, 1.0]),
    }


def train_model(
    spec: ArchitectureSpec,
    graphs: Sequence[Any],
    train_indices: np.ndarray,
    checkpoint: Path,
    seed: int,
    training: TrainingConfig,
    valid_indices: np.ndarray | None = None,
    fixed_epochs: int | None = None,
) -> tuple[PharmaFusionGPSAdapter, list[dict[str, Any]], dict[str, Any], int]:
    device = resolve_device(training.device)
    descriptor_mean = descriptor_scale = None
    descriptor_audit: dict[str, Any] = {}
    if spec.config.fusion != "none":
        descriptor_mean, descriptor_scale, descriptor_audit = fit_descriptor_scaler(graphs, train_indices)
    model, construction = build_model(
        spec,
        checkpoint,
        seed,
        None if descriptor_mean is None else (descriptor_mean, descriptor_scale),
    )
    initial_backbone = _clone_state(model.gine_encoder)
    backbone_parameters = list(model.gine_encoder.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    new_parameters = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids and parameter.requires_grad]
    if not new_parameters:
        raise RuntimeError("No trainable PharmaFusion modules")
    optimizer = torch.optim.AdamW(
        [{"params": new_parameters, "lr": spec.new_module_lr, "group_name": "new", "weight_decay": training.weight_decay}]
    )
    if spec.tuning_mode in {"full", "partial_adapter"}:
        trainable_backbone = [parameter for parameter in backbone_parameters if parameter.requires_grad]
        optimizer.add_param_group(
            {
                "params": trainable_backbone,
                "lr": spec.backbone_lr,
                "group_name": "backbone",
                "weight_decay": training.weight_decay,
            }
        )
    model.to(device)

    train_y = labels_from_graphs([graphs[int(index)] for index in train_indices])
    positive_weight = torch.tensor(
        [(len(train_y) - int(train_y.sum())) / max(int(train_y.sum()), 1)],
        dtype=torch.float32,
        device=device,
    )
    epoch_limit = int(fixed_epochs or training.max_epochs)
    if epoch_limit < 1 or epoch_limit > training.max_epochs:
        raise ValueError("fixed_epochs must be between 1 and training.max_epochs")
    # A fixed-epoch outer refit must follow the prefix of the same LR trajectory
    # used for inner model selection.  Shortening the cosine horizon to the
    # selected epoch would decay too aggressively and under-train the refit.
    scheduler_horizon = int(training.max_epochs)
    selection_start = validation_selection_start_epoch(spec, training)
    best_epoch = 0
    best_pr_auc = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    optimizer_step_seconds = 0.0
    gradient_nonzero_batches = 0
    gradient_batches = 0
    started = time.perf_counter()

    for epoch in range(1, epoch_limit + 1):
        if spec.tuning_mode == "staged" and epoch == spec.freeze_epochs + 1:
            model.set_backbone_mode("full")
            optimizer.add_param_group(
                {
                    "params": list(model.gine_encoder.parameters()),
                    "lr": spec.backbone_lr,
                    "group_name": "backbone",
                    "weight_decay": training.weight_decay,
                }
            )

        new_factor, new_phase = cosine_factor(epoch, scheduler_horizon, training.warmup_epochs, training.min_lr_ratio)
        if spec.tuning_mode == "staged":
            local_backbone_epoch = epoch - spec.freeze_epochs
            if local_backbone_epoch <= 0:
                backbone_factor, backbone_phase = 0.0, "frozen"
            else:
                backbone_factor, backbone_phase = cosine_factor(
                    local_backbone_epoch,
                    max(scheduler_horizon - spec.freeze_epochs, 1),
                    training.warmup_epochs,
                    training.min_lr_ratio,
                )
        elif spec.tuning_mode in {"frozen", "adapter"}:
            backbone_factor, backbone_phase = 0.0, "frozen"
        else:
            backbone_factor, backbone_phase = new_factor, new_phase
        for group in optimizer.param_groups:
            if group["group_name"] == "new":
                group["lr"] = spec.new_module_lr * new_factor
            else:
                group["lr"] = spec.backbone_lr * backbone_factor

        model.train()
        if spec.tuning_mode in {"frozen", "adapter"} or (spec.tuning_mode == "staged" and epoch <= spec.freeze_epochs):
            model.gine_encoder.eval()
        set_all_seeds(seed + epoch * 1009)
        ordered = training_order(train_indices, seed, epoch)
        loader = DataLoader([graphs[int(index)] for index in ordered], batch_size=training.batch_size, shuffle=False)
        epoch_loss = 0.0
        epoch_samples = 0
        epoch_backbone_norm = 0.0
        epoch_new_norm = 0.0
        for batch_values in loader:
            batch_values = batch_values.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch_values)
            target = batch_values.y_cls.view(-1).float()
            loss_values = F.binary_cross_entropy_with_logits(
                output["logits"], target, reduction="none", pos_weight=positive_weight
            )
            sample_weights = graph_sample_weights(batch_values)
            loss = (loss_values * sample_weights).sum() / sample_weights.sum().clamp(min=1e-8)
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
                gradient_batches += 1
                gradient_nonzero_batches += int(backbone_norm_sq > 0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip)
            step_started = time.perf_counter()
            optimizer.step()
            optimizer_step_seconds += time.perf_counter() - step_started
            optimizer_steps += 1
            samples = int(target.numel())
            epoch_loss += float(loss.detach().cpu()) * samples
            epoch_samples += samples
            epoch_backbone_norm += math.sqrt(backbone_norm_sq)
            epoch_new_norm += math.sqrt(new_norm_sq)

        validation_pr = float("nan")
        if valid_indices is not None:
            prediction = predict_model(model, graphs, valid_indices, training.batch_size, device)
            valid_y = labels_from_graphs([graphs[int(index)] for index in valid_indices])
            validation_pr = float(average_precision_score(valid_y, prediction["probability"]))
            selection_eligible = epoch >= selection_start
            if selection_eligible and validation_pr > best_pr_auc + training.min_delta:
                best_pr_auc = validation_pr
                best_epoch = epoch
                best_state = _clone_state(model)
                stale = 0
            elif selection_eligible:
                stale += 1

        group_lrs = {str(group["group_name"]): float(group["lr"]) for group in optimizer.param_groups}
        history.append(
            {
                "epoch": epoch,
                "training_loss": epoch_loss / max(epoch_samples, 1),
                "validation_pr_auc": validation_pr,
                "new_module_lr": group_lrs.get("new", 0.0),
                "backbone_lr": group_lrs.get("backbone", 0.0),
                "new_scheduler_phase": new_phase,
                "backbone_scheduler_phase": backbone_phase,
                "backbone_frozen": int(not any(parameter.requires_grad for parameter in backbone_parameters)),
                "backbone_gradient_norm": epoch_backbone_norm / max(len(loader), 1),
                "new_module_gradient_norm": epoch_new_norm / max(len(loader), 1),
            }
        )
        if valid_indices is not None and epoch >= selection_start and stale >= training.patience:
            break

    if valid_indices is not None:
        if best_state is None:
            raise RuntimeError("No valid inner-validation checkpoint was selected")
        model.load_state_dict(best_state)
    else:
        best_epoch = epoch_limit
    elapsed = time.perf_counter() - started
    construction.update(
        {
            "descriptor_scaler": descriptor_audit,
            "best_epoch": int(best_epoch),
            "optimizer_updates": optimizer_steps,
            "mean_optimizer_step_seconds": optimizer_step_seconds / max(optimizer_steps, 1),
            "training_seconds": elapsed,
            "peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024)
            ),
            "backbone_changed_values": _changed_values(initial_backbone, model.gine_encoder),
            "backbone_gradient_nonzero_fraction": gradient_nonzero_batches / max(gradient_batches, 1),
        }
    )
    return model, history, construction, int(best_epoch)


def fixed_outer_folds(assignments: pd.DataFrame, seed: int = 13) -> list[tuple[np.ndarray, np.ndarray]]:
    required = {"seed", "fold", "model_index", "assignment", "scaffold_group"}
    if not required.issubset(assignments.columns):
        raise ValueError(f"Fold manifest is missing columns: {sorted(required - set(assignments.columns))}")
    subset = assignments[(assignments["seed"] == seed) & (assignments["assignment"] == "outer_validation")]
    all_indices = np.sort(assignments[assignments["seed"] == seed]["model_index"].unique().astype(int))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in sorted(subset["fold"].unique()):
        valid = np.sort(subset[subset["fold"] == fold]["model_index"].to_numpy(dtype=int))
        train = np.setdiff1d(all_indices, valid)
        folds.append((train, valid))
    if len(folds) != 5 or len(np.unique(np.concatenate([valid for _, valid in folds]))) != len(all_indices):
        raise RuntimeError("Fixed development outer-fold manifest is incomplete")
    return folds


def nested_evaluate(
    spec: ArchitectureSpec,
    graphs: Sequence[Any],
    groups: np.ndarray,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    checkpoint: Path,
    model_seed: int,
    training: TrainingConfig,
    output_dir: Path,
    save_checkpoints: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels_from_graphs(graphs)
    prediction = np.full(len(graphs), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    jk_rows: list[dict[str, Any]] = []
    attention_rows: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    embedding_store = {f"E{index}": np.full((len(graphs), 256), np.nan, dtype=np.float32) for index in range(5)}

    for outer_fold, (outer_train, outer_valid) in enumerate(outer_folds):
        print(
            f"[{spec.stage}] {spec.name}: outer fold {outer_fold + 1}/{len(outer_folds)}",
            flush=True,
        )
        inner_local = grouped_folds(labels[outer_train], groups[outer_train], folds=3, seed=13000 + outer_fold)
        inner_epochs: list[int] = []
        inner_scores: list[float] = []
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(inner_local):
            print(f"  inner fold {inner_fold + 1}/3", flush=True)
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            inner_model, inner_history, inner_audit, best_epoch = train_model(
                spec,
                graphs,
                inner_train,
                checkpoint,
                seed=model_seed + outer_fold * 101 + inner_fold * 10007,
                training=training,
                valid_indices=inner_valid,
            )
            inner_prediction = predict_model(
                inner_model,
                graphs,
                inner_valid,
                training.batch_size,
                resolve_device(training.device),
            )["probability"]
            inner_pr = float(average_precision_score(labels[inner_valid], inner_prediction))
            inner_epochs.append(best_epoch)
            inner_scores.append(inner_pr)
            for row in inner_history:
                training_rows.append(
                    {
                        "model": spec.name,
                        "seed": model_seed,
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "run_type": "inner",
                        **row,
                    }
                )
            del inner_model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        selected_epoch = max(1, int(round(float(np.median(inner_epochs)))))
        outer_model, outer_history, outer_audit, _ = train_model(
            spec,
            graphs,
            outer_train,
            checkpoint,
            seed=model_seed + outer_fold * 101,
            training=training,
            fixed_epochs=selected_epoch,
        )
        details = predict_model(
            outer_model,
            graphs,
            outer_valid,
            training.batch_size,
            resolve_device(training.device),
            export_details=True,
        )
        prediction[outer_valid] = details["probability"]
        for key in embedding_store:
            embedding_store[key][outer_valid] = details["embeddings"][key]
        metrics = classification_metrics(labels[outer_valid], details["probability"])
        fold_rows.append(
            {
                "model": spec.name,
                "stage": spec.stage,
                "seed": model_seed,
                "outer_fold": outer_fold,
                "selected_epoch": selected_epoch,
                "inner_mean_pr_auc": float(np.mean(inner_scores)),
                "inner_pr_auc_sd": float(np.std(inner_scores, ddof=1)),
                **metrics,
            }
        )
        for row in outer_history:
            training_rows.append(
                {
                    "model": spec.name,
                    "seed": model_seed,
                    "outer_fold": outer_fold,
                    "inner_fold": -1,
                    "run_type": "outer_refit",
                    **row,
                }
            )
        parameter_rows.append({"model": spec.name, "seed": model_seed, "outer_fold": outer_fold, **outer_audit})
        for layer, weight in enumerate(details["jk_weights"], start=1):
            jk_rows.append({"model": spec.name, "seed": model_seed, "outer_fold": outer_fold, "layer": layer, "weight": float(weight)})
        attention_rows.extend(
            {"model": spec.name, "seed": model_seed, "outer_fold": outer_fold, **row}
            for row in details["attention_rows"]
        )
        gate_rows.extend(
            {"model": spec.name, "seed": model_seed, "outer_fold": outer_fold, **row}
            for row in details["gate_rows"]
        )
        if save_checkpoints:
            torch.save(
                {
                    "model_state_dict": outer_model.state_dict(),
                    "spec": spec.to_dict(),
                    "selected_epoch": selected_epoch,
                    "seed": model_seed,
                    "outer_fold": outer_fold,
                },
                output_dir / f"seed_{model_seed}_fold_{outer_fold}.pt",
            )

    if bool(np.isnan(prediction).any()):
        raise RuntimeError(f"Incomplete OOF prediction for {spec.name}")
    pooled_metrics = classification_metrics(labels, prediction)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output_dir / "training_audit.csv", index=False)
    pd.DataFrame(parameter_rows).to_json(output_dir / "parameter_audit.jsonl", orient="records", lines=True)
    pd.DataFrame(jk_rows).to_csv(output_dir / "jk_weights.csv", index=False)
    pd.DataFrame(attention_rows).to_csv(output_dir / "atom_attention.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(output_dir / "fusion_gates.csv", index=False)
    np.savez_compressed(output_dir / "oof_embeddings.npz", **embedding_store)
    pd.DataFrame(
        {
            "development_index": np.arange(len(graphs), dtype=int),
            "label": labels,
            "probability": prediction,
            "scaffold_group": groups,
        }
    ).to_csv(output_dir / "oof_predictions.csv", index=False)
    summary = {
        "model": spec.name,
        "stage": spec.stage,
        "seed": model_seed,
        "pooled_metrics": pooled_metrics,
        "fold_pr_auc_mean": float(pd.DataFrame(fold_rows)["pr_auc"].mean()),
        "fold_pr_auc_sd": float(pd.DataFrame(fold_rows)["pr_auc"].std(ddof=1)),
        "spec": spec.to_dict(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def embedding_classifier_grids(seed: int = 13) -> dict[str, list[dict[str, Any]]]:
    svm = [
        {"C": c, "gamma": gamma, "class_weight": class_weight}
        for c in (0.1, 1.0, 10.0, 100.0)
        for gamma in ("scale", 0.001, 0.01, 0.1)
        for class_weight in (None, "balanced")
    ]
    logistic = [{"C": c} for c in (0.01, 0.1, 1.0, 10.0, 100.0)]
    forest = [
        {"n_estimators": trees, "max_features": "sqrt", "min_samples_leaf": leaf}
        for trees in (300, 700)
        for leaf in (1, 2)
    ]
    xgboost_all = list(
        ParameterGrid(
            {
                "max_depth": [2, 3, 4],
                "learning_rate": [0.01, 0.03, 0.05],
                "n_estimators": [300, 600, 1000],
                "subsample": [0.7, 0.9, 1.0],
                "colsample_bytree": [0.5, 0.75, 1.0],
            }
        )
    )
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(len(xgboost_all), size=30, replace=False))
    xgboost = [xgboost_all[int(index)] for index in selected]
    return {
        "Logistic Regression": logistic,
        "RBF-SVM": svm,
        "Random Forest": forest,
        "XGBoost": xgboost,
    }


def make_embedding_classifier(kind: str, params: dict[str, Any], seed: int) -> Any:
    if kind == "Logistic Regression":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(C=float(params["C"]), max_iter=5000, random_state=seed)),
            ]
        )
    if kind == "RBF-SVM":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(
                        C=float(params["C"]),
                        gamma=params["gamma"],
                        class_weight=params["class_weight"],
                        probability=True,
                        random_state=seed,
                    ),
                ),
            ]
        )
    if kind == "Random Forest":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if kind == "XGBoost":
        raise RuntimeError("XGBoost must run through the isolated worker to avoid a PyTorch runtime collision")
    raise ValueError(kind)


def run_xgboost_worker(
    mode: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: np.ndarray,
    params: list[dict[str, Any]],
    seed: int,
    valid_y: np.ndarray | None = None,
) -> np.ndarray:
    """Run XGBoost without importing PyTorch and XGBoost in one process."""
    worker = Path(__file__).resolve().parents[2] / "scripts" / "xgboost_embedding_worker.py"
    with tempfile.TemporaryDirectory(prefix="pharmafusion_xgb_") as temporary:
        temporary_path = Path(temporary)
        input_path = temporary_path / "input.npz"
        params_path = temporary_path / "params.json"
        output_path = temporary_path / "output.npy"
        payload: dict[str, Any] = {"train_x": train_x, "train_y": train_y, "valid_x": valid_x}
        if valid_y is not None:
            payload["valid_y"] = valid_y
        np.savez_compressed(input_path, **payload)
        params_path.write_text(json.dumps(params, sort_keys=True), encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(worker),
                "--mode",
                mode,
                "--input",
                str(input_path),
                "--params",
                str(params_path),
                "--output",
                str(output_path),
                "--seed",
                str(seed),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Isolated XGBoost worker failed ({completed.returncode}): {completed.stderr[-2000:]}"
            )
        return np.load(output_path)


def evaluate_embedding_classifiers(
    spec: ArchitectureSpec,
    graphs: Sequence[Any],
    groups: np.ndarray,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    checkpoint: Path,
    model_seed: int,
    training: TrainingConfig,
    output_dir: Path,
    embedding_key: str = "E4",
) -> dict[str, Any]:
    """Nested readout comparison with all preprocessing inside inner training."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels_from_graphs(graphs)
    grids = embedding_classifier_grids(model_seed)
    oof = {kind: np.full(len(graphs), np.nan, dtype=float) for kind in grids}
    fold_rows: list[dict[str, Any]] = []
    hyperparameter_rows: list[dict[str, Any]] = []
    deep_training_rows: list[dict[str, Any]] = []

    for outer_fold, (outer_train, outer_valid) in enumerate(outer_folds):
        print(f"[E] embedding classifiers: outer fold {outer_fold + 1}/{len(outer_folds)}", flush=True)
        inner_local = grouped_folds(labels[outer_train], groups[outer_train], 3, seed=23000 + outer_fold)
        inner_data: list[dict[str, Any]] = []
        inner_epochs: list[int] = []
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(inner_local):
            print(f"  deep embedding inner fold {inner_fold + 1}/3", flush=True)
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            model, history, _, best_epoch = train_model(
                spec,
                graphs,
                inner_train,
                checkpoint,
                seed=model_seed + outer_fold * 101 + inner_fold * 10007,
                training=training,
                valid_indices=inner_valid,
            )
            device = resolve_device(training.device)
            train_embedding = predict_model(model, graphs, inner_train, training.batch_size, device)["embeddings"][embedding_key]
            valid_embedding = predict_model(model, graphs, inner_valid, training.batch_size, device)["embeddings"][embedding_key]
            inner_data.append(
                {
                    "train_x": train_embedding,
                    "train_y": labels[inner_train],
                    "valid_x": valid_embedding,
                    "valid_y": labels[inner_valid],
                }
            )
            inner_epochs.append(best_epoch)
            for row in history:
                deep_training_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "run_type": "classifier_inner_encoder",
                        **row,
                    }
                )
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        selected_params: dict[str, dict[str, Any]] = {}
        for kind, grid in grids.items():
            print(f"  classifier search: {kind} ({len(grid)} candidates)", flush=True)
            scores: list[tuple[float, dict[str, Any], list[float]]] = []
            if kind == "XGBoost":
                scores_by_param: list[list[float]] = [[] for _ in grid]
                for inner_fold, values in enumerate(inner_data):
                    worker_scores = run_xgboost_worker(
                        "grid",
                        values["train_x"],
                        values["train_y"],
                        values["valid_x"],
                        grid,
                        model_seed + outer_fold * 101 + inner_fold,
                        valid_y=values["valid_y"],
                    )
                    for param_index, score in enumerate(worker_scores.tolist()):
                        scores_by_param[param_index].append(float(score))
                iterator = [(index, params, scores_by_param[index]) for index, params in enumerate(grid)]
            else:
                iterator = []
                for param_index, params in enumerate(grid):
                    inner_scores = []
                    for inner_fold, values in enumerate(inner_data):
                        estimator = make_embedding_classifier(kind, params, model_seed + outer_fold * 101 + inner_fold)
                        estimator.fit(values["train_x"], values["train_y"])
                        probability = estimator.predict_proba(values["valid_x"])[:, 1]
                        inner_scores.append(float(average_precision_score(values["valid_y"], probability)))
                    iterator.append((param_index, params, inner_scores))
            for param_index, params, inner_scores in iterator:
                scores.append((float(np.mean(inner_scores)), params, inner_scores))
                hyperparameter_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "classifier": kind,
                        "parameter_index": param_index,
                        "params_json": json.dumps(params, sort_keys=True),
                        "inner_mean_pr_auc": float(np.mean(inner_scores)),
                        "inner_pr_auc_sd": float(np.std(inner_scores, ddof=1)),
                    }
                )
            # Stable tie break retains the earliest, simpler grid entry.
            selected_params[kind] = max(scores, key=lambda item: item[0])[1]

        selected_epoch = max(1, int(round(float(np.median(inner_epochs)))))
        outer_model, outer_history, _, _ = train_model(
            spec,
            graphs,
            outer_train,
            checkpoint,
            seed=model_seed + outer_fold * 101,
            training=training,
            fixed_epochs=selected_epoch,
        )
        device = resolve_device(training.device)
        train_x = predict_model(outer_model, graphs, outer_train, training.batch_size, device)["embeddings"][embedding_key]
        valid_x = predict_model(outer_model, graphs, outer_valid, training.batch_size, device)["embeddings"][embedding_key]
        for kind, params in selected_params.items():
            if kind == "XGBoost":
                probability = run_xgboost_worker(
                    "predict",
                    train_x,
                    labels[outer_train],
                    valid_x,
                    [params],
                    model_seed + outer_fold * 101,
                )
            else:
                estimator = make_embedding_classifier(kind, params, model_seed + outer_fold * 101)
                estimator.fit(train_x, labels[outer_train])
                probability = estimator.predict_proba(valid_x)[:, 1]
            oof[kind][outer_valid] = probability
            fold_rows.append(
                {
                    "classifier": kind,
                    "seed": model_seed,
                    "outer_fold": outer_fold,
                    "selected_epoch": selected_epoch,
                    "selected_params_json": json.dumps(params, sort_keys=True),
                    **classification_metrics(labels[outer_valid], probability),
                }
            )
        for row in outer_history:
            deep_training_rows.append(
                {"outer_fold": outer_fold, "inner_fold": -1, "run_type": "classifier_outer_encoder", **row}
            )

    for kind, probability in oof.items():
        if bool(np.isnan(probability).any()):
            raise RuntimeError(f"Incomplete embedding-classifier OOF prediction: {kind}")
    metric_rows = [
        {"classifier": kind, **classification_metrics(labels, probability)}
        for kind, probability in oof.items()
    ]
    pd.DataFrame(metric_rows).to_csv(output_dir / "embedding_classifier_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "embedding_classifier_fold_metrics.csv", index=False)
    pd.DataFrame(hyperparameter_rows).to_csv(output_dir / "embedding_classifier_hyperparameters.csv", index=False)
    pd.DataFrame(deep_training_rows).to_csv(output_dir / "embedding_classifier_training_audit.csv", index=False)
    prediction_frame = pd.DataFrame({"development_index": np.arange(len(graphs)), "label": labels, "scaffold_group": groups})
    for kind, probability in oof.items():
        prediction_frame[kind] = probability
    prediction_frame.to_csv(output_dir / "embedding_classifier_oof_predictions.csv", index=False)
    selected_kind = str(pd.DataFrame(metric_rows).sort_values("pr_auc", ascending=False).iloc[0]["classifier"])
    result = {
        "selected_classifier": selected_kind,
        "selected_metrics": next(row for row in metric_rows if row["classifier"] == selected_kind),
        "embedding_key": embedding_key,
        "spec": spec.to_dict(),
    }
    (output_dir / "stage_E_selection.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def paired_initialization_evaluate(
    pretrained_spec: ArchitectureSpec,
    scratch_spec: ArchitectureSpec,
    graphs: Sequence[Any],
    groups: np.ndarray,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    checkpoint: Path,
    model_seed: int,
    training: TrainingConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Symmetric matched comparison with one shared inner-selected epoch."""
    pre_payload = pretrained_spec.to_dict()
    scratch_payload = scratch_spec.to_dict()
    for key in ("name", "initialization"):
        pre_payload.pop(key, None)
        scratch_payload.pop(key, None)
    if pre_payload != scratch_payload:
        raise ValueError("Matched initialization specs differ beyond name/initialization")
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels_from_graphs(graphs)
    predictions = {
        "pretrained": np.full(len(graphs), np.nan, dtype=float),
        "scratch_matched": np.full(len(graphs), np.nan, dtype=float),
    }
    fold_rows: list[dict[str, Any]] = []
    fairness_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    exhaustive_training = replace(training, patience=training.max_epochs + 1)

    for outer_fold, (outer_train, outer_valid) in enumerate(outer_folds):
        print(f"[F matched] outer fold {outer_fold + 1}/{len(outer_folds)}", flush=True)
        inner_local = grouped_folds(labels[outer_train], groups[outer_train], 3, seed=33000 + outer_fold)
        validation_by_epoch = np.zeros(training.max_epochs, dtype=float)
        new_hash_pairs: list[tuple[str, str]] = []
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(inner_local):
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            histories = {}
            audits = {}
            for label, spec in (("pretrained", pretrained_spec), ("scratch_matched", scratch_spec)):
                _, history, audit, _ = train_model(
                    spec,
                    graphs,
                    inner_train,
                    checkpoint,
                    seed=model_seed + outer_fold * 101 + inner_fold * 10007,
                    training=exhaustive_training,
                    valid_indices=inner_valid,
                )
                if len(history) != training.max_epochs:
                    raise RuntimeError("Matched inner trajectory ended before the common epoch search budget")
                histories[label] = history
                audits[label] = audit
                validation_by_epoch += np.asarray([float(row["validation_pr_auc"]) for row in history])
                for row in history:
                    training_rows.append(
                        {
                            "pair_member": label,
                            "outer_fold": outer_fold,
                            "inner_fold": inner_fold,
                            "run_type": "matched_inner_common_epoch",
                            **row,
                        }
                    )
            new_hash_pairs.append(
                (audits["pretrained"]["new_module_initial_sha256"], audits["scratch_matched"]["new_module_initial_sha256"])
            )
            pre_lr = [(row["new_module_lr"], row["backbone_lr"]) for row in histories["pretrained"]]
            scratch_lr = [(row["new_module_lr"], row["backbone_lr"]) for row in histories["scratch_matched"]]
            fairness_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "check": "inner_lr_trajectory_identical",
                    "passed": pre_lr == scratch_lr,
                }
            )
        # Average over both initialization cells and all three inner folds.
        common_epoch = int(np.nanargmax(validation_by_epoch) + 1)
        outer_histories = {}
        outer_audits = {}
        for label, spec in (("pretrained", pretrained_spec), ("scratch_matched", scratch_spec)):
            model, history, audit, _ = train_model(
                spec,
                graphs,
                outer_train,
                checkpoint,
                seed=model_seed + outer_fold * 101,
                training=training,
                fixed_epochs=common_epoch,
            )
            probability = predict_model(
                model,
                graphs,
                outer_valid,
                training.batch_size,
                resolve_device(training.device),
            )["probability"]
            predictions[label][outer_valid] = probability
            fold_rows.append(
                {
                    "pair_member": label,
                    "model": spec.name,
                    "seed": model_seed,
                    "outer_fold": outer_fold,
                    "common_selected_epoch": common_epoch,
                    **classification_metrics(labels[outer_valid], probability),
                }
            )
            outer_histories[label] = history
            outer_audits[label] = audit
            for row in history:
                training_rows.append(
                    {
                        "pair_member": label,
                        "outer_fold": outer_fold,
                        "inner_fold": -1,
                        "run_type": "matched_outer_refit",
                        **row,
                    }
                )
        pre_lr = [(row["new_module_lr"], row["backbone_lr"]) for row in outer_histories["pretrained"]]
        scratch_lr = [(row["new_module_lr"], row["backbone_lr"]) for row in outer_histories["scratch_matched"]]
        fairness_rows.extend(
            [
                {
                    "outer_fold": outer_fold,
                    "inner_fold": -1,
                    "check": "outer_lr_trajectory_identical",
                    "passed": pre_lr == scratch_lr,
                },
                {
                    "outer_fold": outer_fold,
                    "inner_fold": -1,
                    "check": "new_module_initialization_identical",
                    "passed": outer_audits["pretrained"]["new_module_initial_sha256"]
                    == outer_audits["scratch_matched"]["new_module_initial_sha256"],
                },
                {
                    "outer_fold": outer_fold,
                    "inner_fold": -1,
                    "check": "common_epoch_identical",
                    "passed": len(outer_histories["pretrained"]) == len(outer_histories["scratch_matched"]),
                },
            ]
        )
        fairness_rows.extend(
            {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "check": "inner_new_module_initialization_identical",
                "passed": pre_hash == scratch_hash,
            }
            for inner_fold, (pre_hash, scratch_hash) in enumerate(new_hash_pairs)
        )

    fairness = pd.DataFrame(fairness_rows)
    if not bool(fairness["passed"].all()):
        raise RuntimeError("Matched initialization fairness audit failed")
    for label, probability in predictions.items():
        if bool(np.isnan(probability).any()):
            raise RuntimeError(f"Incomplete matched OOF prediction: {label}")
    metrics = {
        label: classification_metrics(labels, probability)
        for label, probability in predictions.items()
    }
    prediction_frame = pd.DataFrame(
        {"development_index": np.arange(len(graphs)), "label": labels, "scaffold_group": groups, **predictions}
    )
    prediction_frame.to_csv(output_dir / "matched_oof_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "matched_fold_metrics.csv", index=False)
    fairness.to_csv(output_dir / "matched_fairness_checks.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output_dir / "matched_training_audit.csv", index=False)
    result = {
        "pretrained_metrics": metrics["pretrained"],
        "scratch_matched_metrics": metrics["scratch_matched"],
        "pr_auc_delta_pretrained_minus_scratch_matched": metrics["pretrained"]["pr_auc"]
        - metrics["scratch_matched"]["pr_auc"],
        "fairness_all_passed": True,
        "pretrained_spec": pretrained_spec.to_dict(),
        "scratch_spec": scratch_spec.to_dict(),
    }
    (output_dir / "matched_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def evaluate_probability_stacking(
    graph_spec: ArchitectureSpec,
    graphs: Sequence[Any],
    meta: pd.DataFrame,
    groups: np.ndarray,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    checkpoint: Path,
    model_seed: int,
    training: TrainingConfig,
    output_dir: Path,
) -> dict[str, Any]:
    """Fit a probability meta learner exclusively on outer-training OOF data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = labels_from_graphs(graphs)
    predictions = {
        "Flat RF": np.full(len(graphs), np.nan, dtype=float),
        "Pure graph": np.full(len(graphs), np.nan, dtype=float),
        "Flat RF + graph stacking": np.full(len(graphs), np.nan, dtype=float),
    }
    fold_rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    flat_params = {"n_estimators": 700, "max_features": "sqrt", "min_samples_leaf": 2}

    for outer_fold, (outer_train, outer_valid) in enumerate(outer_folds):
        print(f"[Stacking] outer fold {outer_fold + 1}/{len(outer_folds)}", flush=True)
        graph_inner_oof = np.full(len(outer_train), np.nan, dtype=float)
        flat_inner_oof = np.full(len(outer_train), np.nan, dtype=float)
        inner_epochs: list[int] = []
        inner_local = grouped_folds(labels[outer_train], groups[outer_train], 3, seed=43000 + outer_fold)
        for inner_fold, (inner_train_local, inner_valid_local) in enumerate(inner_local):
            inner_train = outer_train[inner_train_local]
            inner_valid = outer_train[inner_valid_local]
            model, history, _, best_epoch = train_model(
                graph_spec,
                graphs,
                inner_train,
                checkpoint,
                seed=model_seed + outer_fold * 101 + inner_fold * 10007,
                training=training,
                valid_indices=inner_valid,
            )
            graph_probability = predict_model(
                model, graphs, inner_valid, training.batch_size, resolve_device(training.device)
            )["probability"]
            flat_probability, _, _ = fit_predict_flat(
                "random_forest",
                flat_params,
                graphs,
                meta,
                inner_train,
                inner_valid,
                model_seed + outer_fold * 101 + inner_fold * 10007,
            )
            graph_inner_oof[inner_valid_local] = graph_probability
            flat_inner_oof[inner_valid_local] = flat_probability
            inner_epochs.append(best_epoch)
            for row in history:
                training_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": inner_fold,
                        "run_type": "stacking_inner_graph_oof",
                        **row,
                    }
                )
        if bool(np.isnan(graph_inner_oof).any()) or bool(np.isnan(flat_inner_oof).any()):
            raise RuntimeError("Stacking meta features are not complete outer-training OOF predictions")
        meta_features = np.column_stack([flat_inner_oof, graph_inner_oof])
        meta_learner = LogisticRegression(C=1.0, max_iter=5000, random_state=model_seed + outer_fold)
        meta_learner.fit(meta_features, labels[outer_train])
        selected_epoch = max(1, int(round(float(np.median(inner_epochs)))))
        graph_model, history, _, _ = train_model(
            graph_spec,
            graphs,
            outer_train,
            checkpoint,
            seed=model_seed + outer_fold * 101,
            training=training,
            fixed_epochs=selected_epoch,
        )
        graph_outer = predict_model(
            graph_model, graphs, outer_valid, training.batch_size, resolve_device(training.device)
        )["probability"]
        flat_outer, _, _ = fit_predict_flat(
            "random_forest",
            flat_params,
            graphs,
            meta,
            outer_train,
            outer_valid,
            model_seed + outer_fold * 101,
        )
        stacked_outer = meta_learner.predict_proba(np.column_stack([flat_outer, graph_outer]))[:, 1]
        for name, values in (
            ("Flat RF", flat_outer),
            ("Pure graph", graph_outer),
            ("Flat RF + graph stacking", stacked_outer),
        ):
            predictions[name][outer_valid] = values
            fold_rows.append(
                {
                    "model": name,
                    "seed": model_seed,
                    "outer_fold": outer_fold,
                    "selected_graph_epoch": selected_epoch,
                    **classification_metrics(labels[outer_valid], values),
                }
            )
        meta_rows.append(
            {
                "outer_fold": outer_fold,
                "meta_fit_sample_count": len(outer_train),
                "meta_fit_features": "outer-training inner-OOF Flat_RF and Pure_graph probabilities",
                "flat_coefficient": float(meta_learner.coef_[0, 0]),
                "graph_coefficient": float(meta_learner.coef_[0, 1]),
                "intercept": float(meta_learner.intercept_[0]),
                "meta_in_sample_probability_used": False,
                "outer_validation_used_to_fit_meta": False,
            }
        )
        for row in history:
            training_rows.append(
                {"outer_fold": outer_fold, "inner_fold": -1, "run_type": "stacking_outer_graph_refit", **row}
            )

    for name, probability in predictions.items():
        if bool(np.isnan(probability).any()):
            raise RuntimeError(f"Incomplete stacking OOF prediction: {name}")
    metric_rows = [{"model": name, **classification_metrics(labels, values)} for name, values in predictions.items()]
    pd.DataFrame(metric_rows).to_csv(output_dir / "stacking_results.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "stacking_fold_metrics.csv", index=False)
    pd.DataFrame(meta_rows).to_csv(output_dir / "stacking_meta_audit.csv", index=False)
    pd.DataFrame(training_rows).to_csv(output_dir / "stacking_training_audit.csv", index=False)
    frame = pd.DataFrame({"development_index": np.arange(len(graphs)), "label": labels, "scaffold_group": groups})
    for name, values in predictions.items():
        frame[name] = values
    frame.to_csv(output_dir / "stacking_oof_predictions.csv", index=False)
    result = {row["model"]: row for row in metric_rows}
    result["graph_spec"] = graph_spec.to_dict()
    (output_dir / "stacking_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
