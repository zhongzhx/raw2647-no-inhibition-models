"""Convergence-driven leakage-safe pretraining for the shared GINE encoder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "models", ROOT / "utils", ROOT / "data_processing"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gine_v2 import (
    ATOM_FEATURE_CLASS_COUNTS,
    ATOM_PREDICTION_FEATURES,
    BOND_FEATURE_CLASS_COUNTS,
    BOND_PREDICTION_FEATURES,
    GINEEncoder,
    GINEPretrainingModel,
    vicreg_loss,
)
from pipeline_utils import load_graphs, set_seed


RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = [
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
]


def resolve_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor_indices(graph: object) -> list[int]:
    names = list(getattr(graph, "global_feature_names", []))
    missing = [name for name in DESCRIPTOR_NAMES if name not in names]
    if missing:
        raise ValueError(f"Missing required pretraining descriptors: {missing}")
    return [names.index(name) for name in DESCRIPTOR_NAMES]


def descriptor_stats(
    graphs: list,
    train_indices: np.ndarray,
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit descriptor scaling on pretraining-train only."""
    values = torch.cat(
        [graphs[int(idx)].global_feat[:, indices].float() for idx in train_indices], dim=0
    )
    return values.mean(0, keepdim=True), values.std(0, unbiased=False, keepdim=True).clamp(min=1e-6)


def save_history(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)


def exclude_downstream_overlaps(
    graphs: list,
    downstream_index: str,
    output_dir: Path,
) -> tuple[list, int, str]:
    if not downstream_index:
        return graphs, 0, ""
    index_path = Path(downstream_index)
    with index_path.open(newline="", encoding="utf-8") as handle:
        downstream_keys = {
            str(row.get("inchikey", "")).strip()
            for row in csv.DictReader(handle)
            if str(row.get("inchikey", "")).strip()
        }
    kept, removed = [], []
    for graph in graphs:
        key = str(getattr(graph, "inchikey", "")).strip()
        if key in downstream_keys:
            removed.append(
                {
                    "inchikey": key,
                    "canonical_smiles": str(getattr(graph, "canonical_smiles", "")),
                    "compound_name_or_label": str(getattr(graph, "compound_name_or_label", "")),
                }
            )
        else:
            kept.append(graph)
    pd.DataFrame(removed).to_csv(output_dir / "excluded_downstream_overlaps.csv", index=False)
    return kept, len(removed), file_sha256(index_path)


def scaffold_for_graph(graph: Any) -> str:
    smiles = str(getattr(graph, "canonical_smiles", getattr(graph, "smiles", "")))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return f"<invalid>:{getattr(graph, 'inchikey', smiles)}"
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        # Unrelated acyclic compounds should not become one giant artificial group.
        return f"<acyclic>:{getattr(graph, 'inchikey', smiles)}"
    return Chem.MolToSmiles(scaffold, canonical=True)


def scaffold_split(
    graphs: list,
    validation_fraction: float,
    seed: int,
    candidates: int = 64,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    scaffolds = [scaffold_for_graph(graph) for graph in graphs]
    groups = np.asarray(scaffolds, dtype=object)
    dummy = np.zeros((len(graphs), 1), dtype=np.float32)
    splitter = GroupShuffleSplit(
        n_splits=candidates,
        test_size=validation_fraction,
        random_state=seed,
    )
    best: tuple[np.ndarray, np.ndarray] | None = None
    best_error = float("inf")
    for train_idx, valid_idx in splitter.split(dummy, groups=groups):
        error = abs(len(valid_idx) / len(graphs) - validation_fraction)
        if error < best_error:
            best = (train_idx, valid_idx)
            best_error = error
    if best is None:
        raise RuntimeError("Unable to construct scaffold split")
    train_idx, valid_idx = best
    overlap = set(groups[train_idx]) & set(groups[valid_idx])
    if overlap:
        raise RuntimeError(f"Scaffold leakage detected: {len(overlap)} groups")
    return np.asarray(train_idx), np.asarray(valid_idx), scaffolds


def write_split_audit(
    output_dir: Path,
    graphs: list,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    scaffolds: list[str],
) -> dict[str, Any]:
    split = np.full(len(graphs), "train", dtype=object)
    split[valid_idx] = "validation"
    rows = [
        {
            "row_index": idx,
            "inchikey": str(getattr(graph, "inchikey", "")),
            "canonical_smiles": str(getattr(graph, "canonical_smiles", "")),
            "bemis_murcko_scaffold": scaffolds[idx],
            "split": split[idx],
        }
        for idx, graph in enumerate(graphs)
    ]
    pd.DataFrame(rows).to_csv(output_dir / "pretraining_split_assignments.csv", index=False)
    train_scaffolds = {scaffolds[int(idx)] for idx in train_idx}
    valid_scaffolds = {scaffolds[int(idx)] for idx in valid_idx}
    return {
        "train_graph_count": int(len(train_idx)),
        "validation_graph_count": int(len(valid_idx)),
        "validation_fraction_actual": len(valid_idx) / len(graphs),
        "train_scaffold_count": len(train_scaffolds),
        "validation_scaffold_count": len(valid_scaffolds),
        "scaffold_overlap_count": len(train_scaffolds & valid_scaffolds),
    }


def class_weights_and_baselines(
    graphs: list,
    train_idx: np.ndarray,
    attribute: str,
    feature_map: dict[str, int],
    class_counts: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    counts = {
        name: torch.zeros(class_counts[feature_idx], dtype=torch.float64)
        for name, feature_idx in feature_map.items()
    }
    for idx in train_idx:
        values = getattr(graphs[int(idx)], attribute).long()
        for name, feature_idx in feature_map.items():
            count = torch.bincount(
                values[:, feature_idx].clamp(0, class_counts[feature_idx] - 1),
                minlength=class_counts[feature_idx],
            ).double()
            counts[name] += count
    weights, baselines = {}, {}
    for name, values in counts.items():
        observed = values > 0
        raw = torch.zeros_like(values)
        raw[observed] = torch.rsqrt(values[observed])
        expected = (raw * values).sum() / values.sum().clamp(min=1)
        raw = (raw / expected.clamp(min=1e-12)).clamp(max=10.0)
        weights[name] = raw.float()
        baselines[name] = float(values.max() / values.sum().clamp(min=1))
    return weights, baselines


def learning_rate_for_step(
    step: int,
    total_steps: int,
    base_lr: float,
    min_lr_ratio: float,
    warmup_fraction: float,
) -> float:
    warmup_steps = max(1, int(total_steps * warmup_fraction))
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


def gradient_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = sum(float(grad.detach().float().pow(2).sum().cpu()) for grad in gradients if grad is not None)
    return math.sqrt(total)


def objective(
    model: GINEPretrainingModel,
    batch: Any,
    descriptor_columns: list[int],
    descriptor_mean: torch.Tensor,
    descriptor_std: torch.Tensor,
    atom_weights: dict[str, torch.Tensor],
    bond_weights: dict[str, torch.Tensor],
    args: argparse.Namespace,
    use_auxiliary: bool,
) -> dict[str, Any]:
    output = model(
        batch,
        atom_mask_ratio=args.atom_mask_ratio,
        bond_mask_ratio=args.bond_mask_ratio,
        feature_dropout=args.feature_dropout,
        contrastive=use_auxiliary,
    )
    atom_losses = {
        name: F.cross_entropy(output["atom_logits"][name], targets, weight=atom_weights[name])
        for name, targets in output["atom_targets"].items()
    }
    bond_losses = {}
    for name, targets in output["bond_targets"].items():
        if targets.numel():
            bond_losses[name] = F.cross_entropy(
                output["bond_logits"][name], targets, weight=bond_weights[name]
            )
        else:
            bond_losses[name] = output["descriptor_pred"].sum() * 0.0
    atom_loss = torch.stack(list(atom_losses.values())).mean()
    bond_loss = torch.stack(list(bond_losses.values())).mean()
    descriptor_loss = output["descriptor_pred"].sum() * 0.0
    vicreg = {
        key: output["descriptor_pred"].sum() * 0.0
        for key in ("loss", "invariance", "variance", "covariance")
    }
    if use_auxiliary:
        target = (batch.global_feat[:, descriptor_columns].float() - descriptor_mean) / descriptor_std
        descriptor_loss = F.smooth_l1_loss(output["descriptor_pred"], target)
        vicreg = vicreg_loss(
            output["contrastive_z1"],
            output["contrastive_z2"],
            invariance_coefficient=args.vicreg_invariance,
            variance_coefficient=args.vicreg_variance,
            covariance_coefficient=args.vicreg_covariance,
        )
    weighted = {
        "atom": atom_loss,
        "bond": bond_loss,
        "descriptor": args.descriptor_weight * descriptor_loss,
        "vicreg": args.vicreg_weight * vicreg["loss"],
    }
    loss = sum(weighted.values())
    correct, counts = {}, {}
    for family in ("atom", "bond"):
        for name, targets in output[f"{family}_targets"].items():
            key = f"{family}_{name}"
            counts[key] = int(targets.numel())
            correct[key] = int(
                (output[f"{family}_logits"][name].argmax(dim=-1) == targets).sum().detach().cpu()
            ) if targets.numel() else 0
    return {
        "loss": loss,
        "weighted": weighted,
        "raw": {
            "atom": atom_loss,
            "bond": bond_loss,
            "descriptor": descriptor_loss,
            "vicreg": vicreg["loss"],
            "vicreg_invariance": vicreg["invariance"],
            "vicreg_variance": vicreg["variance"],
            "vicreg_covariance": vicreg["covariance"],
        },
        "correct": correct,
        "counts": counts,
        "embedding": output["graph_embedding"],
        "projection": output.get("contrastive_z1"),
    }


def run_epoch(
    model: GINEPretrainingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    descriptor_columns: list[int],
    descriptor_mean: torch.Tensor,
    descriptor_std: torch.Tensor,
    atom_weights: dict[str, torch.Tensor],
    bond_weights: dict[str, torch.Tensor],
    args: argparse.Namespace,
    use_auxiliary: bool,
    global_step: int,
    total_steps: int,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    loss_names = [
        "loss", "atom", "bond", "descriptor", "vicreg",
        "vicreg_invariance", "vicreg_variance", "vicreg_covariance",
    ]
    totals = {name: 0.0 for name in loss_names}
    correct: dict[str, int] = {}
    counts: dict[str, int] = {}
    graph_count = 0
    nonzero_gradient_batches = 0
    encoder_gradient_sum = 0.0
    task_gradients: dict[str, float] = {}
    embeddings: list[torch.Tensor] = []
    projections: list[torch.Tensor] = []
    encoder_parameters = [parameter for parameter in model.gine_encoder.parameters() if parameter.requires_grad]

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, batch in enumerate(loader, start=1):
            if args.steps_per_epoch and step > args.steps_per_epoch:
                break
            batch = batch.to(device)
            if training:
                assert optimizer is not None
                lr = learning_rate_for_step(
                    global_step,
                    total_steps,
                    args.lr,
                    args.min_lr_ratio,
                    args.warmup_fraction,
                )
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
            bundle = objective(
                model, batch, descriptor_columns, descriptor_mean, descriptor_std,
                atom_weights, bond_weights, args, use_auxiliary,
            )
            if training:
                if step == 1:
                    for name, weighted_loss in bundle["weighted"].items():
                        if weighted_loss.requires_grad and float(weighted_loss.detach().abs().cpu()) > 0:
                            task_gradients[f"gradient_{name}"] = gradient_norm(
                                weighted_loss, encoder_parameters
                            )
                        else:
                            task_gradients[f"gradient_{name}"] = 0.0
                bundle["loss"].backward()
                grad_sq = sum(
                    float(parameter.grad.detach().float().pow(2).sum().cpu())
                    for parameter in encoder_parameters
                    if parameter.grad is not None
                )
                grad_norm_value = math.sqrt(grad_sq)
                encoder_gradient_sum += grad_norm_value
                nonzero_gradient_batches += int(grad_norm_value > 0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                global_step += 1
            n = int(batch.num_graphs)
            graph_count += n
            totals["loss"] += float(bundle["loss"].detach().cpu()) * n
            for name in ("atom", "bond", "descriptor", "vicreg"):
                totals[name] += float(bundle["raw"][name].detach().cpu()) * n
            for name in ("vicreg_invariance", "vicreg_variance", "vicreg_covariance"):
                totals[name] += float(bundle["raw"][name].detach().cpu()) * n
            for name, value in bundle["correct"].items():
                correct[name] = correct.get(name, 0) + value
                counts[name] = counts.get(name, 0) + bundle["counts"][name]
            if not training and bundle["projection"] is not None:
                embeddings.append(bundle["embedding"].detach().float().cpu())
                projections.append(bundle["projection"].detach().float().cpu())
    if graph_count == 0:
        raise RuntimeError("Epoch processed zero molecular graphs")
    metrics = {name: value / graph_count for name, value in totals.items()}
    metrics.update({f"accuracy_{name}": correct[name] / max(counts[name], 1) for name in correct})
    metrics["graphs"] = float(graph_count)
    if training:
        metrics["encoder_gradient_norm_mean"] = encoder_gradient_sum / max(nonzero_gradient_batches, 1)
        metrics["nonzero_gradient_batches"] = float(nonzero_gradient_batches)
        metrics.update(task_gradients)
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"] if optimizer else 0.0
    elif projections:
        for prefix, values in (("encoder", embeddings), ("projection", projections)):
            z = torch.cat(values, dim=0)
            centered = z - z.mean(dim=0)
            metrics[f"{prefix}_feature_std_mean"] = float(
                centered.std(dim=0, unbiased=False).mean()
            )
            covariance = centered.T @ centered / max(z.size(0) - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp(min=0)
            probabilities = eigenvalues / eigenvalues.sum().clamp(min=1e-12)
            metrics[f"{prefix}_effective_rank"] = float(
                torch.exp(-(probabilities * probabilities.clamp(min=1e-12).log()).sum())
            )
    return metrics, global_step


def save_curves(history: list[dict[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(history)
    if frame.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(frame.epoch, frame.train_loss, label="train")
    axes[0, 0].plot(frame.epoch, frame.validation_loss, label="validation")
    axes[0, 0].set(title="Composite objective", xlabel="Epoch", ylabel="Loss")
    for column in ("validation_atom", "validation_bond", "validation_descriptor", "validation_vicreg"):
        axes[0, 1].plot(frame.epoch, frame[column], label=column.replace("validation_", ""))
    axes[0, 1].set(title="Validation task losses", xlabel="Epoch", ylabel="Raw loss")
    accuracy_columns = [column for column in frame if column.startswith("validation_accuracy_")]
    for column in accuracy_columns:
        axes[1, 0].plot(frame.epoch, frame[column], label=column.replace("validation_accuracy_", ""))
    axes[1, 0].set(title="Masked feature validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1, 1].plot(frame.epoch, frame.validation_encoder_effective_rank, label="encoder rank")
    axes[1, 1].plot(frame.epoch, frame.validation_projection_effective_rank, label="projection rank")
    axes[1, 1].set(title="Representation collapse audit", xlabel="Epoch")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pretraining_convergence.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="processed_pretrain_graphs_updated_pretrain_data_run/pretrain_graphs.pt")
    parser.add_argument("--downstream_index", default="processed_smiles_graphs_multitask_updated_20260712/processed_index.csv")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument(
        "--lr_schedule_total_epochs",
        type=int,
        default=None,
        help=(
            "Epoch horizon used only for warmup/cosine LR calculation. Defaults to "
            "max_epochs. When extending a completed run with --resume, keep this at "
            "the original horizon so LR remains at the original minimum instead of "
            "jumping upward."
        ),
    )
    parser.add_argument("--min_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--relative_min_delta", type=float, default=0.002)
    parser.add_argument("--mask_only_warmup_epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.05)
    parser.add_argument("--warmup_fraction", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--atom_mask_ratio", type=float, default=0.15)
    parser.add_argument("--bond_mask_ratio", type=float, default=0.15)
    parser.add_argument("--feature_dropout", type=float, default=0.10)
    parser.add_argument("--descriptor_weight", type=float, default=0.10)
    parser.add_argument("--vicreg_weight", type=float, default=0.10)
    parser.add_argument("--vicreg_invariance", type=float, default=25.0)
    parser.add_argument("--vicreg_variance", type=float, default=25.0)
    parser.add_argument("--vicreg_covariance", type=float, default=5.0)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--max_graphs", type=int, default=0)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_epochs > args.max_epochs:
        raise ValueError("min_epochs cannot exceed max_epochs")
    if args.lr_schedule_total_epochs is not None and args.lr_schedule_total_epochs <= 0:
        raise ValueError("lr_schedule_total_epochs must be positive")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = Path(args.data)
    started = time.time()
    graphs = load_graphs(data_path)
    original_graph_count = len(graphs)
    graphs, excluded_overlap_count, downstream_sha256 = exclude_downstream_overlaps(
        graphs, args.downstream_index, output_dir
    )
    if args.max_graphs and args.max_graphs < len(graphs):
        rng = np.random.default_rng(args.seed)
        selected = np.sort(rng.choice(len(graphs), size=args.max_graphs, replace=False))
        graphs = [graphs[int(idx)] for idx in selected]
    train_idx, valid_idx, scaffolds = scaffold_split(
        graphs, args.validation_fraction, args.seed
    )
    split_audit = write_split_audit(output_dir, graphs, train_idx, valid_idx, scaffolds)
    descriptor_columns = descriptor_indices(graphs[0])
    descriptor_mean, descriptor_std = descriptor_stats(graphs, train_idx, descriptor_columns)
    atom_weights, atom_baselines = class_weights_and_baselines(
        graphs, train_idx, "x_cat", ATOM_PREDICTION_FEATURES, ATOM_FEATURE_CLASS_COUNTS
    )
    bond_weights, bond_baselines = class_weights_and_baselines(
        graphs, train_idx, "edge_attr", BOND_PREDICTION_FEATURES, BOND_FEATURE_CLASS_COUNTS
    )
    device = resolve_device(args.device)
    descriptor_mean = descriptor_mean.to(device)
    descriptor_std = descriptor_std.to(device)
    atom_weights = {name: value.to(device) for name, value in atom_weights.items()}
    bond_weights = {name: value.to(device) for name, value in bond_weights.items()}
    encoder = GINEEncoder(args.hidden_dim, args.num_layers, args.dropout, use_layer_norm=True)
    model = GINEPretrainingModel(
        encoder, descriptor_dim=len(descriptor_columns), projection_dim=args.projection_dim
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        [graphs[int(idx)] for idx in train_idx],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        [graphs[int(idx)] for idx in valid_idx],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    updates_per_epoch = args.steps_per_epoch or len(train_loader)
    lr_schedule_total_epochs = args.lr_schedule_total_epochs or args.max_epochs
    total_steps = lr_schedule_total_epochs * updates_per_epoch
    config = vars(args).copy()
    config.update(
        {
            "data_sha256": file_sha256(data_path),
            "downstream_index_sha256": downstream_sha256,
            "original_graph_count": original_graph_count,
            "post_overlap_graph_count": len(graphs),
            "excluded_downstream_overlap_count": excluded_overlap_count,
            "descriptor_names": DESCRIPTOR_NAMES,
            "lr_schedule_total_epochs_effective": lr_schedule_total_epochs,
            "vicreg_space": "transferable_encoder_graph_embedding",
            "descriptor_mean_train_only": descriptor_mean.detach().cpu().view(-1).tolist(),
            "descriptor_std_train_only": descriptor_std.detach().cpu().view(-1).tolist(),
            "atom_majority_baselines": atom_baselines,
            "bond_majority_baselines": bond_baselines,
            **split_audit,
        }
    )
    (output_dir / "pretraining_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    initial_encoder = {
        key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()
    }
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    plateau_reference = float("inf")
    best_epoch = 0
    no_significant_improvement = 0
    global_step = 0
    start_epoch = 1
    converged = False
    last_state_path = output_dir / "last_training_state.pt"
    best_checkpoint_path = output_dir / "gine_encoder_best.pt"
    if args.resume:
        if not last_state_path.exists():
            raise FileNotFoundError(f"Resume checkpoint missing: {last_state_path}")
        state = torch.load(last_state_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = state["history"]
        best_val_loss = state["best_val_loss"]
        plateau_reference = state["plateau_reference"]
        best_epoch = state["best_epoch"]
        no_significant_improvement = state["no_significant_improvement"]
        global_step = state["global_step"]
        start_epoch = state["epoch"] + 1

    for epoch in range(start_epoch, args.max_epochs + 1):
        use_auxiliary = epoch > args.mask_only_warmup_epochs
        train_metrics, global_step = run_epoch(
            model, train_loader, optimizer, device, descriptor_columns,
            descriptor_mean, descriptor_std, atom_weights, bond_weights,
            args, use_auxiliary, global_step, total_steps,
        )
        validation_metrics, _ = run_epoch(
            model, valid_loader, None, device, descriptor_columns,
            descriptor_mean, descriptor_std, atom_weights, bond_weights,
            args, True, global_step, total_steps,
        )
        row: dict[str, Any] = {
            "epoch": epoch,
            "phase": "joint" if use_auxiliary else "mask_warmup",
            "elapsed_seconds": time.time() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        save_history(output_dir / "pretraining_history.csv", history)
        save_curves(history, output_dir)
        validation_loss = validation_metrics["loss"]
        eligible = epoch > args.mask_only_warmup_epochs
        if eligible and validation_loss < best_val_loss:
            best_val_loss = validation_loss
            best_epoch = epoch
            torch.save(
                {
                    "encoder_state_dict": {"gine_encoder": encoder.state_dict()},
                    "architecture": {
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                        "use_layer_norm": True,
                        "pooling": "mean",
                    },
                    "objective": "full_atom_bond_masking+descriptor+VICReg",
                    "best_epoch": epoch,
                    "best_validation_loss": best_val_loss,
                    "pretraining_train_count": len(train_idx),
                    "pretraining_validation_count": len(valid_idx),
                    "excluded_downstream_overlap_count": excluded_overlap_count,
                    "downstream_index_sha256": downstream_sha256,
                    "descriptor_names": DESCRIPTOR_NAMES,
                    "descriptor_mean_train_only": descriptor_mean.detach().cpu(),
                    "descriptor_std_train_only": descriptor_std.detach().cpu(),
                    "seed": args.seed,
                },
                best_checkpoint_path,
            )
        significant_threshold = (
            plateau_reference * (1.0 - args.relative_min_delta)
            if math.isfinite(plateau_reference) else float("inf")
        )
        if eligible and validation_loss < significant_threshold:
            plateau_reference = validation_loss
            no_significant_improvement = 0
        elif epoch >= args.min_epochs:
            no_significant_improvement += 1
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
                "plateau_reference": plateau_reference,
                "best_epoch": best_epoch,
                "no_significant_improvement": no_significant_improvement,
                "global_step": global_step,
            },
            last_state_path,
        )
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if epoch >= args.min_epochs and no_significant_improvement >= args.patience:
            converged = True
            break

    final_encoder = encoder.state_dict()
    changed_numel = sum(
        int((final_encoder[key].detach().cpu() != initial_encoder[key]).sum())
        for key in initial_encoder
    )
    last = history[-1]
    summary = {
        "status": "converged_by_validation_plateau" if converged else "max_epochs_reached_without_plateau",
        "converged": converged,
        "epochs_run": int(last["epoch"]),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "no_significant_improvement_epochs": no_significant_improvement,
        "relative_min_delta": args.relative_min_delta,
        "patience": args.patience,
        "device": str(device),
        "checkpoint": str(best_checkpoint_path),
        "optimizer_contains_all_encoder_parameters": all(
            any(parameter is candidate for group in optimizer.param_groups for candidate in group["params"])
            for parameter in encoder.parameters()
        ),
        "encoder_changed_parameter_values": changed_numel,
        "excluded_downstream_overlap_count": excluded_overlap_count,
        "graph_count": len(graphs),
        **split_audit,
        "last_validation_projection_feature_std_mean": last["validation_projection_feature_std_mean"],
        "last_validation_projection_effective_rank": last["validation_projection_effective_rank"],
        "last_validation_encoder_feature_std_mean": last["validation_encoder_feature_std_mean"],
        "last_validation_encoder_effective_rank": last["validation_encoder_effective_rank"],
        "elapsed_seconds": time.time() - started,
    }
    summary["collapse_free"] = bool(
        summary["last_validation_projection_feature_std_mean"] >= 0.05
        and summary["last_validation_projection_effective_rank"] >= 8.0
        and summary["last_validation_encoder_feature_std_mean"] >= 0.05
        and summary["last_validation_encoder_effective_rank"] >= 8.0
    )
    summary["approved_checkpoint"] = bool(summary["converged"] and summary["collapse_free"])
    (output_dir / "pretraining_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = f"""# PRETRAINING CONVERGENCE REPORT

## Verdict

**{summary['status']}**

- Epochs run: {summary['epochs_run']}
- Best epoch: {best_epoch}
- Best scaffold-validation composite loss: {best_val_loss:.6f}
- Plateau definition: less than {args.relative_min_delta:.2%} relative improvement for {args.patience} consecutive epochs after epoch {args.min_epochs}
- Molecules after downstream-overlap removal: {len(graphs):,}
- Train/validation: {len(train_idx):,} / {len(valid_idx):,}
- Scaffold overlap: {split_audit['scaffold_overlap_count']}
- Exact downstream structures excluded: {excluded_overlap_count}
- Encoder parameter values changed: {changed_numel:,}
- Projection feature std at final epoch: {summary['last_validation_projection_feature_std_mean']:.4f}
- Projection effective rank at final epoch: {summary['last_validation_projection_effective_rank']:.2f}
- Encoder feature std at final epoch: {summary['last_validation_encoder_feature_std_mean']:.4f}
- Encoder effective rank at final epoch: {summary['last_validation_encoder_effective_rank']:.2f}
- Collapse audit passed: {summary['collapse_free']}
- Checkpoint approved: {summary['approved_checkpoint']}

The checkpoint is approved for downstream transfer only when `converged` is true and the collapse audit remains non-degenerate.
"""
    (output_dir / "PRETRAINING_CONVERGENCE_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
