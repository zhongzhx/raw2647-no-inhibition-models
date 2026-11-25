"""Shared GINE v2 components for pretraining and RAW264.7 downstream models.

This module intentionally contains the only GINE encoder used by the redesign.
Pretraining and downstream code instantiate :class:`GINEEncoder` directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

from smiles_to_graph import (
    ATOM_TYPE_TO_ID,
    BOND_DIRECTION_TO_ID,
    BOND_STEREO_TO_ID,
    BOND_TYPE_TO_ID,
    BOOL_TO_ID,
    CHIRALITY_TO_ID,
    DEGREE_TO_ID,
    FORMAL_CHARGE_TO_ID,
    HYBRIDIZATION_TO_ID,
    RADICAL_ELECTRONS_TO_ID,
    TOTAL_NUM_HS_TO_ID,
)


ATOM_TYPE_CLASS_COUNT = len(ATOM_TYPE_TO_ID) + 1
ATOM_TYPE_MASK_ID = ATOM_TYPE_CLASS_COUNT
BOND_TYPE_CLASS_COUNT = len(BOND_TYPE_TO_ID) + 1

# Each class-count includes the unknown category used by processed downstream
# data.  The mask token is a separate final embedding and can therefore never
# collide with a real/unknown chemistry feature value.
ATOM_FEATURE_CLASS_COUNTS = (
    ATOM_TYPE_CLASS_COUNT,
    len(CHIRALITY_TO_ID) + 1,
    len(FORMAL_CHARGE_TO_ID) + 1,
    len(DEGREE_TO_ID) + 1,
    len(TOTAL_NUM_HS_TO_ID) + 1,
    len(HYBRIDIZATION_TO_ID) + 1,
    2,
    2,
    len(RADICAL_ELECTRONS_TO_ID) + 1,
)
ATOM_FEATURE_MASK_IDS = ATOM_FEATURE_CLASS_COUNTS
ATOM_PREDICTION_FEATURES = {
    "atom_type": 0,
    "chirality": 1,
    "formal_charge": 2,
}

BOND_FEATURE_CLASS_COUNTS = (
    BOND_TYPE_CLASS_COUNT,
    len(BOND_DIRECTION_TO_ID) + 1,
    len(BOND_STEREO_TO_ID) + 1,
    len(BOOL_TO_ID),
    len(BOOL_TO_ID),
)
BOND_FEATURE_MASK_IDS = BOND_FEATURE_CLASS_COUNTS
BOND_PREDICTION_FEATURES = {
    "bond_type": 0,
    "stereo": 2,
    "conjugated": 3,
    "in_ring": 4,
}


class GINEEncoder(nn.Module):
    """Residual GINE encoder shared verbatim by pretraining and downstream tasks."""

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("GINEEncoder requires at least one message-passing layer")
        embedding_dim = max(4, hidden_dim // 4)
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(class_count + 1, embedding_dim) for class_count in ATOM_FEATURE_CLASS_COUNTS]
        )
        self.node_encoder = nn.Sequential(
            nn.Linear(embedding_dim * len(ATOM_FEATURE_CLASS_COUNTS) + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.bond_embeddings = nn.ModuleList(
            [nn.Embedding(class_count + 1, hidden_dim) for class_count in BOND_FEATURE_CLASS_COUNTS]
        )
        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, hidden_dim),
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity() for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm

    def encode_nodes(
        self,
        data: Any,
        x_cat_override: torch.Tensor | None = None,
        x_cont_override: torch.Tensor | None = None,
        feature_dropout: float = 0.0,
    ) -> torch.Tensor:
        x_cat = data.x_cat.long() if x_cat_override is None else x_cat_override.long()
        x_cont = data.x_cont.float() if x_cont_override is None else x_cont_override.float()
        parts = []
        for feature_idx, embedding in enumerate(self.atom_embeddings):
            values = x_cat[:, feature_idx].clamp(0, embedding.num_embeddings - 1)
            parts.append(embedding(values))
        node_features = torch.cat(parts + [x_cont], dim=-1)
        encoded = self.node_encoder(node_features)
        if feature_dropout > 0:
            encoded = F.dropout(encoded, p=feature_dropout, training=self.training)
        return encoded

    def encode_edges(
        self,
        data: Any,
        edge_attr_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        edge_attr = data.edge_attr.long() if edge_attr_override is None else edge_attr_override.long()
        parts = []
        for feature_idx, embedding in enumerate(self.bond_embeddings):
            raw_values = edge_attr[:, feature_idx]
            values = raw_values.clamp(0, embedding.num_embeddings - 1)
            parts.append(embedding(values))
        return torch.stack(parts, dim=0).sum(dim=0)

    def forward_nodes(
        self,
        data: Any,
        x_cat_override: torch.Tensor | None = None,
        x_cont_override: torch.Tensor | None = None,
        edge_attr_override: torch.Tensor | None = None,
        feature_dropout: float = 0.0,
    ) -> torch.Tensor:
        return self.forward_layer_outputs(
            data,
            x_cat_override=x_cat_override,
            x_cont_override=x_cont_override,
            edge_attr_override=edge_attr_override,
            feature_dropout=feature_dropout,
        )[-1]

    def forward_layer_outputs(
        self,
        data: Any,
        x_cat_override: torch.Tensor | None = None,
        x_cont_override: torch.Tensor | None = None,
        edge_attr_override: torch.Tensor | None = None,
        feature_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, ...]:
        """Return every message-passing layer without changing checkpoint state.

        The method introduces no parameters or buffers, so existing checkpoint
        keys and shapes remain byte-for-byte compatible.  ``forward_nodes``
        continues to return only the final layer for all existing callers.
        """
        x = self.encode_nodes(data, x_cat_override, x_cont_override, feature_dropout)
        edge_features = self.encode_edges(data, edge_attr_override)
        outputs: list[torch.Tensor] = []
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, data.edge_index, edge_features)
            x = norm(F.relu(x) + residual)
            x = self.dropout(x)
            outputs.append(x)
        return tuple(outputs)

    def pool(self, node_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        return global_mean_pool(node_embeddings, batch)

    def forward(self, data: Any, **kwargs: Any) -> torch.Tensor:
        return self.pool(self.forward_nodes(data, **kwargs), data.batch)


@dataclass
class TransferAudit:
    checkpoint_path: str
    loaded_parameter_numel: int
    target_parameter_numel: int
    load_ratio: float
    missing_keys: list[str]
    unexpected_keys: list[str]
    shape_mismatches: list[str]
    exact_value_match_after_load: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_gine_checkpoint(
    encoder: GINEEncoder,
    checkpoint_path: str,
    min_load_ratio: float = 0.95,
) -> TransferAudit:
    """Load a legacy or v2 encoder and fail when fewer than 95% of target values load."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("encoder_state_dict", checkpoint)
    if isinstance(state, dict) and "structure_encoder" in state:
        state = state["structure_encoder"]
    if isinstance(state, dict) and "gine_encoder" in state:
        state = state["gine_encoder"]
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported encoder checkpoint payload: {checkpoint_path}")

    target = encoder.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    unexpected: list[str] = []
    shape_mismatches: list[str] = []
    loaded_numel = 0
    for key, value in state.items():
        if key not in target:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(target[key].shape):
            shape_mismatches.append(
                f"{key}: checkpoint={tuple(value.shape)} target={tuple(target[key].shape)}"
            )
            continue
        compatible[key] = value
        loaded_numel += int(target[key].numel())
    missing = sorted(set(target) - set(compatible))
    target_numel = sum(int(value.numel()) for value in target.values())
    ratio = loaded_numel / max(target_numel, 1)
    if ratio < min_load_ratio:
        raise RuntimeError(
            f"GINE checkpoint load ratio {ratio:.2%} is below required {min_load_ratio:.0%}; "
            f"missing={len(missing)}, unexpected={len(unexpected)}, shape_mismatch={len(shape_mismatches)}"
        )
    encoder.load_state_dict(compatible, strict=False)
    loaded_state = encoder.state_dict()
    exact_match = all(torch.equal(loaded_state[key].cpu(), value.cpu()) for key, value in compatible.items())
    if not exact_match:
        raise RuntimeError("Loaded GINE parameters changed immediately after load")
    return TransferAudit(
        checkpoint_path=str(checkpoint_path),
        loaded_parameter_numel=loaded_numel,
        target_parameter_numel=target_numel,
        load_ratio=ratio,
        missing_keys=missing,
        unexpected_keys=sorted(unexpected),
        shape_mismatches=shape_mismatches,
        exact_value_match_after_load=exact_match,
    )


class ProjectedBranch(nn.Module):
    def __init__(self, input_dim: int, branch_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, branch_dim),
            nn.LayerNorm(branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values.float())


class IndependentHybridEncoder(nn.Module):
    """Balanced graph, descriptor, fingerprint, and optional family branches."""

    BRANCHES = ("gine", "rdkit", "morgan", "maccs", "family")

    def __init__(
        self,
        gine_encoder: GINEEncoder,
        global_dim: int,
        morgan_dim: int,
        maccs_dim: int,
        family_dim: int = 0,
        branch_dim: int = 64,
        dropout: float = 0.2,
        enabled_branches: tuple[str, ...] = ("gine", "rdkit", "morgan", "maccs"),
        zeroed_branches: tuple[str, ...] = (),
        fusion: str = "concat",
    ) -> None:
        super().__init__()
        unknown = set(enabled_branches) - set(self.BRANCHES)
        if unknown:
            raise ValueError(f"Unknown hybrid branches: {sorted(unknown)}")
        if not enabled_branches:
            raise ValueError("At least one hybrid branch must be enabled")
        invalid_zeroed = set(zeroed_branches) - set(enabled_branches)
        if invalid_zeroed:
            raise ValueError(
                f"Zeroed branches must also be enabled: {sorted(invalid_zeroed)}"
            )
        if "family" in enabled_branches and family_dim <= 0:
            raise ValueError("family_dim must be positive when the family branch is enabled")
        self.gine_encoder = gine_encoder
        self.enabled_branches = tuple(enabled_branches)
        self.zeroed_branches = frozenset(zeroed_branches)
        self.fusion = fusion
        self.gine_projection = ProjectedBranch(gine_encoder.hidden_dim, branch_dim, dropout)
        self.rdkit_projection = ProjectedBranch(global_dim, branch_dim, dropout)
        self.morgan_projection = ProjectedBranch(morgan_dim, branch_dim, dropout)
        self.maccs_projection = ProjectedBranch(maccs_dim, branch_dim, dropout)
        self.family_projection = (
            ProjectedBranch(family_dim, branch_dim, dropout)
            if "family" in self.enabled_branches
            else None
        )
        if fusion == "gated":
            self.gates = nn.Parameter(torch.zeros(len(self.enabled_branches)))
            self.output_dim = branch_dim
        elif fusion == "concat":
            self.gates = None
            self.output_dim = branch_dim * len(self.enabled_branches)
        else:
            raise ValueError(f"Unsupported fusion: {fusion}")

    def branch_embeddings(self, data: Any) -> dict[str, torch.Tensor]:
        embeddings = {
            "gine": self.gine_projection(self.gine_encoder(data)),
            "rdkit": self.rdkit_projection(data.global_feat),
            "morgan": self.morgan_projection(data.morgan_fp),
            "maccs": self.maccs_projection(data.maccs_fp),
        }
        if self.family_projection is not None:
            embeddings["family"] = self.family_projection(data.scaffold_family_fp)
        return embeddings

    def _branch(self, name: str, data: Any) -> torch.Tensor:
        if name == "gine":
            return self.gine_projection(self.gine_encoder(data))
        if name == "rdkit":
            return self.rdkit_projection(data.global_feat)
        if name == "morgan":
            return self.morgan_projection(data.morgan_fp)
        if name == "maccs":
            return self.maccs_projection(data.maccs_fp)
        if name == "family" and self.family_projection is not None:
            return self.family_projection(data.scaffold_family_fp)
        raise ValueError(name)

    def forward(self, data: Any) -> torch.Tensor:
        selected = []
        for name in self.enabled_branches:
            embedding = self._branch(name, data)
            # Multiplication preserves the complete parameterized branch while
            # making its information contribution exactly zero.  This is the
            # parameter-matched A4–A7 ablation required by the locked protocol.
            if name in self.zeroed_branches:
                embedding = embedding * 0.0
            selected.append(embedding)
        if self.fusion == "gated":
            weights = torch.softmax(self.gates, dim=0)
            return sum(weight * embedding for weight, embedding in zip(weights, selected))
        return torch.cat(selected, dim=-1)


class MolecularClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, output_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data: Any) -> torch.Tensor:
        return self.head(self.encoder(data)).view(-1)


class GraphOnlyEncoder(nn.Module):
    def __init__(self, gine_encoder: GINEEncoder) -> None:
        super().__init__()
        self.gine_encoder = gine_encoder
        self.output_dim = gine_encoder.hidden_dim

    def forward(self, data: Any) -> torch.Tensor:
        return self.gine_encoder(data)


def symmetric_bond_mask(edge_index: torch.Tensor, ratio: float) -> torch.Tensor:
    """Mask both directions of a chemical bond together to prevent reverse-edge leakage."""
    edge_count = edge_index.size(1)
    if edge_count == 0 or ratio <= 0:
        return torch.zeros(edge_count, dtype=torch.bool, device=edge_index.device)
    lo = torch.minimum(edge_index[0], edge_index[1])
    hi = torch.maximum(edge_index[0], edge_index[1])
    base = int(edge_index.max().item()) + 1
    pair_ids = lo * base + hi
    unique_pairs, inverse = torch.unique(pair_ids, return_inverse=True)
    selected = torch.rand(unique_pairs.size(0), device=edge_index.device) < ratio
    if unique_pairs.numel() and not bool(selected.any()):
        selected[torch.randint(unique_pairs.size(0), (1,), device=edge_index.device)] = True
    return selected[inverse]


class GINEPretrainingModel(nn.Module):
    """Leakage-safe masked atom/bond, descriptor, and VICReg objectives."""

    def __init__(
        self,
        gine_encoder: GINEEncoder,
        descriptor_dim: int = 9,
        projection_dim: int = 128,
    ) -> None:
        super().__init__()
        self.gine_encoder = gine_encoder
        hidden_dim = gine_encoder.hidden_dim
        self.atom_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, ATOM_FEATURE_CLASS_COUNTS[feature_idx])
                for name, feature_idx in ATOM_PREDICTION_FEATURES.items()
            }
        )
        self.bond_backbone = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        self.bond_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, BOND_FEATURE_CLASS_COUNTS[feature_idx])
                for name, feature_idx in BOND_PREDICTION_FEATURES.items()
            }
        )
        self.descriptor_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, descriptor_dim),
        )
        # VICReg is applied directly to the graph embedding that downstream
        # consumes.  A disposable projection head could look healthy while
        # leaving the transferable encoder collapsed.
        self.projection_dim = hidden_dim

    @staticmethod
    def mask_atoms(
        data: Any,
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Mask the complete atom row so mass/charge cannot reveal atom type."""
        x_cat = data.x_cat.long().clone()
        x_cont = data.x_cont.float().clone()
        targets = {
            name: data.x_cat[:, feature_idx].long().clamp(
                min=0, max=ATOM_FEATURE_CLASS_COUNTS[feature_idx] - 1
            )
            for name, feature_idx in ATOM_PREDICTION_FEATURES.items()
        }
        mask = torch.rand(x_cat.size(0), device=x_cat.device) < ratio
        if x_cat.size(0) and ratio > 0 and not bool(mask.any()):
            mask[torch.randint(x_cat.size(0), (1,), device=x_cat.device)] = True
        for feature_idx, mask_id in enumerate(ATOM_FEATURE_MASK_IDS):
            x_cat[mask, feature_idx] = mask_id
        x_cont[mask] = 0.0
        return x_cat, x_cont, mask, targets

    @staticmethod
    def mask_bonds(
        data: Any,
        ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        edge_attr = data.edge_attr.long().clone()
        targets = {
            name: data.edge_attr[:, feature_idx].long().clamp(
                min=0, max=BOND_FEATURE_CLASS_COUNTS[feature_idx] - 1
            )
            for name, feature_idx in BOND_PREDICTION_FEATURES.items()
        }
        mask = symmetric_bond_mask(data.edge_index, ratio)
        for feature_idx, mask_id in enumerate(BOND_FEATURE_MASK_IDS):
            edge_attr[mask, feature_idx] = mask_id
        return edge_attr, mask, targets

    def encode_augmented(
        self,
        data: Any,
        atom_mask_ratio: float,
        bond_mask_ratio: float,
        feature_dropout: float,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        x_cat, x_cont, atom_mask, atom_targets = self.mask_atoms(data, atom_mask_ratio)
        edge_attr, bond_mask, bond_targets = self.mask_bonds(data, bond_mask_ratio)
        nodes = self.gine_encoder.forward_nodes(
            data,
            x_cat_override=x_cat,
            x_cont_override=x_cont,
            edge_attr_override=edge_attr,
            feature_dropout=feature_dropout,
        )
        graph = self.gine_encoder.pool(nodes, data.batch)
        return graph, nodes, atom_mask, atom_targets, bond_mask, bond_targets

    def forward(
        self,
        data: Any,
        atom_mask_ratio: float = 0.15,
        bond_mask_ratio: float = 0.15,
        feature_dropout: float = 0.1,
        contrastive: bool = False,
    ) -> dict[str, torch.Tensor]:
        graph, nodes, atom_mask, atom_targets, bond_mask, bond_targets = self.encode_augmented(
            data, atom_mask_ratio, bond_mask_ratio, feature_dropout
        )
        src, dst = data.edge_index
        # Process each undirected chemical bond once; both directions remain
        # masked inside message passing, so the reverse edge cannot leak.
        representative_bond_mask = bond_mask & (src < dst)
        bond_pair_features = torch.cat(
            [nodes[src[representative_bond_mask]], nodes[dst[representative_bond_mask]]], dim=-1
        )
        bond_hidden = self.bond_backbone(bond_pair_features)
        output = {
            "graph_embedding": graph,
            "atom_logits": {
                name: head(nodes[atom_mask]) for name, head in self.atom_heads.items()
            },
            "atom_targets": {
                name: target[atom_mask] for name, target in atom_targets.items()
            },
            "atom_mask": atom_mask,
            "bond_logits": {
                name: head(bond_hidden) for name, head in self.bond_heads.items()
            },
            "bond_targets": {
                name: target[representative_bond_mask] for name, target in bond_targets.items()
            },
            "bond_mask": bond_mask,
            "representative_bond_mask": representative_bond_mask,
            "descriptor_pred": self.descriptor_head(graph),
        }
        if contrastive:
            graph_2, *_ = self.encode_augmented(
                data, atom_mask_ratio, bond_mask_ratio, feature_dropout
            )
            output["contrastive_z1"] = graph
            output["contrastive_z2"] = graph_2
        return output


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    if z1.size(0) <= 1:
        return (z1 - z2).pow(2).mean()
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size = matrix.size(0)
    if size <= 1:
        return matrix.new_zeros((0,))
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    invariance_coefficient: float = 25.0,
    variance_coefficient: float = 25.0,
    covariance_coefficient: float = 1.0,
    variance_target: float = 1.0,
) -> dict[str, torch.Tensor]:
    """VICReg without negative pairs, suitable for related natural products."""
    invariance = F.mse_loss(z1, z2)
    if z1.size(0) <= 1:
        zero = invariance * 0.0
        return {"loss": invariance, "invariance": invariance, "variance": zero, "covariance": zero}
    z1_centered = z1 - z1.mean(dim=0)
    z2_centered = z2 - z2.mean(dim=0)
    std1 = torch.sqrt(z1_centered.var(dim=0, unbiased=False) + 1e-4)
    std2 = torch.sqrt(z2_centered.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (
        F.relu(variance_target - std1).mean() + F.relu(variance_target - std2).mean()
    )
    denominator = max(z1.size(0) - 1, 1)
    cov1 = (z1_centered.T @ z1_centered) / denominator
    cov2 = (z2_centered.T @ z2_centered) / denominator
    dimension = max(z1.size(1), 1)
    covariance = (_off_diagonal(cov1).pow(2).sum() + _off_diagonal(cov2).pow(2).sum()) / dimension
    total = (
        invariance_coefficient * invariance
        + variance_coefficient * variance
        + covariance_coefficient * covariance
    )
    return {
        "loss": total,
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }
