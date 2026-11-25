"""PharmaFusion-GPS-Adapter development-only architecture.

The pretrained :class:`GINEEncoder` remains the checkpoint-compatible backbone.
All adapters, multiscale readout, GraphGPS-style blocks, and descriptor fusion
live outside that backbone so legacy state-dict names and tensor shapes do not
change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax, to_dense_batch

from gine_v2 import GINEEncoder


def random_walk_structural_encoding(
    edge_index: torch.Tensor,
    num_nodes: int,
    steps: int = 16,
) -> torch.Tensor:
    """Compute diagonal random-walk probabilities for powers 1..``steps``."""
    if num_nodes < 1:
        return torch.empty((0, steps), dtype=torch.float32)
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    if edge_index.numel():
        src = edge_index[0].detach().cpu().long()
        dst = edge_index[1].detach().cpu().long()
        adjacency[src, dst] = 1.0
        adjacency[dst, src] = 1.0
    degree = adjacency.sum(dim=1)
    isolated = degree.eq(0)
    if bool(isolated.any()):
        idx = torch.nonzero(isolated, as_tuple=False).flatten()
        adjacency[idx, idx] = 1.0
        degree = adjacency.sum(dim=1)
    transition = adjacency / degree.clamp(min=1.0).unsqueeze(1)
    power = transition.clone()
    values: list[torch.Tensor] = []
    for _ in range(steps):
        values.append(torch.diagonal(power))
        power = power @ transition
    return torch.stack(values, dim=1).float()


def clipped_shortest_path_distances(
    edge_index: torch.Tensor,
    num_nodes: int,
    clip: int = 10,
) -> torch.Tensor:
    """Return an integer NxN distance matrix; disconnected pairs use clip+1."""
    if num_nodes < 1:
        return torch.empty((0, 0), dtype=torch.long)
    disconnected = clip + 1
    distance = torch.full((num_nodes, num_nodes), disconnected, dtype=torch.long)
    distance.fill_diagonal_(0)
    if edge_index.numel():
        src = edge_index[0].detach().cpu().long()
        dst = edge_index[1].detach().cpu().long()
        distance[src, dst] = 1
        distance[dst, src] = 1
    # Molecular graphs are small; the capped Floyd-Warshall update is exact and
    # deterministic while avoiding an additional preprocessing dependency.
    for pivot in range(num_nodes):
        via = distance[:, pivot : pivot + 1] + distance[pivot : pivot + 1, :]
        distance = torch.minimum(distance, via)
    return distance.clamp(max=disconnected)


def attach_2d_structural_encodings(data: Any, rwse_steps: int = 16, spd_clip: int = 10) -> Any:
    """Attach batch-safe RWSE and flattened shortest-path values to one graph."""
    if hasattr(data, "x_cat"):
        nodes = int(data.x_cat.size(0))
    elif data.num_nodes is not None:
        nodes = int(data.num_nodes)
    else:
        raise ValueError("Cannot determine graph node count for structural encoding")
    # Explicit storage is required for reliable PyG batching of no-bond and
    # single-atom molecules; feature names such as ``x_cat`` are not always
    # recognized by PyG's automatic node-count inference.
    data.num_nodes = nodes
    data.rwse = random_walk_structural_encoding(data.edge_index, nodes, rwse_steps)
    data.spd_flat = clipped_shortest_path_distances(data.edge_index, nodes, spd_clip).reshape(-1)
    data.rwse_steps = torch.tensor([rwse_steps], dtype=torch.long)
    data.spd_clip = torch.tensor([spd_clip], dtype=torch.long)
    return data


class ResidualAdapter(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        adapter_dim: int = 16,
        dropout: float = 0.18,
        initial_scale: float = 0.1,
        learnable_scale: bool = True,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, adapter_dim)
        self.up = nn.Linear(adapter_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        scale = torch.tensor(float(initial_scale), dtype=torch.float32)
        if learnable_scale:
            self.scale = nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        # A small residual branch at initialization preserves the pretrained path.
        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = self.up(self.dropout(F.gelu(self.down(self.norm(values)))))
        return values + self.scale * update


class WeightedJumpingKnowledge(nn.Module):
    """Softmax-weighted sum of same-width GINE layer outputs."""

    def __init__(self, num_layers: int = 5) -> None:
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(num_layers))

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.raw_weights, dim=0)

    def forward(self, layers: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
        if len(layers) != int(self.raw_weights.numel()):
            raise ValueError(f"Expected {self.raw_weights.numel()} layer outputs, got {len(layers)}")
        weights = self.normalized_weights()
        return sum(weight * layer for weight, layer in zip(weights, layers))


class GatedAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int = 256, gate_dim: int = 64) -> None:
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, gate_dim),
            nn.Tanh(),
            nn.Linear(gate_dim, 1),
        )

    def forward(self, nodes: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(nodes).view(-1)
        weights = softmax(logits, batch)
        pooled = global_add_pool(nodes * weights.unsqueeze(-1), batch)
        return pooled, weights


class GraphPooling(nn.Module):
    def __init__(self, hidden_dim: int = 256, mode: str = "mean") -> None:
        super().__init__()
        if mode not in {"mean", "attention", "multi"}:
            raise ValueError(f"Unsupported pooling mode: {mode}")
        self.mode = mode
        self.attention = GatedAttentionPooling(hidden_dim) if mode in {"attention", "multi"} else None
        self.multi_projection = (
            nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
            )
            if mode == "multi"
            else None
        )

    def forward(self, nodes: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "mean":
            weights = softmax(torch.zeros(nodes.size(0), dtype=nodes.dtype, device=nodes.device), batch)
            return global_mean_pool(nodes, batch), weights
        if self.attention is None:
            raise RuntimeError("Attention pooling module is missing")
        attention, weights = self.attention(nodes, batch)
        if self.mode == "attention":
            return attention, weights
        mean = global_mean_pool(nodes, batch)
        maximum = global_max_pool(nodes, batch)
        if self.multi_projection is None:
            raise RuntimeError("Multi-pooling projection is missing")
        return self.multi_projection(torch.cat([attention, mean, maximum], dim=-1)), weights


class DistanceBiasedSelfAttention(nn.Module):
    """Per-graph multi-head attention with a learned shortest-path bias."""

    def __init__(
        self,
        hidden_dim: int = 128,
        heads: int = 8,
        dropout: float = 0.18,
        spd_clip: int = 10,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.spd_clip = spd_clip
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.distance_bias = nn.Embedding(spd_clip + 2, heads)
        self.dropout = nn.Dropout(dropout)

    def _dense_spd(
        self,
        ptr: torch.Tensor,
        spd_flat: torch.Tensor,
        max_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        graph_count = int(ptr.numel() - 1)
        dense = torch.full(
            (graph_count, max_nodes, max_nodes),
            self.spd_clip + 1,
            dtype=torch.long,
            device=device,
        )
        offset = 0
        for graph_index in range(graph_count):
            nodes = int((ptr[graph_index + 1] - ptr[graph_index]).item())
            values = nodes * nodes
            dense[graph_index, :nodes, :nodes] = spd_flat[offset : offset + values].view(nodes, nodes).to(device)
            offset += values
        if offset != int(spd_flat.numel()):
            raise ValueError("Batched shortest-path manifest does not match graph node counts")
        return dense

    def forward(
        self,
        nodes: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        spd_flat: torch.Tensor,
        use_distance_bias: bool,
    ) -> torch.Tensor:
        dense, valid = to_dense_batch(nodes, batch)
        batch_size, max_nodes, _ = dense.shape
        qkv = self.qkv(dense).view(batch_size, max_nodes, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-1, -2)) * (self.head_dim ** -0.5)
        if use_distance_bias:
            distances = self._dense_spd(ptr, spd_flat, max_nodes, nodes.device)
            bias = self.distance_bias(distances).permute(0, 3, 1, 2)
            scores = scores + bias
        scores = scores.masked_fill(~valid[:, None, None, :], torch.finfo(scores.dtype).min)
        attention = self.dropout(torch.softmax(scores, dim=-1))
        output = torch.matmul(attention, value).permute(0, 2, 1, 3).reshape(batch_size, max_nodes, self.hidden_dim)
        output = self.output(output)
        return output[valid]


class GPSBlock2D(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        heads: int = 8,
        dropout: float = 0.18,
        ffn_multiplier: int = 2,
        spd_clip: int = 10,
    ) -> None:
        super().__init__()
        self.local = GINEConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
            )
        )
        self.global_attention = DistanceBiasedSelfAttention(hidden_dim, heads, dropout, spd_clip)
        self.branch_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_multiplier),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_multiplier, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch: torch.Tensor,
        ptr: torch.Tensor,
        spd_flat: torch.Tensor,
        use_distance_bias: bool,
    ) -> torch.Tensor:
        local = self.local(nodes, edge_index, edge_attr)
        global_values = self.global_attention(nodes, batch, ptr, spd_flat, use_distance_bias)
        nodes = self.branch_norm(nodes + self.dropout(local + global_values))
        return self.ffn_norm(nodes + self.dropout(self.ffn(nodes)))


@dataclass(frozen=True)
class PharmaFusionConfig:
    use_jk: bool = True
    pooling: str = "multi"
    use_adapter: bool = False
    adapter_dim: int = 16
    adapter_scale: float = 0.1
    learnable_adapter_scale: bool = True
    use_gps: bool = False
    gps_use_rwse: bool = False
    gps_use_spd: bool = False
    gps_dim: int = 128
    gps_blocks: int = 2
    gps_heads: int = 8
    rwse_steps: int = 16
    spd_clip: int = 10
    fusion: str = "none"
    descriptor_dim: int = 171
    modality_dropout: float = 0.15
    head_type: str = "mlp"
    encoder_dropout: float = 0.18
    head_dropout: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PharmaFusionGPSAdapter(nn.Module):
    """Composable GINE/Adapter/JK/GPS/multi-pooling/descriptor model."""

    def __init__(self, gine_encoder: GINEEncoder, config: PharmaFusionConfig) -> None:
        super().__init__()
        if gine_encoder.hidden_dim != 256 or gine_encoder.num_layers != 5:
            raise ValueError("PharmaFusion requires the checkpoint-compatible 5x256 GINE")
        if config.fusion not in {"none", "concat", "gated", "gated_dropout"}:
            raise ValueError(config.fusion)
        self.gine_encoder = gine_encoder
        self.config = config
        self.adapters = nn.ModuleList(
            [
                ResidualAdapter(
                    256,
                    config.adapter_dim,
                    config.encoder_dropout,
                    config.adapter_scale,
                    config.learnable_adapter_scale,
                )
                for _ in range(5)
            ]
            if config.use_adapter
            else []
        )
        self.jk = WeightedJumpingKnowledge(5) if config.use_jk else None
        graph_node_dim = 256
        if config.use_gps:
            self.gps_input = nn.Linear(256, config.gps_dim)
            self.rwse_norm = nn.LayerNorm(config.rwse_steps)
            self.rwse_projection = nn.Linear(config.rwse_steps, config.gps_dim)
            self.gps_edge_projection = nn.Linear(256, config.gps_dim)
            self.gps_blocks = nn.ModuleList(
                [
                    GPSBlock2D(
                        config.gps_dim,
                        config.gps_heads,
                        config.encoder_dropout,
                        2,
                        config.spd_clip,
                    )
                    for _ in range(config.gps_blocks)
                ]
            )
            self.gps_output = nn.Sequential(
                nn.Linear(config.gps_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            graph_node_dim = 256
        self.pool = GraphPooling(graph_node_dim, config.pooling)

        self.register_buffer("descriptor_mean", torch.zeros(config.descriptor_dim), persistent=config.fusion != "none")
        self.register_buffer("descriptor_scale", torch.ones(config.descriptor_dim), persistent=config.fusion != "none")
        self.descriptor_adapter = (
            nn.Sequential(
                nn.Linear(config.descriptor_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.15),
            )
            if config.fusion != "none"
            else None
        )
        self.concat_projection = (
            nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.GELU(),
            )
            if config.fusion == "concat"
            else None
        )
        self.gate_network = nn.Linear(512, 256) if config.fusion in {"gated", "gated_dropout"} else None
        self.fusion_norm = nn.LayerNorm(256) if config.fusion in {"gated", "gated_dropout"} else None
        if config.head_type == "linear":
            self.classification_head = nn.Linear(256, 1)
        elif config.head_type == "mlp":
            self.classification_head = nn.Sequential(
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(config.head_dropout),
                nn.Linear(128, 1),
            )
        else:
            raise ValueError(config.head_type)

    def set_descriptor_scaler(self, mean: torch.Tensor, scale: torch.Tensor) -> None:
        if tuple(mean.shape) != tuple(self.descriptor_mean.shape) or tuple(scale.shape) != tuple(self.descriptor_scale.shape):
            raise ValueError("Descriptor scaler shape mismatch")
        self.descriptor_mean.copy_(mean.detach().to(self.descriptor_mean))
        self.descriptor_scale.copy_(scale.detach().to(self.descriptor_scale).clamp(min=1e-8))

    def _gine_layers(self, data: Any) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        nodes = self.gine_encoder.encode_nodes(data)
        edge_features = self.gine_encoder.encode_edges(data)
        base_layers: list[torch.Tensor] = []
        adapted_layers: list[torch.Tensor] = []
        for index, (conv, norm) in enumerate(zip(self.gine_encoder.convs, self.gine_encoder.norms)):
            residual = nodes
            base = conv(nodes, data.edge_index, edge_features)
            base = norm(F.relu(base) + residual)
            base = self.gine_encoder.dropout(base)
            base_layers.append(base)
            nodes = self.adapters[index](base) if self.config.use_adapter else base
            adapted_layers.append(nodes)
        return base_layers, adapted_layers

    def _gps_nodes(self, nodes: torch.Tensor, data: Any) -> torch.Tensor:
        if not self.config.use_gps:
            return nodes
        if not hasattr(data, "rwse") or not hasattr(data, "spd_flat"):
            raise ValueError("GPS input is missing development-only RWSE/SPD encodings")
        values = self.gps_input(nodes)
        if self.config.gps_use_rwse:
            values = values + self.rwse_projection(self.rwse_norm(data.rwse.float()))
        edge = self.gps_edge_projection(self.gine_encoder.encode_edges(data))
        for block in self.gps_blocks:
            values = block(
                values,
                data.edge_index,
                edge,
                data.batch,
                data.ptr,
                data.spd_flat,
                self.config.gps_use_spd,
            )
        return self.gps_output(values)

    def _descriptor_embedding(self, data: Any) -> torch.Tensor:
        if self.descriptor_adapter is None:
            raise RuntimeError("Descriptor branch is disabled")
        descriptor = data.global_feat.float().view(-1, self.config.descriptor_dim)
        standardized = (descriptor - self.descriptor_mean) / self.descriptor_scale
        return self.descriptor_adapter(standardized)

    def _fuse(self, graph: torch.Tensor, data: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.fusion == "none":
            return graph, torch.zeros_like(graph)
        descriptor = self._descriptor_embedding(data)
        if self.config.fusion == "concat":
            if self.concat_projection is None:
                raise RuntimeError("Concat fusion projection is missing")
            return self.concat_projection(torch.cat([graph, descriptor], dim=-1)), torch.ones_like(graph)
        if self.training and self.config.fusion == "gated_dropout" and self.config.modality_dropout > 0:
            keep = (torch.rand((descriptor.size(0), 1), device=descriptor.device) >= self.config.modality_dropout).to(descriptor.dtype)
            descriptor = descriptor * keep / max(1.0 - self.config.modality_dropout, 1e-8)
        if self.gate_network is None or self.fusion_norm is None:
            raise RuntimeError("Gated fusion modules are missing")
        gate = torch.sigmoid(self.gate_network(torch.cat([graph, descriptor], dim=-1)))
        return self.fusion_norm(graph + gate * descriptor), gate

    def forward(self, data: Any, return_stages: bool = False) -> dict[str, Any]:
        base_layers, layers = self._gine_layers(data)
        selected = self.jk(layers) if self.jk is not None else layers[-1]
        pre_gps_graph, pre_gps_attention = self.pool(selected, data.batch)
        graph_nodes = self._gps_nodes(selected, data)
        graph, atom_attention = self.pool(graph_nodes, data.batch)
        fused, fusion_gate = self._fuse(graph, data)
        output: dict[str, Any] = {
            "logits": self.classification_head(fused).view(-1),
            "embedding": fused,
            "atom_attention": atom_attention,
            "fusion_gate": fusion_gate,
            "jk_weights": (
                self.jk.normalized_weights()
                if self.jk is not None
                else torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0], device=fused.device)
            ),
        }
        if return_stages:
            raw_final = self.gine_encoder.forward_layer_outputs(data)[-1]
            output["stage_embeddings"] = {
                "E0": global_mean_pool(raw_final, data.batch),
                "E1": global_mean_pool(selected, data.batch),
                "E2": pre_gps_graph,
                "E3": graph,
                "E4": fused,
            }
            output["pre_gps_atom_attention"] = pre_gps_attention
        return output

    def set_backbone_mode(self, mode: str) -> None:
        """Configure GINE parameter efficiency without affecting new modules."""
        if mode not in {"frozen", "full", "partial"}:
            raise ValueError(mode)
        for parameter in self.gine_encoder.parameters():
            parameter.requires_grad = mode == "full"
        if mode == "partial":
            for layer in range(3, 5):
                for parameter in self.gine_encoder.convs[layer].parameters():
                    parameter.requires_grad = True
                for parameter in self.gine_encoder.norms[layer].parameters():
                    parameter.requires_grad = True

    def parameter_audit(self) -> dict[str, int | float]:
        total = sum(int(parameter.numel()) for parameter in self.parameters())
        trainable = sum(int(parameter.numel()) for parameter in self.parameters() if parameter.requires_grad)
        backbone = sum(int(parameter.numel()) for parameter in self.gine_encoder.parameters())
        adapters = sum(int(parameter.numel()) for parameter in self.adapters.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "trainable_fraction": trainable / max(total, 1),
            "backbone_parameters": backbone,
            "adapter_parameters": adapters,
            "new_module_parameters": total - backbone,
        }
