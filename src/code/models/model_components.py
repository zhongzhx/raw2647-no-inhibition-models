"""Reusable model components for multitask GNN training and graph pretraining.

这份文件只定义“模型长什么样”，不负责读取 Excel、训练循环或保存结果。
主线可以理解为：
1. GraphStructureEncoder 先看原子和化学键，得到局部结构表示；
2. GraphTransformerEncoder 可选地让所有原子互相做全局注意力；
3. FlatFeatureEncoder 只在正式训练/预测时加入 RDKit 描述符和指纹；
4. SharedEncoderCascadedGNN 输出分类和回归两个结果。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool
from torch_geometric.utils import to_dense_batch

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


def _valid_num_heads(hidden_dim: int, requested_heads: int) -> int:
    """Return a Transformer head count that divides hidden_dim."""
    heads = max(1, min(requested_heads, hidden_dim))
    while hidden_dim % heads != 0:
        heads -= 1
    return max(heads, 1)


class GraphStructureEncoder(nn.Module):
    """Atom/bond feature encoder followed by residual GINE message passing."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # 每一列 x_cat 都是一种离散原子属性，例如原子类型、手性、电荷。
        # 神经网络不能直接理解“C/N/O”这类符号，所以先映射成 embedding。
        vocab_sizes = [
            ATOM_TYPE_CLASS_COUNT + 1,
            len(CHIRALITY_TO_ID) + 1,
            len(FORMAL_CHARGE_TO_ID) + 1,
            len(DEGREE_TO_ID) + 1,
            len(TOTAL_NUM_HS_TO_ID) + 1,
            len(HYBRIDIZATION_TO_ID) + 1,
            2,
            2,
            len(RADICAL_ELECTRONS_TO_ID) + 1,
        ]
        embedding_dim = max(4, hidden_dim // 4)
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(size, embedding_dim) for size in vocab_sizes]
        )
        # x_cont 是连续原子属性，例如原子质量和 Gasteiger 电荷；和离散 embedding 拼在一起。
        node_input_dim = embedding_dim * len(vocab_sizes) + 2
        self.node_encoder = nn.Sequential(
            nn.Linear(node_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # edge_attr 里保存化学键类型、方向、立体信息等；GINEConv 需要边也有 hidden_dim 表示。
        bond_vocab_sizes = [
            len(BOND_TYPE_TO_ID) + 1,
            len(BOND_DIRECTION_TO_ID) + 1,
            len(BOND_STEREO_TO_ID) + 1,
            len(BOOL_TO_ID),
            len(BOOL_TO_ID),
        ]
        self.bond_embeddings = nn.ModuleList(
            [nn.Embedding(size, hidden_dim) for size in bond_vocab_sizes]
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
        self.hidden_dim = hidden_dim

    def encode_nodes(
        self,
        data,
        x_cat_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode x_cat and x_cont into node embeddings."""
        embedded_parts = []
        x_cat = data.x_cat.long() if x_cat_override is None else x_cat_override.long()
        for feature_idx, embedding in enumerate(self.atom_embeddings):
            # clamp 防止罕见/未知类别越过 embedding 词表边界。
            feature_values = x_cat[:, feature_idx].clamp(
                min=0,
                max=embedding.num_embeddings - 1,
            )
            embedded_parts.append(embedding(feature_values))
        node_features = torch.cat(embedded_parts + [data.x_cont.float()], dim=-1)
        return self.node_encoder(node_features)

    def encode_edges(self, data) -> torch.Tensor:
        """Encode edge_attr columns and sum them into GINE edge features."""
        edge_attr = data.edge_attr.long()
        embedded_parts = []
        for feature_idx, embedding in enumerate(self.bond_embeddings):
            # 每一种键属性单独 embedding，最后相加成一条边的整体表示。
            values = edge_attr[:, feature_idx].clamp(
                min=0,
                max=embedding.num_embeddings - 1,
            )
            embedded_parts.append(embedding(values))
        return torch.stack(embedded_parts, dim=0).sum(dim=0)

    def forward_nodes(
        self,
        data,
        x_cat_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return final node embeddings after residual GINE layers."""
        x = self.encode_nodes(data, x_cat_override=x_cat_override)
        edge_features = self.encode_edges(data)

        for conv in self.convs:
            # residual connection 保留上一层信息，减轻深层 GNN 过平滑和训练不稳定。
            residual = x
            x = F.relu(conv(x, data.edge_index, edge_features))
            x = x + residual
        return x

    def forward(self, data) -> torch.Tensor:
        """Return pooled graph embedding from structural graph features only."""
        return global_mean_pool(self.forward_nodes(data), data.batch)


class GraphTransformerEncoder(nn.Module):
    """Dense-batch Transformer over GINE node embeddings."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int | None = None,
    ) -> None:
        super().__init__()
        heads = _valid_num_heads(hidden_dim, num_heads)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=dim_feedforward or hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, node_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # PyG 图是“所有节点拼成一长条”；Transformer 需要 [batch, 节点数, hidden] 的 dense 序列。
        dense_nodes, mask = to_dense_batch(node_embeddings, batch)
        encoded = self.encoder(dense_nodes, src_key_padding_mask=~mask)
        # mask 去掉补齐出来的假节点，只平均真实原子。
        masked_encoded = encoded * mask.unsqueeze(-1).to(encoded.dtype)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).to(encoded.dtype)
        return masked_encoded.sum(dim=1) / counts


class FlatFeatureEncoder(nn.Module):
    """Encode RDKit global descriptors and fingerprints for downstream fine-tuning."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        global_dim: int,
        morgan_dim: int,
        maccs_dim: int,
        use_global_feat: bool = True,
        use_morgan_fp: bool = True,
        use_maccs_fp: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_global_feat = use_global_feat
        self.use_morgan_fp = use_morgan_fp
        self.use_maccs_fp = use_maccs_fp
        graph_feature_dim = global_dim + morgan_dim + maccs_dim
        self.encoder = nn.Sequential(
            nn.Linear(graph_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, data) -> torch.Tensor:
        if not (self.use_global_feat or self.use_morgan_fp or self.use_maccs_fp):
            # 做 graph-only 消融时，flat 分支返回全 0，模型只依赖图 encoder。
            return data.global_feat.new_zeros((data.global_feat.size(0), self.hidden_dim))

        global_feat = data.global_feat.float()
        morgan_fp = data.morgan_fp.float()
        maccs_fp = data.maccs_fp.float()
        # 这些开关用于特征消融：关闭某一类 flat feature 时，用 0 占位保持输入维度一致。
        if not self.use_global_feat:
            global_feat = torch.zeros_like(global_feat)
        if not self.use_morgan_fp:
            morgan_fp = torch.zeros_like(morgan_fp)
        if not self.use_maccs_fp:
            maccs_fp = torch.zeros_like(maccs_fp)
        flat_features = torch.cat(
            [
                global_feat,
                morgan_fp,
                maccs_fp,
            ],
            dim=-1,
        )
        return self.encoder(flat_features)


class HybridMolecularEncoder(nn.Module):
    """Fuse graph-only embedding with downstream flat-feature embedding."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        global_dim: int,
        morgan_dim: int,
        maccs_dim: int,
        use_3d: bool = False,
        use_graph_transformer: bool = False,
        transformer_num_heads: int = 4,
        transformer_num_layers: int = 2,
        use_global_feat: bool = True,
        use_morgan_fp: bool = True,
        use_maccs_fp: bool = True,
    ) -> None:
        super().__init__()
        self.use_3d = use_3d
        self.use_graph_transformer = use_graph_transformer
        self.use_global_feat = use_global_feat
        self.use_morgan_fp = use_morgan_fp
        self.use_maccs_fp = use_maccs_fp
        self.structure_encoder = GraphStructureEncoder(hidden_dim, num_layers, dropout)
        self.transformer_encoder = GraphTransformerEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
        )
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        ) if use_3d else None
        self.flat_encoder = FlatFeatureEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            global_dim=global_dim,
            morgan_dim=morgan_dim,
            maccs_dim=maccs_dim,
            use_global_feat=use_global_feat,
            use_morgan_fp=use_morgan_fp,
            use_maccs_fp=use_maccs_fp,
        )
        self.output_dim = hidden_dim * 2

    @property
    def convs(self) -> nn.ModuleList:
        """Compatibility alias for older code that inspected encoder.convs."""
        return self.structure_encoder.convs

    def encode_graph(self, data) -> torch.Tensor:
        """Return graph-only embedding; never reads global descriptors or fingerprints."""
        node_embeddings = self.structure_encoder.forward_nodes(data)
        if self.use_3d:
            if not hasattr(data, "pos"):
                raise ValueError("use_3d=True but batch does not contain data.pos")
            pos = data.pos.float()
            graph_centers = global_mean_pool(pos, data.batch)
            centered = pos - graph_centers[data.batch]
            radius = centered.pow(2).sum(dim=-1, keepdim=True).sqrt()
            graph_scale = global_mean_pool(radius, data.batch).clamp(min=1e-6)
            pos_features = torch.cat([centered / graph_scale[data.batch], radius], dim=-1)
            node_embeddings = node_embeddings + self.pos_encoder(pos_features)
        if self.use_graph_transformer:
            # Transformer 分支用于学习跨远距离原子的全局关系。
            return self.transformer_encoder(node_embeddings, data.batch)
        # GINE-only 分支直接把节点表示平均成分子表示，方便做消融。
        return global_mean_pool(node_embeddings, data.batch)

    def encode_flat(self, data) -> torch.Tensor:
        return self.flat_encoder(data)

    def forward(self, data) -> torch.Tensor:
        graph_embedding = self.encode_graph(data)
        flat_embedding = self.encode_flat(data)
        # 最终分子向量 = 图结构表示 + 下游 flat 特征表示。
        return torch.cat([graph_embedding, flat_embedding], dim=-1)


class GraphEncoder(HybridMolecularEncoder):
    """Backward-compatible name for the hybrid molecular encoder."""


class ClassificationGNN(nn.Module):
    """Binary classifier using a molecular encoder."""

    def __init__(
        self,
        encoder: HybridMolecularEncoder,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        return self.head(self.encoder(data)).view(-1)


class RegressionGNN(nn.Module):
    """pIC50 regressor using a molecular encoder."""

    def __init__(
        self,
        encoder: HybridMolecularEncoder,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> torch.Tensor:
        return self.head(self.encoder(data)).view(-1)


class SharedEncoderCascadedGNN(nn.Module):
    """Shared molecular encoder with classification and regression heads."""

    def __init__(
        self,
        encoder: HybridMolecularEncoder,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classification_head = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.regression_head = nn.Sequential(
            nn.Linear(encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(data)
        cls_logits = self.classification_head(embedding).view(-1)
        pred_pic50 = self.regression_head(embedding).view(-1)
        return cls_logits, pred_pic50


class GraphTransformerPretrainingModel(nn.Module):
    """Graph-only pretraining model with fingerprint and descriptor heads."""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        global_dim: int,
        morgan_dim: int,
        maccs_dim: int,
        use_graph_transformer: bool = True,
        transformer_num_heads: int = 4,
        transformer_num_layers: int = 2,
        mask_atom_ratio: float = 0.15,
    ) -> None:
        super().__init__()
        self.use_graph_transformer = use_graph_transformer
        self.mask_atom_ratio = mask_atom_ratio
        self.structure_encoder = GraphStructureEncoder(hidden_dim, num_layers, dropout)
        self.transformer_encoder = GraphTransformerEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            num_heads=transformer_num_heads,
            num_layers=transformer_num_layers,
        )
        self.morgan_head = nn.Linear(hidden_dim, morgan_dim)
        self.maccs_head = nn.Linear(hidden_dim, maccs_dim)
        self.global_head = nn.Linear(hidden_dim, global_dim)
        self.atom_type_head = nn.Linear(hidden_dim, ATOM_TYPE_CLASS_COUNT)

    def _mask_atom_types(
        self,
        data,
        mask_atom_ratio: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_cat = data.x_cat.long().clone()
        target_atom_types = data.x_cat[:, 0].long().clamp(max=ATOM_TYPE_CLASS_COUNT - 1)
        if mask_atom_ratio <= 0:
            mask = torch.zeros(x_cat.size(0), dtype=torch.bool, device=x_cat.device)
            return x_cat, mask, target_atom_types
        # 预训练时随机遮住部分原子类型，让模型根据周围结构猜回来，类似 BERT 的 mask 任务。
        mask = torch.rand(x_cat.size(0), device=x_cat.device) < mask_atom_ratio
        if x_cat.size(0) > 0 and not bool(mask.any()):
            mask[torch.randint(x_cat.size(0), (1,), device=x_cat.device)] = True
        x_cat[mask, 0] = ATOM_TYPE_MASK_ID
        return x_cat, mask, target_atom_types

    def encode_graph(
        self,
        data,
        mask_atom_ratio: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ratio = self.mask_atom_ratio if mask_atom_ratio is None else mask_atom_ratio
        x_cat, atom_mask, atom_targets = self._mask_atom_types(data, ratio)
        node_embeddings = self.structure_encoder.forward_nodes(data, x_cat_override=x_cat)
        if self.use_graph_transformer:
            graph_embedding = self.transformer_encoder(node_embeddings, data.batch)
        else:
            graph_embedding = global_mean_pool(node_embeddings, data.batch)
        return graph_embedding, node_embeddings, atom_mask, atom_targets

    def forward(self, data, mask_atom_ratio: float | None = None) -> dict[str, torch.Tensor]:
        graph_embedding, node_embeddings, atom_mask, atom_targets = self.encode_graph(
            data,
            mask_atom_ratio=mask_atom_ratio,
        )
        return {
            # 三个分子级预训练目标：预测 Morgan、MACCS 和标准化后的 global descriptors。
            "morgan_logits": self.morgan_head(graph_embedding),
            "maccs_logits": self.maccs_head(graph_embedding),
            "global_pred": self.global_head(graph_embedding),
            # 节点级预训练目标：只对被 mask 的原子预测原子类型。
            "atom_type_logits": self.atom_type_head(node_embeddings[atom_mask]),
            "atom_type_targets": atom_targets[atom_mask],
            "atom_type_mask_count": atom_mask.sum(),
        }

    def encoder_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        """Return only the graph encoders intended for fine-tuning transfer."""
        return {
            "structure_encoder": self.structure_encoder.state_dict(),
            "transformer_encoder": self.transformer_encoder.state_dict(),
        }
