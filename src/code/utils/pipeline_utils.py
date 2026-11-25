"""Common utilities shared by data processing, training, pretraining, and experiments.

这个模块放项目流程里反复用到的小工具：
读取 graph 文件、固定随机种子、保存 CSV、检查 global feature schema、
以及只用训练集统计量标准化 global_feat。
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def load_graphs(path: Path) -> list:
    """读取 PyG Data 列表；PyTorch 2.6+ 需要 weights_only=False。"""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def set_seed(seed: int) -> None:
    """固定 Python、NumPy 和 PyTorch 随机种子，让实验更可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_rows(rows: list[dict[str, Any]], path: Path) -> None:
    """保存一组字典为 CSV；rows 为空时创建空文件。"""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def tensor_shape_text(value: Any) -> str:
    """把张量 shape 转成适合人工 review 的 '行x列' 文本。"""
    if not torch.is_tensor(value):
        return ""
    return "x".join(str(dim) for dim in value.shape)


def tensor_sum_as_int(value: Any) -> int:
    """统计张量元素和；常用于查看 fingerprint 中置 1 的 bit 数。"""
    if not torch.is_tensor(value):
        return 0
    return int(value.detach().cpu().sum().item())


def validate_global_feature_consistency(graph_list: list) -> int:
    """确保一批图的 global_feat 维度和 global_feature_names 顺序完全一致。"""
    expected_dim = None
    expected_names = None
    for idx, graph in enumerate(graph_list):
        if not hasattr(graph, "global_feat") or not torch.is_tensor(graph.global_feat):
            raise ValueError(f"Graph {idx} is missing global_feat")
        if not hasattr(graph, "global_feature_names"):
            raise ValueError(f"Graph {idx} is missing global_feature_names")
        names = list(graph.global_feature_names)
        dim = int(graph.global_feat.shape[-1])
        if dim != len(names):
            raise ValueError(
                f"Graph {idx} global feature mismatch: dim={dim}, names={len(names)}"
            )
        if expected_dim is None:
            expected_dim = dim
            expected_names = names
        elif dim != expected_dim or names != expected_names:
            raise ValueError(
                f"Graph {idx} global features differ from batch schema: "
                f"dim={dim}, expected_dim={expected_dim}"
            )
    return int(expected_dim or 0)


def compute_global_feature_scaler(
    graphs: list,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """只用训练集计算 global_feat 的均值和标准差，避免验证/预测信息泄漏。"""
    if not graphs:
        raise ValueError("Cannot compute global feature scaler from an empty graph list")
    feature_names = list(getattr(graphs[0], "global_feature_names", []))
    features = []
    for idx, graph in enumerate(graphs):
        if not hasattr(graph, "global_feat") or not torch.is_tensor(graph.global_feat):
            raise ValueError(f"Graph {idx} is missing global_feat")
        if not hasattr(graph, "global_feature_names"):
            raise ValueError(f"Graph {idx} is missing global_feature_names")
        names = list(graph.global_feature_names)
        if names != feature_names:
            raise ValueError(f"Graph {idx} global_feature_names differ from training schema")
        if graph.global_feat.shape[-1] != len(names):
            raise ValueError(
                f"Graph {idx} global feature mismatch: "
                f"{graph.global_feat.shape[-1]} values vs {len(names)} names"
            )
        features.append(graph.global_feat.float())
    global_features = torch.cat(features, dim=0)
    mean = global_features.mean(dim=0, keepdim=True)
    std = global_features.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
    return mean, std, feature_names


def standardize_global_features(
    graphs: list,
    mean: torch.Tensor,
    std: torch.Tensor,
    feature_names: list[str],
) -> list:
    """复制图对象，只替换 global_feat；Morgan/MACCS 指纹保持 0/1 原样。"""
    standardized_graphs = []
    mean = mean.cpu()
    std = std.cpu()
    for idx, graph in enumerate(graphs):
        names = list(getattr(graph, "global_feature_names", []))
        if names != feature_names:
            raise ValueError(f"Graph {idx} global_feature_names differ from scaler schema")
        cloned_graph = graph.clone()
        cloned_graph.global_feat = (graph.global_feat.float() - mean) / std
        cloned_graph.global_feature_names = feature_names
        standardized_graphs.append(cloned_graph)
    return standardized_graphs
