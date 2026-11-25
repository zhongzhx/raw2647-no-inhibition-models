"""
RAW264.7 抗炎活性 SMILES 批量处理脚本。

功能：
1. 读取训练 Excel。
2. 清洗 SMILES 和 IC50 标签。
3. 调用 smiles_to_graph.smiles_to_pyg_data 转成 PyG Data。
4. 为二分类模型和精确 pIC50 回归模型准备标签。
5. 输出 processed_graphs.pt、索引 CSV、失败 CSV 和统计 JSON。
"""

from __future__ import annotations

# 标准库：命令行、JSON、数学和路径处理
import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
for module_dir in (CODE_ROOT / "utils", CODE_ROOT / "data_processing"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

# 第三方库：表格处理和张量保存
import pandas as pd
import torch
from rdkit import Chem

# 项目内模块：沿用现有 SMILES 到图的特征工程
from pipeline_utils import (
    tensor_shape_text,
    tensor_sum_as_int,
    validate_global_feature_consistency,
)
from smiles_to_graph import (
    ADDED_FRAGMENT_FEATURE_COUNT,
    ADDED_RDKIT_DESCRIPTOR_COUNT,
    GLOBAL_FEATURE_NAMES,
    smiles_to_pyg_data,
)


# 输出 CSV 需要保留的元数据列
OPTIONAL_METADATA_COLUMNS = [
    "label_confidence",
    "source_title",
    "source_doi",
    "scaffold_family",
    "note",
]

# 输出文件固定列名，即使没有失败/跳过样本也写出表头
PROCESSED_INDEX_COLUMNS = [
    "original_row_index",
    "compound_name_or_label",
    "isomeric_smiles",
    "canonical_smiles",
    "inchikey",
    "ic50_raw_uM",
    "ic50_uM_numeric",
    "pic50",
    "active_label",
    "has_exact_pic50",
    "is_censored_gt50",
    "censored_threshold_uM",
    "num_nodes",
    "num_edges",
    *OPTIONAL_METADATA_COLUMNS,
]
FAILED_SMILES_COLUMNS = [
    "original_row_index",
    "compound_name_or_label",
    "isomeric_smiles",
    "reason",
]
SKIPPED_LABEL_COLUMNS = [
    "original_row_index",
    "compound_name_or_label",
    "isomeric_smiles",
    "reason",
]
FEATURE_LABEL_REVIEW_BASE_COLUMNS = [
    "original_row_index",
    "compound_name_or_label",
    "isomeric_smiles",
    "canonical_smiles",
    "inchikey",
    "ic50_raw_uM",
    "ic50_uM_numeric",
    "pic50",
    "active_label",
    "has_exact_pic50",
    "is_censored_gt50",
    "censored_threshold_uM",
    "num_nodes",
    "num_edges",
    "x_cat_shape",
    "x_cont_shape",
    "edge_index_shape",
    "edge_attr_shape",
    "global_feat_shape",
    "morgan_fp_shape",
    "maccs_fp_shape",
    "morgan_fp_on_bits",
    "maccs_fp_on_bits",
    "has_3d_pos",
    "pos_shape",
    *OPTIONAL_METADATA_COLUMNS,
]
DUPLICATE_STRUCTURE_COLUMNS = [
    "original_row_index",
    "compound_name_or_label",
    "isomeric_smiles",
    "canonical_smiles",
    "inchikey",
    "ic50_raw_uM",
    "ic50_uM_numeric",
    "pic50",
    "active_label",
    "has_exact_pic50",
    "is_censored_gt50",
    "censored_threshold_uM",
    "duplicate_group_size",
    "kept",
    "duplicate_action",
    "duplicate_reason",
]
LABEL_CONFLICT_COLUMNS = DUPLICATE_STRUCTURE_COLUMNS


@dataclass
class ParsedIC50Label:
    """保存一个样本的两模型标签解析结果。"""

    active_label: int
    pic50: float
    has_exact_pic50: int
    is_censored_gt50: int
    censored_threshold_uM: float = math.nan


@dataclass
class ProcessedCandidate:
    """保存已解析标签和已标准化结构、等待转图的候选样本。"""

    original_row_index: Any
    compound_name_or_label: str
    isomeric_smiles: str
    canonical_smiles: str
    inchikey: str
    label: ParsedIC50Label
    row: pd.Series


def resolve_input_path(input_path: str) -> Path:
    """解析输入路径；兼容脚本搬到 data_processing/ 后的目录结构。"""
    candidate = Path(input_path)
    if candidate.exists():
        return candidate

    script_dir_candidate = Path(__file__).resolve().parent / input_path
    if script_dir_candidate.exists():
        return script_dir_candidate

    code_dir_candidate = Path(__file__).resolve().parents[1] / input_path
    if code_dir_candidate.exists():
        return code_dir_candidate

    raw_data_candidate = Path(__file__).resolve().parents[2] / "data" / "raw" / input_path
    if raw_data_candidate.exists():
        return raw_data_candidate

    raise FileNotFoundError(f"Input file not found: {input_path}")


def normalize_text(value: Any) -> str:
    """把 Excel 单元格安全转成去空格字符串，缺失值返回空字符串。"""
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_float_or_nan(value: Any) -> float:
    """安全转成 float，失败或 nan/inf 时返回 NaN。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(result) or math.isinf(result):
        return math.nan
    return result


def parse_first_number(text: str) -> float:
    """从类似 '22.5 ± 4.7' 或 '>50 μM' 中提取第一个数字。"""
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return math.nan
    return safe_float_or_nan(match.group(0))


def is_right_censored_gt50(raw_text: str) -> bool:
    """识别 >20、>50、>80、>=50、≥100 等右删失格式。"""
    compact = raw_text.replace(" ", "").replace("μ", "u").replace("µ", "u")
    compact = compact.replace("＞", ">").replace("≥", ">=")
    return bool(re.match(r"^>=?\d", compact))


def is_unsupported_left_censored(raw_text: str) -> bool:
    """识别 < 或 <= 这类左删失标签；本任务不把它们当作精确回归样本。"""
    compact = raw_text.replace(" ", "").replace("＜", "<").replace("≤", "<=")
    return compact.startswith("<")


def is_no_detected_activity(raw_text: str) -> bool:
    """Parse NS inactivity annotations and legacy ND source strings.

    These annotations supply an inactive classification label and no exact potency.
    """
    compact = raw_text.strip().lower().replace(".", "").replace(" ", "")
    return compact in {"ns", "nd", "notdetected", "noactivity", "inactive"}


def is_unreliable_label_confidence(value: Any) -> bool:
    """识别明显不可靠的 label_confidence，用于跳过训练标签。"""
    text = normalize_text(value).lower()
    if not text:
        return False
    unreliable_tokens = ["low", "uncertain", "unreliable", "exclude", "bad"]
    return any(token in text for token in unreliable_tokens)


def pic50_from_ic50_uM(ic50_uM: float) -> float:
    """根据 IC50(uM) 计算 pIC50 = -log10(IC50 * 1e-6)。"""
    if math.isnan(ic50_uM) or ic50_uM <= 0:
        return math.nan
    return -math.log10(ic50_uM * 1e-6)


def parse_ic50_label(
    raw_value: Any,
    numeric_value: Any,
    pic50_value: Any,
) -> ParsedIC50Label:
    """把 IC50 原始值/数值列解析成分类标签和精确 pIC50 回归标签。"""
    raw_text = normalize_text(raw_value)
    numeric_ic50 = safe_float_or_nan(numeric_value)
    provided_pic50 = safe_float_or_nan(pic50_value)

    # ND 表示无活性：用于分类负样本，但没有精确 IC50/pIC50 回归标签。
    if raw_text and is_no_detected_activity(raw_text):
        return ParsedIC50Label(
            active_label=0,
            pic50=math.nan,
            has_exact_pic50=0,
            is_censored_gt50=0,
            censored_threshold_uM=math.nan,
        )

    # 右删失样本（>任意数值）：分类标签为 0；不当作 exact IC50 回归标签。
    if raw_text and is_right_censored_gt50(raw_text):
        return ParsedIC50Label(
            active_label=0,
            pic50=math.nan,
            has_exact_pic50=0,
            is_censored_gt50=1,
            censored_threshold_uM=parse_first_number(raw_text),
        )

    # 左删失样本没有在当前需求中定义 loss 形式，保守跳过
    if raw_text and is_unsupported_left_censored(raw_text):
        raise ValueError(f"Unsupported left-censored IC50 value: {raw_text!r}")

    # 优先使用 ic50_uM_numeric；若缺失，则尝试从原始字符串提取第一个数值
    if math.isnan(numeric_ic50) and raw_text:
        numeric_ic50 = parse_first_number(raw_text)

    if math.isnan(numeric_ic50) or numeric_ic50 <= 0:
        raise ValueError("Cannot determine a valid IC50 value")

    # 明确数值样本按 50 uM 阈值判断分类标签；无论活性高低都保留真实 pIC50 进入回归监督。
    active_label = 1 if numeric_ic50 <= 50.0 else 0
    pic50 = provided_pic50 if not math.isnan(provided_pic50) else pic50_from_ic50_uM(numeric_ic50)
    if math.isnan(pic50):
        raise ValueError("Cannot determine exact pIC50")

    return ParsedIC50Label(
        active_label=active_label,
        pic50=pic50,
        has_exact_pic50=1,
        is_censored_gt50=0,
        censored_threshold_uM=math.nan,
    )


def require_columns(df: pd.DataFrame) -> None:
    """检查必要列，避免后面静默生成错误数据。"""
    if "isomeric_smiles" not in df.columns:
        raise ValueError("Input Excel must contain column: isomeric_smiles")
    if "ic50_raw_uM" not in df.columns and "ic50_uM_numeric" not in df.columns:
        raise ValueError("Input Excel must contain ic50_raw_uM or ic50_uM_numeric")


def get_row_value(row: pd.Series, column: str) -> Any:
    """安全读取可选列，缺失时返回 NaN。"""
    return row[column] if column in row.index else math.nan


def canonicalize_smiles(smiles: str) -> tuple[str, str]:
    """用 RDKit 解析 SMILES，并返回 canonical isomeric SMILES 和 InChIKey。"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    try:
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        canonical_smiles = smiles
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = canonical_smiles
    return canonical_smiles, inchikey


def candidate_to_duplicate_record(
    candidate: ProcessedCandidate,
    duplicate_group_size: int,
    kept: bool,
    duplicate_action: str,
    duplicate_reason: str,
) -> dict[str, Any]:
    """把重复结构候选样本展开成 CSV 记录。"""
    return {
        "original_row_index": candidate.original_row_index,
        "compound_name_or_label": candidate.compound_name_or_label,
        "isomeric_smiles": candidate.isomeric_smiles,
        "canonical_smiles": candidate.canonical_smiles,
        "inchikey": candidate.inchikey,
        "ic50_raw_uM": get_row_value(candidate.row, "ic50_raw_uM"),
        "ic50_uM_numeric": get_row_value(candidate.row, "ic50_uM_numeric"),
        "pic50": candidate.label.pic50,
        "active_label": candidate.label.active_label,
        "has_exact_pic50": candidate.label.has_exact_pic50,
        "is_censored_gt50": candidate.label.is_censored_gt50,
        "censored_threshold_uM": candidate.label.censored_threshold_uM,
        "duplicate_group_size": duplicate_group_size,
        "kept": int(kept),
        "duplicate_action": duplicate_action,
        "duplicate_reason": duplicate_reason,
    }


def representative_sort_key(candidate: ProcessedCandidate) -> tuple[int, int, int]:
    """非冲突重复结构保留一个代表：优先 exact IC50，其次非右删失，再按原始行号。"""
    try:
        row_index = int(candidate.original_row_index)
    except (TypeError, ValueError):
        row_index = 0
    return (
        -int(candidate.label.has_exact_pic50),
        int(candidate.label.is_censored_gt50),
        row_index,
    )


def deduplicate_candidates_by_structure(
    candidates: list[ProcessedCandidate],
) -> tuple[list[ProcessedCandidate], list[dict[str, Any]], list[dict[str, Any]]]:
    """按 InChIKey 去重；active/inactive 冲突的重复结构整组移除。"""
    grouped: dict[str, list[ProcessedCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.inchikey].append(candidate)

    selected_candidates: list[ProcessedCandidate] = []
    duplicate_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []

    for group in grouped.values():
        group = sorted(group, key=representative_sort_key)
        if len(group) == 1:
            selected_candidates.append(group[0])
            continue

        labels = {candidate.label.active_label for candidate in group}
        if len(labels) > 1:
            reason = "same InChIKey has both active and inactive labels"
            for candidate in group:
                record = candidate_to_duplicate_record(
                    candidate=candidate,
                    duplicate_group_size=len(group),
                    kept=False,
                    duplicate_action="removed_label_conflict",
                    duplicate_reason=reason,
                )
                duplicate_rows.append(record)
                conflict_rows.append(record)
            continue

        representative = group[0]
        selected_candidates.append(representative)
        reason = "same InChIKey and same active_label; kept one representative"
        for candidate in group:
            kept = candidate is representative
            duplicate_rows.append(
                candidate_to_duplicate_record(
                    candidate=candidate,
                    duplicate_group_size=len(group),
                    kept=kept,
                    duplicate_action=(
                        "kept_representative" if kept else "removed_duplicate_same_label"
                    ),
                    duplicate_reason=reason,
                )
            )

    selected_candidates.sort(key=representative_sort_key)
    return selected_candidates, duplicate_rows, conflict_rows


def add_multitask_labels_to_graph(
    graph,
    label: ParsedIC50Label,
    original_row_index: Any,
    compound_name_or_label: str,
    canonical_smiles: str,
    inchikey: str,
    original_smiles: str,
) -> None:
    """把多任务标签和样本元数据写入 PyG Data 对象。"""
    # y_cls 给分类 head 使用；y_reg 给回归 head 使用。
    graph.active_label = torch.tensor([label.active_label], dtype=torch.long)
    graph.y_cls = torch.tensor([label.active_label], dtype=torch.long)
    graph.pic50 = torch.tensor([label.pic50], dtype=torch.float32)
    graph.y_reg = torch.tensor([label.pic50], dtype=torch.float32)
    # has_exact_pic50 决定是否进入回归 loss；>数值右删失样本只参与分类。
    graph.has_exact_pic50 = torch.tensor([label.has_exact_pic50], dtype=torch.long)
    graph.is_censored_gt50 = torch.tensor([label.is_censored_gt50], dtype=torch.long)
    graph.censored_threshold_uM = torch.tensor(
        [label.censored_threshold_uM],
        dtype=torch.float32,
    )
    graph.original_row_index = torch.tensor([int(original_row_index)], dtype=torch.long)
    graph.compound_name_or_label = compound_name_or_label
    graph.canonical_smiles = canonical_smiles
    graph.inchikey = inchikey
    graph.original_smiles = original_smiles


def process_dataframe(
    df: pd.DataFrame,
    strict: bool,
    add_global_features: bool,
    add_fingerprint: bool,
    add_3d: bool,
    explicit_hydrogens: bool,
) -> tuple[
    list,
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """逐行处理 DataFrame，返回图列表、成功索引、SMILES 失败和标签跳过记录。"""
    require_columns(df)

    graph_list = []
    processed_rows = []
    failed_smiles = []
    skipped_labels = []
    candidates: list[ProcessedCandidate] = []

    for row_index, row in df.iterrows():
        compound_name = normalize_text(get_row_value(row, "compound_name_or_label"))
        smiles = normalize_text(get_row_value(row, "isomeric_smiles"))

        # SMILES 缺失属于结构失败，记录到 failed_smiles
        if not smiles:
            failed_smiles.append(
                {
                    "original_row_index": row_index,
                    "compound_name_or_label": compound_name,
                    "isomeric_smiles": smiles,
                    "reason": "Empty SMILES",
                }
            )
            continue

        try:
            canonical_smiles, inchikey = canonicalize_smiles(smiles)
        except Exception as exc:
            failed_smiles.append(
                {
                    "original_row_index": row_index,
                    "compound_name_or_label": compound_name,
                    "isomeric_smiles": smiles,
                    "reason": str(exc),
                }
            )
            continue

        # label_confidence 明显不可靠时跳过训练标签
        if is_unreliable_label_confidence(get_row_value(row, "label_confidence")):
            skipped_labels.append(
                {
                    "original_row_index": row_index,
                    "compound_name_or_label": compound_name,
                    "isomeric_smiles": smiles,
                    "reason": "Unreliable label_confidence",
                }
            )
            continue

        # 先解析标签；无法判断 IC50 的样本不进入图训练集
        try:
            label = parse_ic50_label(
                raw_value=get_row_value(row, "ic50_raw_uM"),
                numeric_value=get_row_value(row, "ic50_uM_numeric"),
                pic50_value=get_row_value(row, "pic50_from_ic50_numeric"),
            )
        except Exception as exc:
            skipped_labels.append(
                {
                    "original_row_index": row_index,
                    "compound_name_or_label": compound_name,
                    "isomeric_smiles": smiles,
                    "reason": str(exc),
                }
            )
            continue

        candidates.append(
            ProcessedCandidate(
                original_row_index=row_index,
                compound_name_or_label=compound_name,
                isomeric_smiles=smiles,
                canonical_smiles=canonical_smiles,
                inchikey=inchikey,
                label=label,
                row=row,
            )
        )

    candidates, duplicate_rows, conflict_rows = deduplicate_candidates_by_structure(candidates)

    for candidate in candidates:
        label = candidate.label
        # 再转图；无效 SMILES 不中断程序，记录失败原因
        try:
            # 这里生成模型真正读取的 PyG Data：节点、边、global descriptors 和 fingerprint。
            graph = smiles_to_pyg_data(
                smiles=candidate.canonical_smiles,
                y=None,
                strict=strict,
                add_global_features=add_global_features,
                add_fingerprint=add_fingerprint,
                add_3d=add_3d,
                explicit_hydrogens=explicit_hydrogens,
            )
        except Exception as exc:
            failed_smiles.append(
                {
                    "original_row_index": candidate.original_row_index,
                    "compound_name_or_label": candidate.compound_name_or_label,
                    "isomeric_smiles": candidate.isomeric_smiles,
                    "reason": str(exc),
                }
            )
            continue

        add_multitask_labels_to_graph(
            graph,
            label,
            candidate.original_row_index,
            candidate.compound_name_or_label,
            candidate.canonical_smiles,
            candidate.inchikey,
            candidate.isomeric_smiles,
        )
        graph_list.append(graph)

        processed_record = {
            "original_row_index": candidate.original_row_index,
            "compound_name_or_label": candidate.compound_name_or_label,
            "isomeric_smiles": candidate.isomeric_smiles,
            "canonical_smiles": candidate.canonical_smiles,
            "inchikey": candidate.inchikey,
            "ic50_raw_uM": get_row_value(candidate.row, "ic50_raw_uM"),
            "ic50_uM_numeric": get_row_value(candidate.row, "ic50_uM_numeric"),
            "pic50": label.pic50,
            "active_label": label.active_label,
            "has_exact_pic50": label.has_exact_pic50,
            "is_censored_gt50": label.is_censored_gt50,
            "censored_threshold_uM": label.censored_threshold_uM,
            "num_nodes": graph.num_nodes,
            "num_edges": graph.num_edges,
        }
        for column in OPTIONAL_METADATA_COLUMNS:
            processed_record[column] = get_row_value(candidate.row, column)
        processed_rows.append(processed_record)

    return graph_list, processed_rows, failed_smiles, skipped_labels, duplicate_rows, conflict_rows


def build_feature_label_review_rows(
    graph_list: list,
    processed_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """构建人工 review 用的特征-标签表：标签、图规模、指纹规模和全局描述符一并展开。"""
    review_rows = []

    for graph, processed_row in zip(graph_list, processed_rows):
        review_row = dict(processed_row)

        # 记录核心图特征张量形状，人工检查时能快速发现空图/异常维度
        review_row.update(
            {
                "x_cat_shape": tensor_shape_text(getattr(graph, "x_cat", None)),
                "x_cont_shape": tensor_shape_text(getattr(graph, "x_cont", None)),
                "edge_index_shape": tensor_shape_text(getattr(graph, "edge_index", None)),
                "edge_attr_shape": tensor_shape_text(getattr(graph, "edge_attr", None)),
                "global_feat_shape": tensor_shape_text(getattr(graph, "global_feat", None)),
                "morgan_fp_shape": tensor_shape_text(getattr(graph, "morgan_fp", None)),
                "maccs_fp_shape": tensor_shape_text(getattr(graph, "maccs_fp", None)),
                "morgan_fp_on_bits": tensor_sum_as_int(getattr(graph, "morgan_fp", None)),
                "maccs_fp_on_bits": tensor_sum_as_int(getattr(graph, "maccs_fp", None)),
                "has_3d_pos": int(hasattr(graph, "pos") and torch.is_tensor(graph.pos)),
                "pos_shape": tensor_shape_text(getattr(graph, "pos", None)),
            }
        )

        # 展开 RDKit 全局描述符，作为轻量级、可读的分子特征 review 表
        if hasattr(graph, "global_feat") and hasattr(graph, "global_feature_names"):
            values = graph.global_feat.detach().cpu().view(-1).tolist()
            for name, value in zip(graph.global_feature_names, values):
                review_row[f"global_{name}"] = value

        review_rows.append(review_row)

    return review_rows


def write_outputs(
    output_dir: Path,
    graph_list: list,
    processed_rows: list[dict[str, Any]],
    failed_smiles: list[dict[str, Any]],
    skipped_labels: list[dict[str, Any]],
    duplicate_rows: list[dict[str, Any]],
    conflict_rows: list[dict[str, Any]],
    total_rows: int,
) -> dict[str, Any]:
    """写出所有处理结果文件，并返回 summary 字典。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    # 所有分子必须有同样长度、同样顺序的 global feature，否则模型输入会错位。
    global_feature_dim = validate_global_feature_consistency(graph_list) if graph_list else 0

    # processed_graphs.pt 是正式训练脚本读取的核心文件。
    torch.save(graph_list, output_dir / "processed_graphs.pt")
    pd.DataFrame(processed_rows, columns=PROCESSED_INDEX_COLUMNS).to_csv(
        output_dir / "processed_index.csv",
        index=False,
    )
    feature_label_review_rows = build_feature_label_review_rows(graph_list, processed_rows)
    # feature_label_review.csv 给人工检查用：标签、图大小、指纹和所有 RDKit 特征都展开。
    pd.DataFrame(feature_label_review_rows).to_csv(
        output_dir / "feature_label_review.csv",
        index=False,
    )
    pd.DataFrame(failed_smiles, columns=FAILED_SMILES_COLUMNS).to_csv(
        output_dir / "failed_smiles.csv",
        index=False,
    )
    pd.DataFrame(skipped_labels, columns=SKIPPED_LABEL_COLUMNS).to_csv(
        output_dir / "skipped_or_uncertain_labels.csv",
        index=False,
    )
    pd.DataFrame(duplicate_rows, columns=DUPLICATE_STRUCTURE_COLUMNS).to_csv(
        output_dir / "duplicate_structures.csv",
        index=False,
    )
    pd.DataFrame(conflict_rows, columns=LABEL_CONFLICT_COLUMNS).to_csv(
        output_dir / "label_conflicts.csv",
        index=False,
    )

    active_1 = sum(row["active_label"] == 1 for row in processed_rows)
    active_0 = sum(row["active_label"] == 0 for row in processed_rows)
    exact_count = sum(row["has_exact_pic50"] == 1 for row in processed_rows)
    censored_count = sum(row["is_censored_gt50"] == 1 for row in processed_rows)
    graphs_with_3d_pos = sum(
        int(hasattr(graph, "pos") and torch.is_tensor(graph.pos))
        for graph in graph_list
    )

    summary = {
        "total_rows": total_rows,
        "valid_graphs": len(graph_list),
        "failed_smiles": len(failed_smiles),
        "skipped_labels": len(skipped_labels),
        "duplicate_structure_rows": len(duplicate_rows),
        "label_conflict_rows_removed": len(conflict_rows),
        "label_conflict_structure_count": len(
            {row["inchikey"] for row in conflict_rows}
        ),
        "active_label_1_count": active_1,
        "active_label_0_count": active_0,
        "exact_pic50_count": exact_count,
        "censored_gt50_count": censored_count,
        "graphs_with_3d_pos": graphs_with_3d_pos,
        "global_feature_dim": global_feature_dim,
        "global_feature_name_count": len(GLOBAL_FEATURE_NAMES),
        "added_rdkit_descriptor_count": ADDED_RDKIT_DESCRIPTOR_COUNT,
        "added_fragment_feature_count": ADDED_FRAGMENT_FEATURE_COUNT,
        "output_dir": str(output_dir),
        "feature_label_review": str(output_dir / "feature_label_review.csv"),
    }

    with open(output_dir / "processing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


def parse_args() -> argparse.Namespace:
    """解析命令行参数，路径默认适配 code/ 目录运行。"""
    parser = argparse.ArgumentParser(description="批量处理 RAW264.7 抗炎活性 SMILES 数据")
    parser.add_argument("--input", default="train data.xlsx", help="输入 Excel 文件路径")
    parser.add_argument(
        "--output_dir",
        default="processed_smiles_graphs_multitask",
        help="输出目录",
    )
    parser.add_argument("--strict", action="store_true", help="未知特征是否严格报错")
    parser.add_argument("--no_global_features", action="store_true", help="不保存全局描述符")
    parser.add_argument("--no_fingerprint", action="store_true", help="不保存分子指纹")
    parser.add_argument(
        "--add_3d",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否尝试生成 3D 坐标；默认开启，可用 --no-add_3d 关闭",
    )
    parser.add_argument("--explicit_hydrogens", action="store_true", help="图中显式加入 H")
    return parser.parse_args()


def main() -> None:
    """主入口：读取 Excel、处理样本、保存文件并打印统计。"""
    args = parse_args()
    input_path = resolve_input_path(args.input)
    output_dir = Path(args.output_dir)

    df = pd.read_excel(input_path)
    # 第一步：把每一行训练表转成带 y_cls/y_reg 的分子图。
    (
        graph_list,
        processed_rows,
        failed_smiles,
        skipped_labels,
        duplicate_rows,
        conflict_rows,
    ) = process_dataframe(
        df=df,
        strict=args.strict,
        add_global_features=not args.no_global_features,
        add_fingerprint=not args.no_fingerprint,
        add_3d=args.add_3d,
        explicit_hydrogens=args.explicit_hydrogens,
    )
    # 第二步：写出训练文件、失败记录、人工 review 表和 summary。
    summary = write_outputs(
        output_dir=output_dir,
        graph_list=graph_list,
        processed_rows=processed_rows,
        failed_smiles=failed_smiles,
        skipped_labels=skipped_labels,
        duplicate_rows=duplicate_rows,
        conflict_rows=conflict_rows,
        total_rows=len(df),
    )

    print("=" * 80)
    print("RAW264.7 抗炎活性 SMILES 批量处理完成")
    print(f"成功处理样本数: {summary['valid_graphs']}")
    print(f"active=1 样本数: {summary['active_label_1_count']}")
    print(f"active=0 样本数: {summary['active_label_0_count']}")
    print(f"精确 pIC50 样本数: {summary['exact_pic50_count']}")
    print(f"IC50 > 数值右删失样本数: {summary['censored_gt50_count']}")
    print(f"含 3D 坐标样本数: {summary['graphs_with_3d_pos']}")
    print(f"当前 global feature 总维度: {summary['global_feature_dim']}")
    print(f"新增 RDKit descriptor 数量: {summary['added_rdkit_descriptor_count']}")
    print(f"新增 Fragments 官能团数量: {summary['added_fragment_feature_count']}")
    print(f"SMILES 失败样本数: {summary['failed_smiles']}")
    print(f"标签跳过/不确定样本数: {summary['skipped_labels']}")
    print(f"重复结构记录行数: {summary['duplicate_structure_rows']}")
    print(f"因 active/inactive 冲突移除的结构数: {summary['label_conflict_structure_count']}")
    print(f"因 active/inactive 冲突移除的样本行数: {summary['label_conflict_rows_removed']}")
    print(f"输出目录: {summary['output_dir']}")
    print(f"人工 review 特征标签表: {summary['feature_label_review']}")


if __name__ == "__main__":
    main()
