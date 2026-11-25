"""
Batch-process unlabeled SMILES for graph-transformer pretraining.

The output graphs contain graph tensors, enhanced global descriptors, Morgan
fingerprints, and MACCS fingerprints, but no RAW264.7 activity labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
for module_dir in (CODE_ROOT / "utils", CODE_ROOT / "data_processing"):
    if str(module_dir) not in sys.path:
        sys.path.insert(0, str(module_dir))

import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from batch_process_smiles_multitask import normalize_text
from pipeline_utils import (
    tensor_shape_text,
    tensor_sum_as_int,
    validate_global_feature_consistency,
)
from smiles_to_graph import (
    ADDED_FRAGMENT_FEATURE_COUNT,
    ADDED_RDKIT_DESCRIPTOR_COUNT,
    smiles_to_pyg_data,
)


SMILES_COLUMN_CANDIDATES = [
    "smiles",
    "SMILES",
    "canonical_smiles",
    "isomeric_smiles",
]
ID_COLUMN_CANDIDATES = [
    "compound_id",
    "compound_name_or_label",
    "compound_name",
    "name",
    "Name",
    "id",
    "ID",
]


def resolve_input_path(input_path: str) -> Path:
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


def read_table(path: Path, no_header: bool = False) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, header=None if no_header else 0)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, header=None if no_header else 0)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def valid_smiles_count(values: pd.Series, sample_size: int = 500) -> int:
    count = 0
    for value in values.dropna().astype(str).head(sample_size):
        if Chem.MolFromSmiles(value.strip()) is not None:
            count += 1
    return count


def resolve_column(
    df: pd.DataFrame,
    requested: str,
    candidates: list[str],
    auto_detect_smiles: bool = False,
) -> Any:
    if requested and requested != "auto":
        try:
            column = int(requested)
        except ValueError:
            column = requested
        if column not in df.columns:
            raise ValueError(f"Requested column {requested!r} not found")
        return column

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    if auto_detect_smiles:
        scored = [
            (valid_smiles_count(df[column]), column)
            for column in df.columns
        ]
        scored = [(score, column) for score, column in scored if score > 0]
        if not scored:
            raise ValueError("Could not auto-detect SMILES column")
        scored.sort(key=lambda item: (-item[0], str(item[1])))
        return scored[0][1]
    return None


def parse_allowed_elements(text: str) -> set[str]:
    return {item.strip() for item in text.split(",") if item.strip()}


def canonicalize_smiles(smiles: str) -> tuple[Chem.Mol | None, str, str, str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, "", "", "Invalid SMILES"
    try:
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        canonical_smiles = smiles
    try:
        inchikey = Chem.MolToInchiKey(mol)
    except Exception:
        inchikey = canonical_smiles
    return mol, canonical_smiles, inchikey, ""


def molecule_filter_reason(
    mol: Chem.Mol,
    smiles: str,
    args: argparse.Namespace,
    allowed_elements: set[str],
) -> str:
    if args.remove_mixture and "." in smiles:
        return "Mixture SMILES"
    atom_symbols = {atom.GetSymbol() for atom in mol.GetAtoms()}
    disallowed = sorted(atom_symbols - allowed_elements)
    if disallowed:
        return f"Disallowed elements: {','.join(disallowed)}"
    mol_wt = Descriptors.MolWt(mol)
    if mol_wt < args.min_mol_wt:
        return f"MolWt below minimum: {mol_wt:.3f}"
    if mol_wt > args.max_mol_wt:
        return f"MolWt above maximum: {mol_wt:.3f}"
    heavy_atoms = mol.GetNumHeavyAtoms()
    if heavy_atoms < args.min_heavy_atoms:
        return f"Heavy atoms below minimum: {heavy_atoms}"
    if heavy_atoms > args.max_heavy_atoms:
        return f"Heavy atoms above maximum: {heavy_atoms}"
    return ""


def load_exclude_inchikeys(path_text: str) -> set[str]:
    if not path_text:
        return set()
    path = resolve_input_path(path_text)
    df = read_table(path, no_header=False)
    smiles_column = resolve_column(
        df,
        requested="auto",
        candidates=SMILES_COLUMN_CANDIDATES,
        auto_detect_smiles=True,
    )
    excluded = set()
    for value in df[smiles_column].dropna().astype(str):
        mol, _, inchikey, _ = canonicalize_smiles(value.strip())
        if mol is not None and inchikey:
            excluded.add(inchikey)
    return excluded


def build_feature_review_row(row: dict[str, Any], graph) -> dict[str, Any]:
    review_row = dict(row)
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
        }
    )
    if hasattr(graph, "global_feat") and hasattr(graph, "global_feature_names"):
        values = graph.global_feat.detach().cpu().view(-1).tolist()
        for name, value in zip(graph.global_feature_names, values):
            review_row[f"global_{name}"] = value
    return review_row


def process_dataframe(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[list, list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """把无标签 SMILES 表处理成预训练图；这里只做结构清洗，不读取活性标签。"""
    # 自动识别 SMILES 列，兼容不同外部天然产物数据库的列名。
    smiles_column = resolve_column(
        df,
        requested=args.smiles_column,
        candidates=SMILES_COLUMN_CANDIDATES,
        auto_detect_smiles=True,
    )
    id_column = (
        None
        if args.id_column == ""
        else resolve_column(df, args.id_column, ID_COLUMN_CANDIDATES)
    )
    allowed_elements = parse_allowed_elements(args.allowed_elements)
    # 如果提供 RAW264.7 下游数据，就按 InChIKey 排除重合分子，防止结构级泄漏。
    exclude_inchikeys = load_exclude_inchikeys(args.exclude_smiles_file)
    seen_inchikeys: set[str] = set()
    candidates: list[dict[str, Any]] = []
    failed_rows: list[dict[str, Any]] = []
    counters = {
        "raw_count": len(df),
        "parsed_success_count": 0,
        "filtered_count": 0,
        "duplicate_count": 0,
        "excluded_downstream_duplicate_count": 0,
        "deduped_count": 0,
        "sampled_count": 0,
    }

    for row_index, row in df.iterrows():
        raw_smiles = normalize_text(row[smiles_column])
        compound_id = (
            normalize_text(row[id_column])
            if id_column is not None
            else str(row_index)
        )
        if not raw_smiles:
            failed_rows.append(
                {
                    "original_row_index": row_index,
                    "compound_id": compound_id,
                    "smiles": raw_smiles,
                    "reason": "Empty SMILES",
                }
            )
            continue

        mol, canonical_smiles, inchikey, parse_reason = canonicalize_smiles(raw_smiles)
        if mol is None:
            failed_rows.append(
                {
                    "original_row_index": row_index,
                    "compound_id": compound_id,
                    "smiles": raw_smiles,
                    "reason": parse_reason,
                }
            )
            continue
        counters["parsed_success_count"] += 1

        # 分子量、重原子数、混合物和元素过滤，用来去掉明显不适合训练的小片段/大分子/盐混合物。
        filter_reason = molecule_filter_reason(mol, raw_smiles, args, allowed_elements)
        if filter_reason:
            counters["filtered_count"] += 1
            failed_rows.append(
                {
                    "original_row_index": row_index,
                    "compound_id": compound_id,
                    "smiles": raw_smiles,
                    "canonical_smiles": canonical_smiles,
                    "inchikey": inchikey,
                    "reason": filter_reason,
                }
            )
            continue
        if inchikey in seen_inchikeys:
            counters["duplicate_count"] += 1
            continue
        if inchikey in exclude_inchikeys:
            counters["excluded_downstream_duplicate_count"] += 1
            continue
        # InChIKey 去重比字符串去重更稳，因为同一结构可能有不同 SMILES 写法。
        seen_inchikeys.add(inchikey)
        candidates.append(
            {
                "original_row_index": row_index,
                "compound_id": compound_id,
                "smiles": raw_smiles,
                "canonical_smiles": canonical_smiles,
                "inchikey": inchikey,
                "mol_wt": Descriptors.MolWt(mol),
                "heavy_atom_count": mol.GetNumHeavyAtoms(),
            }
        )

    counters["deduped_count"] = len(candidates)
    if args.sample_size is not None and args.sample_size < len(candidates):
        # 可选抽样用于快速 smoke test；固定 seed 保证每次抽到同一批。
        rng = random.Random(args.seed)
        candidates = rng.sample(candidates, args.sample_size)
    candidates.sort(key=lambda row: int(row["original_row_index"]))
    counters["sampled_count"] = len(candidates)

    graph_list = []
    processed_rows = []
    for candidate in candidates:
        try:
            # 预训练图不需要标签，但需要 global_feat/Morgan/MACCS 作为自监督预测目标。
            graph = smiles_to_pyg_data(
                candidate["canonical_smiles"],
                strict=False,
                add_global_features=True,
                add_fingerprint=True,
                add_3d=False,
                explicit_hydrogens=False,
            )
        except Exception as exc:
            failed_rows.append({**candidate, "reason": str(exc)})
            continue
        graph.original_row_index = torch.tensor(
            [int(candidate["original_row_index"])],
            dtype=torch.long,
        )
        graph.compound_name_or_label = candidate["compound_id"]
        graph.canonical_smiles = candidate["canonical_smiles"]
        graph.inchikey = candidate["inchikey"]
        graph_list.append(graph)
        processed_rows.append(candidate)

    counters["final_output_count"] = len(graph_list)
    return graph_list, processed_rows, failed_rows, counters


def write_outputs(
    output_dir: Path,
    graph_list: list,
    processed_rows: list[dict[str, Any]],
    failed_rows: list[dict[str, Any]],
    counters: dict[str, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # 保存前再检查所有图的 global feature schema 一致，避免预训练目标维度混乱。
    global_dim = validate_global_feature_consistency(graph_list) if graph_list else 0
    morgan_dim = int(graph_list[0].morgan_fp.shape[-1]) if graph_list else 0
    maccs_dim = int(graph_list[0].maccs_fp.shape[-1]) if graph_list else 0
    feature_review_rows = [
        build_feature_review_row(row, graph)
        for row, graph in zip(processed_rows, graph_list)
    ]

    # pretrain_graphs.pt 是 v2 GINE 预训练器的默认输入。
    torch.save(graph_list, output_dir / "pretrain_graphs.pt")
    pd.DataFrame(processed_rows).to_csv(output_dir / "processed_index.csv", index=False)
    pd.DataFrame(failed_rows).to_csv(output_dir / "failed_smiles.csv", index=False)
    pd.DataFrame(feature_review_rows).to_csv(output_dir / "feature_review.csv", index=False)

    summary = {
        **counters,
        "global_feature_dim": global_dim,
        "morgan_dim": morgan_dim,
        "maccs_dim": maccs_dim,
        "added_rdkit_descriptor_count": ADDED_RDKIT_DESCRIPTOR_COUNT,
        "added_fragment_feature_count": ADDED_FRAGMENT_FEATURE_COUNT,
        "inchikey_dedup_enabled": True,
        "downstream_exclude_enabled": bool(args.exclude_smiles_file),
        "exclude_smiles_file": args.exclude_smiles_file,
        "output_dir": str(output_dir),
        "pretrain_graphs": str(output_dir / "pretrain_graphs.pt"),
    }
    with open(output_dir / "processing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process unlabeled SMILES for pretraining")
    parser.add_argument("--input", required=True, help="CSV 或 Excel 输入文件")
    parser.add_argument("--output_dir", default="processed_pretrain_graphs")
    parser.add_argument("--no_header", action="store_true", help="输入文件没有表头")
    parser.add_argument("--smiles_column", default="auto", help="SMILES 列名或 0-based 列索引")
    parser.add_argument("--id_column", default="auto", help="compound_id/name 列名或索引；空字符串表示用行号")
    parser.add_argument("--min_mol_wt", type=float, default=100.0)
    parser.add_argument("--max_mol_wt", type=float, default=1000.0)
    parser.add_argument("--min_heavy_atoms", type=int, default=5)
    parser.add_argument("--max_heavy_atoms", type=int, default=80)
    parser.add_argument(
        "--remove_mixture",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="默认过滤包含 '.' 的 mixture/salt SMILES",
    )
    parser.add_argument(
        "--allowed_elements",
        default="C,N,O,S,P,F,Cl,Br,I,B,Si,Se,H",
    )
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--exclude_smiles_file",
        default="",
        help="下游 RAW264.7 CSV/Excel；按 InChIKey 排除重合结构",
    )
    return parser.parse_args()


def main() -> None:
    RDLogger.DisableLog("rdApp.*")
    args = parse_args()
    input_path = resolve_input_path(args.input)
    df = read_table(input_path, no_header=args.no_header)
    graph_list, processed_rows, failed_rows, counters = process_dataframe(df, args)
    summary = write_outputs(
        Path(args.output_dir),
        graph_list,
        processed_rows,
        failed_rows,
        counters,
        args,
    )

    print("=" * 80)
    print("无标签预训练 SMILES 批量处理完成")
    print(f"原始数量: {summary['raw_count']}")
    print(f"解析成功数量: {summary['parsed_success_count']}")
    print(f"过滤数量: {summary['filtered_count']}")
    print(f"InChIKey 去重删除数量: {summary['duplicate_count']}")
    print(f"排除 downstream RAW264.7 重复分子数量: {summary['excluded_downstream_duplicate_count']}")
    print(f"去重后数量: {summary['deduped_count']}")
    print(f"最终输出数量: {summary['final_output_count']}")
    print(f"当前 global feature 总维度: {summary['global_feature_dim']}")
    print(f"新增 RDKit descriptor 数量: {summary['added_rdkit_descriptor_count']}")
    print(f"新增 Fragments 官能团数量: {summary['added_fragment_feature_count']}")
    print("预训练数据 InChIKey 去重: enabled")
    print(
        "排除 downstream RAW264.7 重复分子: "
        f"{'enabled' if summary['downstream_exclude_enabled'] else 'disabled'}"
    )
    print(f"Morgan 维度: {summary['morgan_dim']}")
    print(f"MACCS 维度: {summary['maccs_dim']}")
    print(f"输出目录: {summary['output_dir']}")


if __name__ == "__main__":
    main()
