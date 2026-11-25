"""
SMILES 到 PyTorch Geometric Data 图对象的转换工具。

本模块只负责小分子/天然产物 GNN 二分类任务的数据处理：
原子作为节点，化学键作为双向边，并额外提供分子级描述符、
Morgan fingerprint、MACCS fingerprint 和可选 3D 坐标。
"""

from __future__ import annotations

# 标准库：用于安全处理浮点数和类型提示
import math
from typing import Any

# 第三方库：RDKit 负责化学特征，torch/PyG 负责图张量
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import (
    AllChem,
    Descriptors,
    Fragments,
    MACCSkeys,
    QED,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from torch_geometric.data import Data


# =========================
# 原子离散特征字典
# =========================

# 原子类型：覆盖常见有机元素、卤素、部分金属离子和显式氢
ATOM_TYPE_TO_ID = {
    "C": 0,
    "N": 1,
    "O": 2,
    "F": 3,
    "P": 4,
    "S": 5,
    "Cl": 6,
    "Br": 7,
    "I": 8,
    "B": 9,
    "Si": 10,
    "Se": 11,
    "Na": 12,
    "K": 13,
    "Ca": 14,
    "Mg": 15,
    "H": 16,
}
UNKNOWN_ATOM_TYPE_ID = len(ATOM_TYPE_TO_ID)

# 手性：使用 RDKit atom.GetChiralTag()
CHIRALITY_TO_ID = {
    "CHI_UNSPECIFIED": 0,
    "CHI_TETRAHEDRAL_CW": 1,
    "CHI_TETRAHEDRAL_CCW": 2,
    "CHI_OTHER": 3,
}
UNKNOWN_CHIRALITY_ID = len(CHIRALITY_TO_ID)

# 形式电荷：天然产物和盐形式中常见范围
FORMAL_CHARGE_TO_ID = {
    -3: 0,
    -2: 1,
    -1: 2,
    0: 3,
    1: 4,
    2: 5,
    3: 6,
}
UNKNOWN_FORMAL_CHARGE_ID = len(FORMAL_CHARGE_TO_ID)

# 原子连接度：使用 atom.GetDegree()
DEGREE_TO_ID = {value: value for value in range(7)}
UNKNOWN_DEGREE_ID = len(DEGREE_TO_ID)

# 总氢数：heavy atom graph 下保留氢信息的重要特征
TOTAL_NUM_HS_TO_ID = {value: value for value in range(5)}
UNKNOWN_TOTAL_NUM_HS_ID = len(TOTAL_NUM_HS_TO_ID)

# 杂化方式：使用 atom.GetHybridization()
HYBRIDIZATION_TO_ID = {
    "UNSPECIFIED": 0,
    "S": 1,
    "SP": 2,
    "SP2": 3,
    "SP3": 4,
    "SP3D": 5,
    "SP3D2": 6,
}
UNKNOWN_HYBRIDIZATION_ID = len(HYBRIDIZATION_TO_ID)

# 布尔特征：False=0, True=1
BOOL_TO_ID = {
    False: 0,
    True: 1,
}

# 自由基电子数：使用 atom.GetNumRadicalElectrons()
RADICAL_ELECTRONS_TO_ID = {value: value for value in range(5)}
UNKNOWN_RADICAL_ELECTRONS_ID = len(RADICAL_ELECTRONS_TO_ID)


# =========================
# 化学键离散特征字典
# =========================

# 键类型：使用 bond.GetBondType()
BOND_TYPE_TO_ID = {
    "SINGLE": 0,
    "DOUBLE": 1,
    "TRIPLE": 2,
    "AROMATIC": 3,
}
UNKNOWN_BOND_TYPE_ID = len(BOND_TYPE_TO_ID)

# 键方向：使用 bond.GetBondDir()
BOND_DIRECTION_TO_ID = {
    "NONE": 0,
    "ENDUPRIGHT": 1,
    "ENDDOWNRIGHT": 2,
    "EITHERDOUBLE": 3,
}
UNKNOWN_BOND_DIRECTION_ID = len(BOND_DIRECTION_TO_ID)

# 键立体信息：使用 bond.GetStereo()
BOND_STEREO_TO_ID = {
    "STEREONONE": 0,
    "STEREOANY": 1,
    "STEREOZ": 2,
    "STEREOE": 3,
    "STEREOCIS": 4,
    "STEREOTRANS": 5,
}
UNKNOWN_BOND_STEREO_ID = len(BOND_STEREO_TO_ID)


# =========================
# 分子级全局描述符
# =========================

def _descriptor_function(name: str):
    """Return an RDKit descriptor function by stable descriptor name."""
    for descriptor_name, descriptor_fn in Descriptors.descList:
        if descriptor_name == name:
            return descriptor_fn
    raise AttributeError(f"RDKit descriptor not found: {name}")


# 原有 14 个基础分子级特征必须保留在最前，避免旧 review 字段丢失。
BASE_GLOBAL_FEATURES = [
    ("MolWt", Descriptors.MolWt),
    ("ExactMolWt", Descriptors.ExactMolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", rdMolDescriptors.CalcTPSA),
    ("NumHDonors", rdMolDescriptors.CalcNumHBD),
    ("NumHAcceptors", rdMolDescriptors.CalcNumHBA),
    ("NumRotatableBonds", rdMolDescriptors.CalcNumRotatableBonds),
    ("RingCount", rdMolDescriptors.CalcNumRings),
    ("NumAromaticRings", rdMolDescriptors.CalcNumAromaticRings),
    ("NumAliphaticRings", rdMolDescriptors.CalcNumAliphaticRings),
    ("HeavyAtomCount", lambda rdkit_mol: rdkit_mol.GetNumHeavyAtoms()),
    ("NumHeteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
    ("FractionCSP3", rdMolDescriptors.CalcFractionCSP3),
    ("QED", QED.qed),
]

EXTENDED_PHYSICOCHEMICAL_FEATURE_NAMES = [
    "MolMR",
    "LabuteASA",
    "BertzCT",
    "BalabanJ",
    "HeavyAtomMolWt",
    "NHOHCount",
    "NOCount",
    "NumSaturatedRings",
    "NumAromaticHeterocycles",
    "NumAliphaticHeterocycles",
    "NumSaturatedHeterocycles",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
]
EXTENDED_PHYSICOCHEMICAL_FEATURES = [
    (name, _descriptor_function(name))
    for name in EXTENDED_PHYSICOCHEMICAL_FEATURE_NAMES
]

VSA_DESCRIPTOR_PREFIXES = (
    "PEOE_VSA",
    "SlogP_VSA",
    "SMR_VSA",
    "EState_VSA",
    "VSA_EState",
)
VSA_DESCRIPTOR_FEATURES = sorted(
    [
        (name, descriptor_fn)
        for name, descriptor_fn in Descriptors.descList
        if name.startswith(VSA_DESCRIPTOR_PREFIXES)
    ],
    key=lambda item: item[0],
)

FRAGMENT_FEATURES = sorted(
    [
        (name, getattr(Fragments, name))
        for name in dir(Fragments)
        if name.startswith("fr_") and callable(getattr(Fragments, name))
    ],
    key=lambda item: item[0],
)

GLOBAL_FEATURES = (
    BASE_GLOBAL_FEATURES
    + EXTENDED_PHYSICOCHEMICAL_FEATURES
    + VSA_DESCRIPTOR_FEATURES
    + FRAGMENT_FEATURES
)

# 分子级特征顺序固定，便于训练阶段复用和解释
GLOBAL_FEATURE_NAMES = [name for name, _ in GLOBAL_FEATURES]
BASE_GLOBAL_FEATURE_NAMES = [
    "MolWt",
    "ExactMolWt",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "FractionCSP3",
    "QED",
]
ADDED_RDKIT_DESCRIPTOR_COUNT = (
    len(EXTENDED_PHYSICOCHEMICAL_FEATURES) + len(VSA_DESCRIPTOR_FEATURES)
)
ADDED_FRAGMENT_FEATURE_COUNT = len(FRAGMENT_FEATURES)


def _enum_name(value: Any) -> str:
    """提取 RDKit 枚举值的稳定名称，例如 ChiralType.CHI_UNSPECIFIED -> CHI_UNSPECIFIED。"""
    return str(value).split(".")[-1]


def _safe_float(value: Any, default: float = 0.0) -> float:
    """将 RDKit 结果安全转成 float，遇到 nan/inf/异常时返回默认值。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isnan(result) or math.isinf(result):
        return default
    return result


def _lookup_feature(
    value: Any,
    mapping: dict[Any, int],
    unknown_id: int,
    feature_name: str,
    strict: bool,
) -> int:
    """把离散特征映射成整数 ID；严格模式下遇到未知值直接报错。"""
    if value in mapping:
        return mapping[value]

    if strict:
        allowed = ", ".join(str(key) for key in mapping)
        raise ValueError(
            f"Unsupported {feature_name}: {value!r}. "
            f"Allowed values are: {allowed}"
        )

    return unknown_id


def _compute_gasteiger_charges(mol: Chem.Mol) -> None:
    """计算 Gasteiger 部分电荷；失败时不抛出，让连续特征回退到 0.0。"""
    try:
        AllChem.ComputeGasteigerCharges(mol)
    except Exception:
        return


def _atom_categorical_features(atom: Chem.Atom, strict: bool) -> list[int]:
    """提取单个原子的 9 个离散特征，顺序必须与需求保持一致。"""
    atom_type = atom.GetSymbol()
    chirality = _enum_name(atom.GetChiralTag())
    formal_charge = atom.GetFormalCharge()
    degree = atom.GetDegree()
    total_num_hs = atom.GetTotalNumHs()
    hybridization = _enum_name(atom.GetHybridization())
    is_aromatic = atom.GetIsAromatic()
    is_in_ring = atom.IsInRing()
    radical_electrons = atom.GetNumRadicalElectrons()

    return [
        _lookup_feature(
            atom_type,
            ATOM_TYPE_TO_ID,
            UNKNOWN_ATOM_TYPE_ID,
            "atom type",
            strict,
        ),
        _lookup_feature(
            chirality,
            CHIRALITY_TO_ID,
            UNKNOWN_CHIRALITY_ID,
            "chirality",
            strict,
        ),
        _lookup_feature(
            formal_charge,
            FORMAL_CHARGE_TO_ID,
            UNKNOWN_FORMAL_CHARGE_ID,
            "formal charge",
            strict,
        ),
        _lookup_feature(
            degree,
            DEGREE_TO_ID,
            UNKNOWN_DEGREE_ID,
            "degree",
            strict,
        ),
        _lookup_feature(
            total_num_hs,
            TOTAL_NUM_HS_TO_ID,
            UNKNOWN_TOTAL_NUM_HS_ID,
            "total_num_hs",
            strict,
        ),
        _lookup_feature(
            hybridization,
            HYBRIDIZATION_TO_ID,
            UNKNOWN_HYBRIDIZATION_ID,
            "hybridization",
            strict,
        ),
        BOOL_TO_ID[is_aromatic],
        BOOL_TO_ID[is_in_ring],
        _lookup_feature(
            radical_electrons,
            RADICAL_ELECTRONS_TO_ID,
            UNKNOWN_RADICAL_ELECTRONS_ID,
            "radical_electrons",
            strict,
        ),
    ]


def _atom_continuous_features(atom: Chem.Atom) -> list[float]:
    """提取单个原子的连续特征：原子质量和 Gasteiger 部分电荷。"""
    atomic_mass = _safe_float(atom.GetMass()) / 100.0

    # Gasteiger 电荷可能不存在、失败、nan 或 inf，统一回退为 0.0
    gasteiger_charge = 0.0
    if atom.HasProp("_GasteigerCharge"):
        gasteiger_charge = _safe_float(atom.GetProp("_GasteigerCharge"))

    return [atomic_mass, gasteiger_charge]


def _bond_features(bond: Chem.Bond, strict: bool) -> list[int]:
    """提取单条化学键的 5 个离散特征，两个方向共用同一组特征。"""
    bond_type = _enum_name(bond.GetBondType())
    bond_direction = _enum_name(bond.GetBondDir())
    bond_stereo = _enum_name(bond.GetStereo())
    is_conjugated = bond.GetIsConjugated()
    is_in_ring = bond.IsInRing()

    return [
        _lookup_feature(
            bond_type,
            BOND_TYPE_TO_ID,
            UNKNOWN_BOND_TYPE_ID,
            "bond type",
            strict,
        ),
        _lookup_feature(
            bond_direction,
            BOND_DIRECTION_TO_ID,
            UNKNOWN_BOND_DIRECTION_ID,
            "bond direction",
            strict,
        ),
        _lookup_feature(
            bond_stereo,
            BOND_STEREO_TO_ID,
            UNKNOWN_BOND_STEREO_ID,
            "bond stereo",
            strict,
        ),
        BOOL_TO_ID[is_conjugated],
        BOOL_TO_ID[is_in_ring],
    ]


def _compute_global_features(mol: Chem.Mol) -> tuple[torch.Tensor, list[str]]:
    """计算分子级 RDKit 描述符，并返回 shape=[1, num_global_features] 的张量。"""
    values = []
    for _, descriptor_fn in GLOBAL_FEATURES:
        try:
            # 单个 descriptor 失败时只把该特征置 0，避免整批分子处理被一个异常打断。
            values.append(_safe_float(descriptor_fn(mol)))
        except Exception:
            values.append(0.0)
    global_feat = torch.tensor([values], dtype=torch.float32)
    if global_feat.shape[-1] != len(GLOBAL_FEATURE_NAMES):
        raise RuntimeError(
            "Global feature dimension mismatch: "
            f"{global_feat.shape[-1]} values vs {len(GLOBAL_FEATURE_NAMES)} names"
        )
    return global_feat, list(GLOBAL_FEATURE_NAMES)


def _compute_fingerprints(mol: Chem.Mol) -> tuple[torch.Tensor, torch.Tensor]:
    """计算 Morgan fingerprint 和 MACCS fingerprint，返回 float32 张量。"""
    try:
        # Morgan 指纹固定 2048 bit，适合传统 ML baseline 和下游 flat feature 分支。
        morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=2,
            fpSize=2048,
        )
        morgan_fp = morgan_generator.GetFingerprint(mol)
        morgan_array = np.zeros((2048,), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(morgan_fp, morgan_array)
    except Exception:
        morgan_array = np.zeros((2048,), dtype=np.float32)

    try:
        # MACCS 是另一套固定规则的结构 key，和 Morgan 互补。
        maccs_fp = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = np.zeros((maccs_fp.GetNumBits(),), dtype=np.float32)
        DataStructs.ConvertToNumpyArray(maccs_fp, maccs_array)
    except Exception:
        # RDKit MACCS 默认长度通常为 167；异常时用默认长度空指纹兜底
        maccs_array = np.zeros((167,), dtype=np.float32)

    return (
        torch.tensor(morgan_array, dtype=torch.float32).view(1, -1),
        torch.tensor(maccs_array, dtype=torch.float32).view(1, -1),
    )


def _try_add_3d_coordinates(
    data: Data,
    mol: Chem.Mol,
    explicit_hydrogens: bool,
) -> None:
    """尝试生成 3D 坐标；失败时不添加 data.pos，也不影响图转换。"""
    try:
        mol_with_h = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 2026
        embed_status = AllChem.EmbedMolecule(mol_with_h, params)
        if embed_status != 0:
            return

        try:
            AllChem.UFFOptimizeMolecule(mol_with_h)
        except Exception:
            pass

        conformer = mol_with_h.GetConformer()
        atom_count = mol_with_h.GetNumAtoms() if explicit_hydrogens else mol.GetNumAtoms()
        coordinates = []
        for atom_idx in range(atom_count):
            position = conformer.GetAtomPosition(atom_idx)
            coordinates.append([position.x, position.y, position.z])

        data.pos = torch.tensor(coordinates, dtype=torch.float32)
    except Exception:
        return


def _label_tensor(y: int | float) -> torch.Tensor:
    """把图级标签转换为张量：整数标签用于分类，浮点标签用于回归。"""
    if isinstance(y, bool):
        return torch.tensor([int(y)], dtype=torch.long)
    if isinstance(y, int):
        return torch.tensor([y], dtype=torch.long)
    return torch.tensor([y], dtype=torch.float32)


def smiles_to_pyg_data(
    smiles: str,
    y: int | float | None = None,
    strict: bool = False,
    add_global_features: bool = True,
    add_fingerprint: bool = True,
    add_3d: bool = False,
    explicit_hydrogens: bool = False,
) -> Data:
    """
    Convert a SMILES string into a PyTorch Geometric Data object.

    参数说明：
    - strict=False 时，未知离散特征映射到 unknown id。
    - add_global_features=True 时，添加 RDKit 分子级描述符。
    - add_fingerprint=True 时，添加 Morgan fingerprint 和 MACCS fingerprint。
    - add_3d=True 时，尝试生成 3D 坐标并保存到 data.pos。
    - explicit_hydrogens=False 时，默认使用 heavy atom graph。
    """
    # 解析 SMILES；无效 SMILES 直接报错，由批量函数负责记录失败行
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")

    # 默认使用 heavy atom graph；只有显式要求时才把 H 加入节点和边
    graph_mol = Chem.AddHs(mol) if explicit_hydrogens else mol

    # Gasteiger 电荷写入 graph_mol 的 atom property，连续特征会安全读取
    _compute_gasteiger_charges(graph_mol)

    # 提取原子离散特征和连续特征
    atom_cat_features = [
        _atom_categorical_features(atom, strict=strict)
        for atom in graph_mol.GetAtoms()
    ]
    atom_cont_features = [
        _atom_continuous_features(atom)
        for atom in graph_mol.GetAtoms()
    ]

    x_cat = torch.tensor(atom_cat_features, dtype=torch.long)
    x_cont = torch.tensor(atom_cont_features, dtype=torch.float32)

    # 构建双向边索引和 5 维边特征
    edge_indices: list[list[int]] = []
    edge_features: list[list[int]] = []
    for bond in graph_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        encoded_bond = _bond_features(bond, strict=strict)

        # PyG 中无向图通常用 i->j 和 j->i 两条有向边表示
        edge_indices.extend([[begin_idx, end_idx], [end_idx, begin_idx]])
        edge_features.extend([encoded_bond, encoded_bond])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.long)
    else:
        # 单原子分子没有化学键，也必须保持固定 shape
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.long)

    # data.x 保持旧代码兼容，同时新增 x_cat / x_cont
    data = Data(x=x_cat, edge_index=edge_index, edge_attr=edge_attr)
    data.x_cat = x_cat
    data.x_cont = x_cont
    data.smiles = smiles

    # 可选添加分子级描述符
    if add_global_features:
        # global_feature_names 与 global_feat 一起保存，后续训练/预测可检查特征顺序是否一致。
        global_feat, global_feature_names = _compute_global_features(mol)
        data.global_feat = global_feat
        data.global_feature_names = global_feature_names

    # 可选添加分子指纹
    if add_fingerprint:
        morgan_fp, maccs_fp = _compute_fingerprints(mol)
        data.morgan_fp = morgan_fp
        data.maccs_fp = maccs_fp

    # 可选添加 3D 坐标，失败不影响主转换
    if add_3d:
        _try_add_3d_coordinates(data, graph_mol, explicit_hydrogens)

    # 可选添加图级标签
    if y is not None:
        data.y = _label_tensor(y)

    return data


def dataframe_to_graph_list(
    df,
    smiles_col: str = "smiles",
    label_col: str | None = "label",
    strict: bool = False,
    add_global_features: bool = True,
    add_fingerprint: bool = True,
    add_3d: bool = False,
    explicit_hydrogens: bool = False,
):
    """
    Convert a pandas DataFrame containing SMILES and labels into a list of PyG Data objects.
    Invalid SMILES should be skipped and recorded.
    """
    # 保存成功转换的图，以及失败行的 row_index / smiles / reason
    graph_list = []
    failed_records = []

    # 逐行处理，保留 DataFrame 原始 index，方便回查数据清洗问题
    for row_index, row in df.iterrows():
        smiles = row[smiles_col]
        label = row[label_col] if label_col is not None else None

        try:
            graph = smiles_to_pyg_data(
                smiles=smiles,
                y=label,
                strict=strict,
                add_global_features=add_global_features,
                add_fingerprint=add_fingerprint,
                add_3d=add_3d,
                explicit_hydrogens=explicit_hydrogens,
            )
            graph_list.append(graph)
        except Exception as exc:
            failed_records.append(
                {
                    "row_index": row_index,
                    "smiles": smiles,
                    "reason": str(exc),
                }
            )

    return graph_list, failed_records


if __name__ == "__main__":
    # 天然产物示例：用于快速检查节点、边、全局描述符和指纹张量
    test_smiles = [
        "O=C(C1(C)C)CC[C@@]2(C)C1=C(O)C([C@@H](C)[C@]2(O)CCC3=COC=C3)=O",
    ]

    # 打印核心张量形状和前几行，便于人工检查数据处理是否符合预期
    for smiles_item in test_smiles:
        graph_data = smiles_to_pyg_data(smiles_item)
        print("=" * 80)
        print("smiles:", graph_data.smiles)
        print("x_cat shape:", graph_data.x_cat.shape)
        print("x_cont shape:", graph_data.x_cont.shape)
        print("edge_index shape:", graph_data.edge_index.shape)
        print("edge_attr shape:", graph_data.edge_attr.shape)
        print("global_feat shape:", graph_data.global_feat.shape)
        print("len(global_feature_names):", len(graph_data.global_feature_names))
        print("morgan_fp shape:", graph_data.morgan_fp.shape)
        print("maccs_fp shape:", graph_data.maccs_fp.shape)
        print("global_feat/name match:", graph_data.global_feat.shape[-1] == len(graph_data.global_feature_names))
        print("global feature total dim:", len(GLOBAL_FEATURE_NAMES))
        print("added RDKit descriptor count:", ADDED_RDKIT_DESCRIPTOR_COUNT)
        print("added Fragments count:", ADDED_FRAGMENT_FEATURE_COUNT)
        print("num_nodes:", graph_data.num_nodes)
        print("num_edges:", graph_data.num_edges)
        print("x_cat 前 5 行:")
        print(graph_data.x_cat[:5])
        print("x_cont 前 5 行:")
        print(graph_data.x_cont[:5])
        print("edge_attr 前 5 行:")
        print(graph_data.edge_attr[:5])
        print("global_feat:")
        print(graph_data.global_feat)
