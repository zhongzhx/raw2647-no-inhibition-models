#!/usr/bin/env python3
"""Run selected-model structural interpretation on current scaffold-OOF outputs.

The analysis combines cross-scaffold motif effects, scaffold-level prediction
patterns, model-disagreement/error diagnostics, and a Morgan-fingerprint
surrogate used only for local atom-environment visualisation. Surrogate maps
are explicitly distinguished from direct neural gradients.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "rebuild_current_20260727"
CHEMICAL = OUT / "tables" / "chemical_space" / "chemical_space_dataset_current955.csv"
FIGURE_DIR = OUT / "figures" / "interpretability"
TABLE_DIR = OUT / "tables" / "interpretability"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#6B7280"
GRID = "#D1D5DB"

MOTIFS = {
    "phenolic hydroxyl": "[OX2H]-c",
    "catechol": "c1c([OX2H])c([OX2H])ccc1",
    "resorcinol": "c1c([OX2H])cc([OX2H])cc1",
    "aromatic methoxy": "c-[OX2]-[CH3]",
    "carboxylic acid": "C(=O)[OX2H1]",
    "ester/lactone": "C(=O)O[#6]",
    "amide": "C(=O)N",
    "alpha,beta-unsaturated carbonyl": "[C,c]=[C,c]-C(=O)",
    "flavone-like core": "O=c1cc(-c2ccccc2)oc2ccccc12",
    "coumarin-like core": "O=c1oc2ccccc2cc1",
    "indole": "c1ccc2[nH]ccc2c1",
    "quinoline": "c1ccc2ncccc2c1",
    "basic aliphatic amine": "[NX3;H0,H1,H2;!$(N-C=O);!$(N-S=O)]",
    "furan": "c1cocc1",
    "sulfur heteroaromatic": "[s;R]",
    "glycoside-like ring": "[O;R]1[CH;R][CH;R][CH;R][CH;R]1",
    "epoxide": "[O;R]1[CH2,CH;R][CH2,CH;R]1",
    "halogenated carbon": "[#6]-[F,Cl,Br,I]",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", facecolor="white")
    plt.close(fig)




def cluster_bootstrap_difference(
    frame: pd.DataFrame,
    indicator: str,
    outcome: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float, float]:
    groups = [group.index.to_numpy(int) for _, group in frame.groupby("scaffold_group")]
    rng = np.random.default_rng(seed)

    def difference(indices: np.ndarray) -> float:
        sample = frame.loc[indices]
        present = sample.loc[sample[indicator].eq(1), outcome]
        absent = sample.loc[sample[indicator].eq(0), outcome]
        if len(present) == 0 or len(absent) == 0:
            return np.nan
        return float(present.mean() - absent.mean())

    observed = difference(frame.index.to_numpy(int))
    values = []
    for _ in range(repetitions):
        sampled = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[i] for i in sampled])
        value = difference(indices)
        if np.isfinite(value):
            values.append(value)
    boot = np.asarray(values, dtype=float)
    low, high = np.quantile(boot, [0.025, 0.975])
    p_value = min(
        1.0,
        2
        * min(
            (np.sum(boot <= 0) + 1) / (len(boot) + 1),
            (np.sum(boot >= 0) + 1) / (len(boot) + 1),
        ),
    )
    return observed, float(low), float(high), float(p_value)


def load_inputs(run_dir: Path) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completion = json.loads((run_dir / "RUN_COMPLETION.json").read_text())
    if completion.get("status") != "FORMAL_OOF_COMPLETE":
        raise RuntimeError("Formal OOF is incomplete")
    lock = json.loads((run_dir / "SELECTED_MODEL_LOCK.json").read_text())
    selected_id = str(lock["selected_model_id"])
    ensemble = pd.read_csv(run_dir / "final_ensemble_oof_predictions.csv")
    selected = ensemble.loc[ensemble["model_id"].eq(selected_id)].copy()
    if len(selected) != 955:
        raise RuntimeError("Selected-model OOF predictions do not cover 955 molecules")
    all_predictions = ensemble.loc[~ensemble["model_id"].eq("BEST")].copy()
    seed_averaged = pd.read_csv(run_dir / "seed_averaged_oof_predictions.csv")
    chemical = pd.read_csv(CHEMICAL)
    merged = selected.merge(
        chemical[
            [
                "inchikey",
                "scaffold_family_completed",
                "scaffold_family_source",
                "compound_name_or_label",
            ]
        ],
        on="inchikey",
        how="left",
        validate="one_to_one",
    )
    if merged["scaffold_family_completed"].isna().any():
        raise RuntimeError("Chemical-space metadata alignment failed")
    return selected_id, merged, all_predictions, seed_averaged


def motif_analysis(
    selected: pd.DataFrame,
    seed_averaged: pd.DataFrame,
    selected_id: str,
    repetitions: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = selected.reset_index(drop=True).copy()
    mols = [Chem.MolFromSmiles(value) for value in frame["canonical_smiles"]]
    compiled = {name: Chem.MolFromSmarts(smarts) for name, smarts in MOTIFS.items()}
    for name, query in compiled.items():
        if query is None:
            raise ValueError(f"Invalid motif SMARTS: {name}")
        frame[name] = [int(mol.HasSubstructMatch(query)) for mol in mols]

    assignment = seed_averaged.loc[seed_averaged["model_id"].eq(selected_id)]
    assignment = assignment[["assignment_seed", "sample_index", "probability"]]
    rows = []
    for offset, name in enumerate(MOTIFS):
        present = frame[name].eq(1)
        count = int(present.sum())
        scaffold_count = int(frame.loc[present, "scaffold_group"].nunique())
        if count < 20 or scaffold_count < 5 or count == len(frame):
            continue
        pred = cluster_bootstrap_difference(
            frame, name, "probability", repetitions, 20260727 + offset * 2
        )
        observed = cluster_bootstrap_difference(
            frame, name, "active_label", repetitions, 20260728 + offset * 2
        )
        assignment_diffs = []
        for _, group in assignment.groupby("assignment_seed"):
            joined = frame[["sample_index", name]].merge(
                group, on="sample_index", validate="one_to_one"
            )
            assignment_diffs.append(
                float(
                    joined.loc[joined[name].eq(1), "probability"].mean()
                    - joined.loc[joined[name].eq(0), "probability"].mean()
                )
            )
        sign = np.sign(pred[0])
        sign_consistency = float(
            np.mean(np.sign(np.asarray(assignment_diffs)) == sign)
        )
        rows.append(
            {
                "motif": name,
                "smarts": MOTIFS[name],
                "molecule_count": count,
                "scaffold_count": scaffold_count,
                "predicted_probability_difference": pred[0],
                "predicted_ci95_low": pred[1],
                "predicted_ci95_high": pred[2],
                "predicted_bootstrap_p": pred[3],
                "observed_active_fraction_difference": observed[0],
                "observed_ci95_low": observed[1],
                "observed_ci95_high": observed[2],
                "observed_bootstrap_p": observed[3],
                "assignment_sign_consistency": sign_consistency,
                **{
                    f"assignment_difference_{i + 1}": value
                    for i, value in enumerate(assignment_diffs)
                },
            }
        )
    result = pd.DataFrame(rows)
    result["cross_scaffold_stable"] = (
        (result["predicted_ci95_low"] * result["predicted_ci95_high"] > 0)
        & (result["assignment_sign_consistency"] >= 0.8)
    ).astype(int)
    presence = frame[
        ["sample_index", "inchikey", "scaffold_group", "active_label", "probability"]
        + list(MOTIFS)
    ]
    return result.sort_values("predicted_probability_difference"), presence


def scaffold_analysis(selected: pd.DataFrame) -> pd.DataFrame:
    table = (
        selected.loc[selected["scaffold_group"].ne("[ACYCLIC]")]
        .groupby("scaffold_group", as_index=False)
        .agg(
            molecule_count=("inchikey", "size"),
            active_count=("active_label", "sum"),
            observed_active_fraction=("active_label", "mean"),
            mean_selected_probability=("probability", "mean"),
            probability_sd=("probability", "std"),
            dominant_family=(
                "scaffold_family_completed",
                lambda values: values.value_counts().index[0],
            ),
            representative_inchikey=("inchikey", "first"),
            representative_smiles=("canonical_smiles", "first"),
        )
    )
    table["inactive_count"] = table["molecule_count"] - table["active_count"]
    return table.sort_values(
        ["mean_selected_probability", "molecule_count"], ascending=[False, False]
    )


def draw_global_interpretability(
    motifs: pd.DataFrame, scaffolds: pd.DataFrame, selected_id: str
) -> None:
    stable = motifs.loc[motifs["cross_scaffold_stable"].eq(1)].copy()
    if len(stable) < 4:
        stable = motifs.iloc[
            np.argsort(np.abs(motifs["predicted_probability_difference"]))[-8:]
        ].copy()
    stable = stable.sort_values("predicted_probability_difference")
    eligible_scaffolds = scaffolds.loc[scaffolds["molecule_count"].ge(5)]
    top = eligible_scaffolds.head(6)
    bottom = eligible_scaffolds.tail(6)
    scaffold_plot = pd.concat([bottom, top]).drop_duplicates("scaffold_group")
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), constrained_layout=True)

    y = np.arange(len(stable))
    value = stable["predicted_probability_difference"].to_numpy(float)
    low = stable["predicted_ci95_low"].to_numpy(float)
    high = stable["predicted_ci95_high"].to_numpy(float)
    colors = [ORANGE if v > 0 else BLUE for v in value]
    axes[0].errorbar(
        value, y, xerr=[value - low, high - value],
        fmt="none", ecolor=GREY, capsize=3, linewidth=1.2
    )
    axes[0].scatter(value, y, c=colors, s=35, zorder=3)
    axes[0].axvline(0, color=GREY, linestyle="--")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(stable["motif"])
    axes[0].set(
        title="A  Cross-scaffold motif associations",
        xlabel="Selected-model probability difference (present − absent)",
    )

    scaffold_plot = scaffold_plot.sort_values("mean_selected_probability")
    labels = [
        f"S{index:02d} | {row.dominant_family} (n={row.molecule_count})"
        for index, row in enumerate(scaffold_plot.itertuples(index=False), 1)
    ]
    y = np.arange(len(scaffold_plot))
    axes[1].barh(
        y,
        scaffold_plot["mean_selected_probability"],
        color=[
            ORANGE if value >= 0.5 else BLUE
            for value in scaffold_plot["mean_selected_probability"]
        ],
    )
    axes[1].scatter(
        scaffold_plot["observed_active_fraction"], y,
        color="#111827", marker="D", s=25, label="Observed active fraction"
    )
    axes[1].axvline(0.5, color=GREY, linestyle="--")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(labels, fontsize=7.3)
    axes[1].set_xlim(0, 1.05)
    axes[1].set(
        title="B  Recurrent high- and low-score scaffolds",
        xlabel="Mean probability / observed active fraction",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(
        f"Structural patterns learned by selected model {selected_id}",
        fontsize=14, fontweight="bold"
    )
    save(fig, "F10_global_structural_interpretation")


def morgan_matrix(mols: list[Chem.Mol]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    matrix = np.zeros((len(mols), 2048), dtype=np.float32)
    for row, mol in enumerate(mols):
        fp = generator.GetFingerprint(mol)
        DataStructs.ConvertToNumpyArray(fp, matrix[row])
    return matrix


def atom_environment_bits(
    mol: Chem.Mol,
    bit_info: dict[int, tuple[tuple[int, int], ...]],
) -> list[set[int]]:
    affected = [set() for _ in range(mol.GetNumAtoms())]
    for bit, occurrences in bit_info.items():
        for center, radius in occurrences:
            atoms = {int(center)}
            if radius > 0:
                environment = Chem.FindAtomEnvironmentOfRadiusN(mol, int(radius), int(center))
                for bond_index in environment:
                    bond = mol.GetBondWithIdx(int(bond_index))
                    atoms.add(bond.GetBeginAtomIdx())
                    atoms.add(bond.GetEndAtomIdx())
            for atom in atoms:
                affected[atom].add(int(bit))
    return affected


def surrogate_and_local_maps(
    selected: pd.DataFrame, selected_id: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = selected.sort_values("sample_index").reset_index(drop=True).copy()
    mols = [Chem.MolFromSmiles(value) for value in frame["canonical_smiles"]]
    matrix = morgan_matrix(mols)
    target = frame["probability"].to_numpy(float)
    groups = frame["scaffold_group"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(frame), dtype=float)
    fold_rows = []
    for fold, (train, valid) in enumerate(splitter.split(matrix, target, groups)):
        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=20260727 + fold,
            n_jobs=-1,
        )
        model.fit(matrix[train], target[train])
        oof[valid] = model.predict(matrix[valid])
        fold_rows.append(
            {
                "fold": fold,
                "train_count": len(train),
                "validation_count": len(valid),
                "r2": r2_score(target[valid], oof[valid]),
                "mae": mean_absolute_error(target[valid], oof[valid]),
            }
        )
    fidelity = pd.DataFrame(
        fold_rows
        + [
            {
                "fold": "pooled",
                "train_count": len(frame),
                "validation_count": len(frame),
                "r2": r2_score(target, oof),
                "mae": mean_absolute_error(target, oof),
            }
        ]
    )
    frame["surrogate_oof_probability"] = oof
    model = RandomForestRegressor(
        n_estimators=1000,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=20260727,
        n_jobs=-1,
    )
    model.fit(matrix, target)

    # Choose high-scoring measured actives from distinct scaffolds.
    representatives = (
        frame.loc[frame["active_label"].eq(1)]
        .sort_values("probability", ascending=False)
        .drop_duplicates("scaffold_group")
        .head(12)
        .copy()
    )
    map_dir = FIGURE_DIR / "local_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    atom_rows = []
    image_paths = []
    for rank, row in enumerate(representatives.itertuples(index=False), 1):
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        additional = rdFingerprintGenerator.AdditionalOutput()
        additional.AllocateBitInfoMap()
        fp = generator.GetFingerprint(mol, additionalOutput=additional)
        base = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, base)
        base_prediction = float(model.predict(base.reshape(1, -1))[0])
        affected = atom_environment_bits(mol, additional.GetBitInfoMap())
        weights = []
        for atom_index, bits in enumerate(affected):
            masked = base.copy()
            if bits:
                masked[list(bits)] = 0
            masked_prediction = float(model.predict(masked.reshape(1, -1))[0])
            weight = base_prediction - masked_prediction
            weights.append(weight)
            atom = mol.GetAtomWithIdx(atom_index)
            atom_rows.append(
                {
                    "rank": rank,
                    "inchikey": row.inchikey,
                    "selected_model_id": selected_id,
                    "selected_oof_probability": row.probability,
                    "surrogate_full_fit_probability": base_prediction,
                    "atom_index": atom_index,
                    "atomic_symbol": atom.GetSymbol(),
                    "atom_environment_bits_masked": len(bits),
                    "surrogate_atom_environment_contribution": weight,
                }
            )
        drawer = rdMolDraw2D.MolDraw2DCairo(700, 520)
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol, weights, drawer, colorMap="coolwarm", alpha=0.45, contourLines=8
        )
        drawer.FinishDrawing()
        image_path = map_dir / f"rank_{rank:02d}_{row.inchikey}.png"
        image_path.write_bytes(drawer.GetDrawingText())
        image_paths.append(image_path)
    atom_data = pd.DataFrame(atom_rows)

    fig, axes = plt.subplots(3, 4, figsize=(14.0, 10.8), constrained_layout=True)
    for ax, path, row in zip(axes.ravel(), image_paths, representatives.itertuples(index=False)):
        ax.imshow(Image.open(path))
        ax.axis("off")
        ax.set_title(
            f"P={row.probability:.3f}; label=active\n"
            f"{row.scaffold_family_completed}",
            fontsize=8.0,
        )
    fig.suptitle(
        "High-scoring measured actives: Morgan-surrogate atom-environment maps",
        fontsize=14, fontweight="bold"
    )
    fig.text(
        0.5, 0.002,
        "Red/blue indicates positive/negative change in the selected OOF probability surrogate after atom-environment masking; these are not direct neural gradients.",
        ha="center", fontsize=8.2, color=GREY
    )
    save(fig, "F11_local_surrogate_attribution")
    return fidelity, representatives, atom_data


def disagreement_analysis(
    selected: pd.DataFrame, all_predictions: pd.DataFrame, selected_id: str
) -> tuple[pd.DataFrame, dict[str, float]]:
    wide = all_predictions.pivot(
        index="sample_index", columns="model_id", values="probability"
    )
    data = selected[
        ["sample_index", "inchikey", "canonical_smiles", "scaffold_group", "active_label", "probability"]
    ].merge(
        wide.std(axis=1).rename("between_model_probability_sd"),
        left_on="sample_index", right_index=True, validate="one_to_one"
    )
    data["selected_absolute_error"] = np.abs(
        data["active_label"] - data["probability"]
    )
    data["selected_misclassified"] = (
        (data["probability"] >= 0.5).astype(int) != data["active_label"]
    ).astype(int)
    rho, p_value = spearmanr(
        data["between_model_probability_sd"], data["selected_absolute_error"]
    )
    data["disagreement_quintile"] = pd.qcut(
        data["between_model_probability_sd"], 5,
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        duplicates="drop",
    )
    bins = (
        data.groupby("disagreement_quintile", observed=True)
        .agg(
            molecule_count=("inchikey", "size"),
            mean_disagreement=("between_model_probability_sd", "mean"),
            mean_absolute_error=("selected_absolute_error", "mean"),
            misclassification_rate=("selected_misclassified", "mean"),
        )
        .reset_index()
    )
    summary = {
        "selected_model_id": selected_id,
        "spearman_rho_disagreement_vs_absolute_error": float(rho),
        "spearman_p_value_naive_descriptive": float(p_value),
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), constrained_layout=True)
    axes[0].scatter(
        data["between_model_probability_sd"],
        data["selected_absolute_error"],
        c=np.where(data["selected_misclassified"].eq(1), ORANGE, BLUE),
        s=14, alpha=0.48
    )
    axes[0].set(
        title=f"A  Model disagreement and error (Spearman ρ={rho:.2f})",
        xlabel="SD of probabilities across model families",
        ylabel="Selected-model absolute error",
    )
    x = np.arange(len(bins))
    axes[1].bar(
        x - 0.17, bins["mean_absolute_error"], width=0.34,
        color=BLUE, label="Mean absolute error"
    )
    axes[1].bar(
        x + 0.17, bins["misclassification_rate"], width=0.34,
        color=ORANGE, label="Misclassification rate"
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bins["disagreement_quintile"])
    axes[1].set(
        title="B  Error by disagreement quintile",
        xlabel="Between-model disagreement",
        ylabel="Rate",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Prediction disagreement as an uncertainty diagnostic",
        fontsize=14, fontweight="bold"
    )
    save(fig, "F12_disagreement_error")
    bins.to_csv(TABLE_DIR / "F12_disagreement_bins_data.csv", index=False)
    return data, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="outputs/rebuild_current_20260727/downstream_oof",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    selected_id, selected, all_predictions, seed_averaged = load_inputs(run_dir)
    motifs, presence = motif_analysis(
        selected, seed_averaged, selected_id, args.bootstrap_repetitions
    )
    scaffolds = scaffold_analysis(selected)
    motifs.to_csv(TABLE_DIR / "F10_motif_effects_data.csv", index=False)
    presence.to_csv(TABLE_DIR / "F10_motif_presence_matrix.csv", index=False)
    scaffolds.to_csv(TABLE_DIR / "F10_scaffold_prediction_data.csv", index=False)
    draw_global_interpretability(motifs, scaffolds, selected_id)
    fidelity, representatives, atom_data = surrogate_and_local_maps(
        selected, selected_id
    )
    fidelity.to_csv(TABLE_DIR / "F11_surrogate_fidelity_data.csv", index=False)
    representatives.to_csv(TABLE_DIR / "F11_representative_molecules_data.csv", index=False)
    atom_data.to_csv(TABLE_DIR / "F11_atom_contribution_data.csv", index=False)
    disagreement, disagreement_summary = disagreement_analysis(
        selected, all_predictions, selected_id
    )
    disagreement.to_csv(TABLE_DIR / "F12_disagreement_molecule_data.csv", index=False)
    stable = motifs.loc[motifs["cross_scaffold_stable"].eq(1)]
    manifest = {
        "status": "INTERPRETABILITY_COMPLETE",
        "selected_model_id": selected_id,
        "motifs_tested_after_prevalence_filter": len(motifs),
        "cross_scaffold_stable_motifs": stable["motif"].tolist(),
        "surrogate_pooled_r2": float(
            fidelity.loc[fidelity["fold"].eq("pooled"), "r2"].iloc[0]
        ),
        "surrogate_pooled_mae": float(
            fidelity.loc[fidelity["fold"].eq("pooled"), "mae"].iloc[0]
        ),
        "local_map_scope": (
            "Morgan fingerprint random-forest surrogate of selected scaffold-OOF "
            "probabilities; atom maps are not direct neural gradients"
        ),
        **disagreement_summary,
    }
    (TABLE_DIR / "interpretability_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
