#!/usr/bin/env python3
"""Generate F10-F12 from strict development OOF and locked validation predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "provenance"))
import run_current_interpretability as common  # noqa: E402


OUT = ROOT / "outputs" / "rebuild_current_20260727"
RUN = OUT / "downstream_locked_validation"
VALIDATION = RUN / "locked_validation_evaluation"
CHEMICAL = OUT / "tables" / "chemical_space" / "chemical_space_dataset_current955.csv"
FIGURE_DIR = OUT / "figures" / "interpretability"
TABLE_DIR = OUT / "tables" / "interpretability"

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GREY = "#6B7280"
GRID = "#D1D5DB"
RD_FONT_FILE = "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"


def configure_molecule_drawer(drawer: rdMolDraw2D.MolDraw2DCairo) -> None:
    """Match embedded structure labels and bonds to the manuscript figure style."""
    options = drawer.drawOptions()
    options.fontFile = RD_FONT_FILE
    options.bondLineWidth = 1.0
    options.clearBackground = False


def load_inputs() -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    completion = json.loads((VALIDATION / "VALIDATION_COMPLETION.json").read_text())
    if completion.get("status") != "LOCKED_VALIDATION_COMPLETE":
        raise RuntimeError("Locked validation is incomplete")
    selected_lock = json.loads((RUN / "SELECTED_MODEL_LOCK.json").read_text())
    selected_id = str(selected_lock["selected_model_id"])
    if selected_id != "T3":
        raise RuntimeError(f"Unexpected selected model: {selected_id}")

    chemical = pd.read_csv(CHEMICAL)
    metadata = chemical[
        [
            "inchikey",
            "compound_name_or_label",
            "scaffold_family_completed",
            "scaffold_family_source",
            "label_source",
            "audit_pic50",
            "exact_ic50_uM_for_plot",
        ]
    ]
    ensemble = pd.read_csv(RUN / "final_ensemble_oof_predictions.csv")
    development = ensemble.loc[ensemble["model_id"].eq(selected_id)].copy()
    if len(development) != 765:
        raise RuntimeError("Selected development OOF does not cover 765 molecules")
    development = development.merge(
        metadata, on="inchikey", how="left", validate="one_to_one"
    )
    if development["scaffold_family_completed"].isna().any():
        raise RuntimeError("Development metadata alignment failed")
    seed_averaged = pd.read_csv(RUN / "seed_averaged_oof_predictions.csv")

    validation = pd.read_csv(VALIDATION / "validation_predictions.csv")
    if len(validation) != 190:
        raise RuntimeError("Locked validation predictions do not cover 190 molecules")
    validation = validation.merge(
        metadata, on="inchikey", how="left", validate="one_to_one"
    )
    if validation["scaffold_family_completed"].isna().any():
        raise RuntimeError("Validation metadata alignment failed")
    return selected_id, development, seed_averaged, validation


def add_validation_motif_confirmation(
    motifs: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = validation.copy()
    mols = [Chem.MolFromSmiles(value) for value in frame["canonical_smiles"]]
    for name, smarts in common.MOTIFS.items():
        query = Chem.MolFromSmarts(smarts)
        frame[name] = [int(mol.HasSubstructMatch(query)) for mol in mols]

    rows = []
    for row in motifs.itertuples(index=False):
        present = frame[row.motif].eq(1)
        absent = ~present
        if present.any() and absent.any():
            predicted = float(frame.loc[present, "BEST"].mean() - frame.loc[absent, "BEST"].mean())
            observed = float(
                frame.loc[present, "active_label"].mean()
                - frame.loc[absent, "active_label"].mean()
            )
        else:
            predicted = np.nan
            observed = np.nan
        rows.append(
            {
                "motif": row.motif,
                "validation_molecule_count": int(present.sum()),
                "validation_scaffold_count": int(
                    frame.loc[present, "scaffold_group"].nunique()
                ),
                "validation_predicted_probability_difference": predicted,
                "validation_observed_active_fraction_difference": observed,
                "development_validation_predicted_direction_consistent": int(
                    np.isfinite(predicted)
                    and np.sign(predicted)
                    == np.sign(row.predicted_probability_difference)
                ),
                "development_validation_observed_direction_consistent": int(
                    np.isfinite(observed)
                    and np.sign(observed)
                    == np.sign(row.observed_active_fraction_difference)
                ),
            }
        )
    confirmation = pd.DataFrame(rows)
    result = motifs.merge(confirmation, on="motif", validate="one_to_one")
    presence = frame[
        [
            "sample_index",
            "inchikey",
            "scaffold_group",
            "active_label",
            "BEST",
            "scaffold_family_completed",
        ]
        + list(common.MOTIFS)
    ].copy()
    return result, presence


def render_plain_molecule(mol: Chem.Mol, path: Path) -> None:
    drawer = rdMolDraw2D.MolDraw2DCairo(700, 520)
    configure_molecule_drawer(drawer)
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    path.write_bytes(drawer.GetDrawingText())


def surrogate_and_validation_structures(
    development: pd.DataFrame,
    validation: pd.DataFrame,
    selected_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dev = development.sort_values("sample_index").reset_index(drop=True).copy()
    dev_mols = [Chem.MolFromSmiles(value) for value in dev["canonical_smiles"]]
    matrix = common.morgan_matrix(dev_mols)
    target = dev["probability"].to_numpy(float)
    groups = dev["scaffold_group"].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=5)
    oof = np.zeros(len(dev), dtype=float)
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
                "train_count": int(len(train)),
                "validation_count": int(len(valid)),
                "r2": float(r2_score(target[valid], oof[valid])),
                "mae": float(mean_absolute_error(target[valid], oof[valid])),
            }
        )
    pooled_r2 = float(r2_score(target, oof))
    pooled_mae = float(mean_absolute_error(target, oof))
    fidelity_passed = pooled_r2 >= 0.50
    fidelity = pd.DataFrame(
        fold_rows
        + [
            {
                "fold": "pooled",
                "train_count": int(len(dev)),
                "validation_count": int(len(dev)),
                "r2": pooled_r2,
                "mae": pooled_mae,
                "fidelity_threshold": 0.50,
                "fidelity_passed": int(fidelity_passed),
            }
        ]
    )
    fidelity["fidelity_threshold"] = fidelity["fidelity_threshold"].fillna(0.50)
    fidelity["fidelity_passed"] = fidelity["fidelity_passed"].fillna(
        int(fidelity_passed)
    )

    representatives = (
        validation.loc[validation["active_label"].eq(1)]
        .sort_values("BEST", ascending=False)
        .drop_duplicates("scaffold_group")
        .head(12)
        .copy()
    )
    representatives["rank"] = np.arange(1, len(representatives) + 1)
    representatives["local_surrogate_fidelity_passed"] = int(fidelity_passed)
    map_dir = FIGURE_DIR / "local_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=True
    )
    atom_rows: list[dict] = []
    image_paths: list[Path] = []

    full_model: RandomForestRegressor | None = None
    if fidelity_passed:
        full_model = RandomForestRegressor(
            n_estimators=1000,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=20260727,
            n_jobs=-1,
        )
        full_model.fit(matrix, target)

    for row in representatives.itertuples(index=False):
        mol = Chem.MolFromSmiles(row.canonical_smiles)
        image_path = map_dir / f"rank_{int(row.rank):02d}_{row.inchikey}.png"
        if not fidelity_passed or full_model is None:
            render_plain_molecule(mol, image_path)
            image_paths.append(image_path)
            continue

        additional = rdFingerprintGenerator.AdditionalOutput()
        additional.AllocateBitInfoMap()
        fp = generator.GetFingerprint(mol, additionalOutput=additional)
        base = np.zeros(2048, dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fp, base)
        base_prediction = float(full_model.predict(base.reshape(1, -1))[0])
        affected = common.atom_environment_bits(mol, additional.GetBitInfoMap())
        weights = []
        for atom_index, bits in enumerate(affected):
            masked = base.copy()
            if bits:
                masked[list(bits)] = 0
            masked_prediction = float(full_model.predict(masked.reshape(1, -1))[0])
            weight = base_prediction - masked_prediction
            weights.append(weight)
            atom_rows.append(
                {
                    "rank": int(row.rank),
                    "inchikey": row.inchikey,
                    "selected_model_id": selected_id,
                    "locked_validation_probability": float(row.BEST),
                    "surrogate_full_development_fit_probability": base_prediction,
                    "atom_index": atom_index,
                    "atomic_symbol": mol.GetAtomWithIdx(atom_index).GetSymbol(),
                    "atom_environment_bits_masked": len(bits),
                    "surrogate_atom_environment_contribution": weight,
                }
            )
        drawer = rdMolDraw2D.MolDraw2DCairo(700, 520)
        configure_molecule_drawer(drawer)
        SimilarityMaps.GetSimilarityMapFromWeights(
            mol,
            weights,
            drawer,
            colorMap="coolwarm",
            alpha=0.45,
            contourLines=8,
        )
        drawer.FinishDrawing()
        image_path.write_bytes(drawer.GetDrawingText())
        image_paths.append(image_path)

    atom_columns = [
        "rank",
        "inchikey",
        "selected_model_id",
        "locked_validation_probability",
        "surrogate_full_development_fit_probability",
        "atom_index",
        "atomic_symbol",
        "atom_environment_bits_masked",
        "surrogate_atom_environment_contribution",
    ]
    atom_data = pd.DataFrame(atom_rows, columns=atom_columns)

    fig, axes = plt.subplots(3, 4, figsize=(14.0, 10.8), constrained_layout=True)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, path, row in zip(
        axes.ravel(), image_paths, representatives.itertuples(index=False)
    ):
        ax.imshow(Image.open(path))
        ax.set_title(
            f"#{int(row.rank)}  P={row.BEST:.3f}\n"
            f"{str(row.scaffold_family_completed)[:46]}",
            fontsize=7.8,
        )
    title = (
        "Locked-validation high-scoring measured actives: faithful local surrogate maps"
        if fidelity_passed
        else "Locked-validation high-scoring measured actives (local surrogate fidelity gate failed)"
    )
    fig.suptitle(title, fontsize=13.5, fontweight="bold")
    note = (
        f"Morgan surrogate scaffold-OOF R²={pooled_r2:.3f} (threshold 0.50); "
        + (
            "colors show proxy probability changes after atom-environment masking, not causal effects."
            if fidelity_passed
            else "plain structures are shown and no atom attribution is interpreted."
        )
    )
    fig.text(0.5, 0.002, note, ha="center", fontsize=8.2, color=GREY)
    common.save(fig, "F11_local_surrogate_attribution")
    return fidelity, representatives, atom_data


def disagreement_and_applicability(
    validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, float]]:
    similarity = pd.read_csv(
        OUT / "tables" / "validation" / "ValidationSimilarityStrata.csv"
    )[["inchikey", "nearest_development_morgan_tanimoto", "similarity_stratum"]]
    frame = validation.merge(similarity, on="inchikey", validate="one_to_one")
    model_columns = ["A0", "A1", "A2", "A3", "T1", "T3"]
    frame["a3_t1_absolute_disagreement"] = np.abs(frame["A3"] - frame["T1"])
    frame["between_model_probability_sd"] = frame[model_columns].std(axis=1)
    frame["best_absolute_error"] = np.abs(frame["active_label"] - frame["BEST"])
    frame["f0_absolute_error"] = np.abs(frame["active_label"] - frame["F0"])
    frame["best_misclassified"] = (
        (frame["BEST"] >= 0.5).astype(int) != frame["active_label"]
    ).astype(int)
    frame["f0_misclassified"] = (
        (frame["F0"] >= 0.5).astype(int) != frame["active_label"]
    ).astype(int)
    frame["high_confidence_false_positive"] = (
        frame["active_label"].eq(0) & frame["BEST"].ge(0.80)
    ).astype(int)
    frame["high_confidence_false_negative"] = (
        frame["active_label"].eq(1) & frame["BEST"].le(0.20)
    ).astype(int)
    rho, p_value = spearmanr(
        frame["a3_t1_absolute_disagreement"], frame["best_absolute_error"]
    )
    frame["disagreement_quintile"] = pd.qcut(
        frame["a3_t1_absolute_disagreement"],
        5,
        labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"],
        duplicates="drop",
    )
    bins = (
        frame.groupby("disagreement_quintile", observed=True)
        .agg(
            molecule_count=("inchikey", "size"),
            mean_a3_t1_disagreement=("a3_t1_absolute_disagreement", "mean"),
            best_mean_absolute_error=("best_absolute_error", "mean"),
            f0_mean_absolute_error=("f0_absolute_error", "mean"),
            best_misclassification_rate=("best_misclassified", "mean"),
            f0_misclassification_rate=("f0_misclassified", "mean"),
        )
        .reset_index()
    )
    applicability = (
        frame.groupby("similarity_stratum", as_index=False)
        .agg(
            molecule_count=("inchikey", "size"),
            scaffold_count=("scaffold_group", "nunique"),
            similarity_min=("nearest_development_morgan_tanimoto", "min"),
            similarity_median=("nearest_development_morgan_tanimoto", "median"),
            similarity_max=("nearest_development_morgan_tanimoto", "max"),
            best_mean_absolute_error=("best_absolute_error", "mean"),
            f0_mean_absolute_error=("f0_absolute_error", "mean"),
            best_misclassification_rate=("best_misclassified", "mean"),
            f0_misclassification_rate=("f0_misclassified", "mean"),
        )
    )
    errors = frame.loc[
        frame["high_confidence_false_positive"].eq(1)
        | frame["high_confidence_false_negative"].eq(1)
    ].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9), constrained_layout=True)
    axes[0].scatter(
        frame["a3_t1_absolute_disagreement"],
        frame["best_absolute_error"],
        c=np.where(frame["best_misclassified"].eq(1), ORANGE, BLUE),
        s=22,
        alpha=0.58,
    )
    axes[0].set(
        title=f"A  A3-T1 disagreement vs BEST error (rho={rho:.2f})",
        xlabel="|P(A3) - P(T1)|",
        ylabel="BEST absolute error",
    )
    x = np.arange(len(bins))
    axes[1].bar(
        x - 0.18,
        bins["best_misclassification_rate"],
        width=0.36,
        color=ORANGE,
        label="BEST",
    )
    axes[1].bar(
        x + 0.18,
        bins["f0_misclassification_rate"],
        width=0.36,
        color=PURPLE,
        label="F0",
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(bins["disagreement_quintile"], rotation=24, ha="right")
    axes[1].set(
        title="B  Misclassification by disagreement quintile",
        ylabel="Misclassification rate",
    )
    axes[1].legend(fontsize=8)

    x = np.arange(len(applicability))
    axes[2].bar(
        x - 0.18,
        applicability["best_misclassification_rate"],
        width=0.36,
        color=ORANGE,
        label="BEST",
    )
    axes[2].bar(
        x + 0.18,
        applicability["f0_misclassification_rate"],
        width=0.36,
        color=PURPLE,
        label="F0",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(
        applicability["similarity_stratum"], rotation=20, ha="right", fontsize=8
    )
    axes[2].set(
        title="C  Error by nearest-development similarity",
        ylabel="Misclassification rate",
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Locked-validation disagreement, errors, and applicability domain",
        fontsize=14,
        fontweight="bold",
    )
    common.save(fig, "F12_disagreement_error")
    summary = {
        "spearman_rho_a3_t1_disagreement_vs_best_absolute_error": float(rho),
        "spearman_p_value_descriptive": float(p_value),
        "high_confidence_false_positive_count": int(
            frame["high_confidence_false_positive"].sum()
        ),
        "high_confidence_false_negative_count": int(
            frame["high_confidence_false_negative"].sum()
        ),
    }
    return frame, bins, applicability, errors, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    common.configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    selected_id, development, seed_averaged, validation = load_inputs()

    motifs, development_presence = common.motif_analysis(
        development, seed_averaged, selected_id, args.bootstrap_repetitions
    )
    motifs, validation_presence = add_validation_motif_confirmation(
        motifs, validation
    )
    scaffolds = common.scaffold_analysis(development)
    motifs.to_csv(TABLE_DIR / "F10_motif_effects_data.csv", index=False)
    development_presence.to_csv(
        TABLE_DIR / "F10_motif_presence_development.csv", index=False
    )
    validation_presence.to_csv(
        TABLE_DIR / "F10_motif_presence_validation.csv", index=False
    )
    scaffolds.to_csv(TABLE_DIR / "F10_scaffold_prediction_data.csv", index=False)
    common.draw_global_interpretability(motifs, scaffolds, selected_id)

    fidelity, representatives, atom_data = surrogate_and_validation_structures(
        development, validation, selected_id
    )
    fidelity.to_csv(TABLE_DIR / "F11_surrogate_fidelity_data.csv", index=False)
    representatives.to_csv(
        TABLE_DIR / "F11_representative_molecules_data.csv", index=False
    )
    atom_data.to_csv(TABLE_DIR / "F11_atom_contribution_data.csv", index=False)

    disagreement, bins, applicability, errors, disagreement_summary = (
        disagreement_and_applicability(validation)
    )
    disagreement.to_csv(
        TABLE_DIR / "F12_disagreement_molecule_data.csv", index=False
    )
    bins.to_csv(TABLE_DIR / "F12_disagreement_bins_data.csv", index=False)
    applicability.to_csv(
        TABLE_DIR / "F12_similarity_error_strata_data.csv", index=False
    )
    errors.to_csv(
        TABLE_DIR / "F12_high_confidence_errors_data.csv", index=False
    )

    pooled = fidelity.loc[fidelity["fold"].astype(str).eq("pooled")].iloc[0]
    stable = motifs.loc[motifs["cross_scaffold_stable"].eq(1)]
    manifest = {
        "status": "LOCKED_INTERPRETABILITY_COMPLETE",
        "selected_model_id": selected_id,
        "development_oof_count": int(len(development)),
        "locked_validation_count": int(len(validation)),
        "motifs_tested_after_prevalence_filter": int(len(motifs)),
        "development_cross_scaffold_stable_motifs": stable["motif"].tolist(),
        "stable_motif_validation_predicted_direction_consistency": (
            float(
                stable["development_validation_predicted_direction_consistent"].mean()
            )
            if len(stable)
            else None
        ),
        "surrogate_pooled_scaffold_oof_r2": float(pooled["r2"]),
        "surrogate_pooled_scaffold_oof_mae": float(pooled["mae"]),
        "local_attribution_fidelity_threshold": 0.50,
        "local_attribution_permitted": bool(pooled["r2"] >= 0.50),
        **disagreement_summary,
    }
    (TABLE_DIR / "interpretability_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
