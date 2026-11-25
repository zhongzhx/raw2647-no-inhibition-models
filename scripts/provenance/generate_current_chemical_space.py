#!/usr/bin/env python3
"""Generate reproducible chemical-space figures for the current 955-molecule set.

Existing scaffold-family annotations are retained. Missing annotations are
completed in an auditable hierarchy: identical annotated Murcko scaffold,
high-similarity annotated neighbour, then deterministic broad structural rules.
The completed family field is used for descriptive analysis and, in the
version-2 downstream protocol, as an explicitly audited label-independent
structure-metadata branch.
"""

from __future__ import annotations

import json
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = (
    ROOT
    / "outputs"
    / "rebuild_current_20260727"
    / "data_audit"
    / "final_modeling_manifest.csv"
)
OUT = ROOT / "outputs" / "rebuild_current_20260727"
FIGURE_DIR = OUT / "figures" / "chemical_space"
TABLE_DIR = OUT / "tables" / "chemical_space"

# Okabe-Ito colour-blind-safe palette.
ACTIVE = "#D55E00"
INACTIVE = "#0072B2"
ACCENT = "#009E73"
SKY = "#56B4E9"
PURPLE = "#CC79A7"
GREY = "#6B7280"
GRID = "#D1D5DB"


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


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_DIR / f"{stem}.png", dpi=300, facecolor="white")
    fig.savefig(FIGURE_DIR / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def murcko_smiles(mol: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold.GetNumAtoms() == 0:
        return "[ACYCLIC]"
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def has(mol: Chem.Mol, smarts: str) -> bool:
    query = Chem.MolFromSmarts(smarts)
    return bool(query is not None and mol.HasSubstructMatch(query))


def broad_structural_family(mol: Chem.Mol) -> str:
    """Assign a conservative broad class without implying biosynthetic origin."""
    rings = int(Lipinski.RingCount(mol))
    aromatic = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    n_atoms = sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms())
    sugar = has(mol, "[O;R]1[CH;R][CH;R][CH;R][CH;R]1")
    if rings == 0:
        return "acyclic compound"
    if has(mol, "O=c1oc2ccccc2cc1") or has(mol, "O=c1cc(-c2ccccc2)oc2ccccc12"):
        return "flavonoid-like scaffold"
    if has(mol, "O=C(O)c1ccccc1"):
        return "benzoic-acid derivative"
    if has(mol, "c1ccc2occc2c1"):
        return "benzofuran-like scaffold"
    if has(mol, "c1ccc2[nH]ccc2c1"):
        return "indole-like alkaloid"
    if has(mol, "c1ccc2ncccc2c1"):
        return "quinoline-like alkaloid"
    if has(mol, "c1ccccc1C=CC(=O)"):
        return "phenylpropanoid/chalcone-like scaffold"
    if sugar:
        return "glycoside-like scaffold"
    if n_atoms >= 1 and aromatic >= 1:
        return "aromatic nitrogen heterocycle"
    if n_atoms >= 1 and aromatic == 0:
        return "aliphatic nitrogen heterocycle"
    if rings >= 4 and aromatic <= 1:
        return "polycyclic terpenoid/steroid-like scaffold"
    if rings >= 2 and aromatic == 0:
        return "alicyclic terpenoid-like scaffold"
    if aromatic >= 3:
        return "polycyclic aromatic scaffold"
    if aromatic == 2:
        return "biaryl/fused aromatic scaffold"
    if aromatic == 1:
        return "monocyclic aromatic scaffold"
    return "other oxygenated cyclic scaffold"


def complete_scaffold_families(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["_mol"] = [Chem.MolFromSmiles(s) for s in result["canonical_smiles"]]
    if any(mol is None for mol in result["_mol"]):
        raise ValueError("Final modelling manifest contains an invalid canonical SMILES")
    result["murcko_scaffold"] = [murcko_smiles(mol) for mol in result["_mol"]]
    raw_family = result["scaffold_family"].astype("string").str.strip()
    raw_family = raw_family.mask(raw_family.eq(""))
    result["scaffold_family_original"] = raw_family
    result["scaffold_family_completed"] = raw_family
    result["scaffold_family_source"] = np.where(
        raw_family.notna(), "existing_annotation", ""
    )
    result["family_assignment_similarity"] = np.where(raw_family.notna(), 1.0, np.nan)
    result["family_assignment_reference_inchikey"] = np.where(
        raw_family.notna(), result["inchikey"], ""
    )

    # First propagate a majority annotation only within an identical Murcko core.
    annotated = result.loc[result["scaffold_family_completed"].notna()]
    scaffold_to_family: dict[str, str] = {}
    for scaffold, group in annotated.groupby("murcko_scaffold"):
        if scaffold == "[ACYCLIC]":
            continue
        counts = Counter(group["scaffold_family_completed"].astype(str))
        scaffold_to_family[scaffold] = sorted(
            counts, key=lambda value: (-counts[value], value)
        )[0]
    for idx in result.index[result["scaffold_family_completed"].isna()]:
        scaffold = result.at[idx, "murcko_scaffold"]
        if scaffold in scaffold_to_family:
            family = scaffold_to_family[scaffold]
            ref = annotated.loc[
                (annotated["murcko_scaffold"].eq(scaffold))
                & (annotated["scaffold_family_completed"].eq(family))
            ].iloc[0]
            result.at[idx, "scaffold_family_completed"] = family
            result.at[idx, "scaffold_family_source"] = "identical_murcko_scaffold"
            result.at[idx, "family_assignment_similarity"] = 1.0
            result.at[idx, "family_assignment_reference_inchikey"] = ref["inchikey"]

    # Then use only a high-similarity annotated neighbour; leave ambiguous cases
    # to broad deterministic rules.
    annotated_idx = list(result.index[result["scaffold_family_original"].notna()])
    annotated_fps = [
        AllChem.GetMorganFingerprintAsBitVect(
            result.at[idx, "_mol"], radius=2, nBits=2048, useChirality=True
        )
        for idx in annotated_idx
    ]
    for idx in result.index[result["scaffold_family_completed"].isna()]:
        fp = AllChem.GetMorganFingerprintAsBitVect(
            result.at[idx, "_mol"], radius=2, nBits=2048, useChirality=True
        )
        similarities = DataStructs.BulkTanimotoSimilarity(fp, annotated_fps)
        best_pos = int(np.argmax(similarities))
        best_similarity = float(similarities[best_pos])
        if best_similarity >= 0.72:
            ref_idx = annotated_idx[best_pos]
            result.at[idx, "scaffold_family_completed"] = result.at[
                ref_idx, "scaffold_family_original"
            ]
            result.at[idx, "scaffold_family_source"] = "annotated_neighbour"
            result.at[idx, "family_assignment_similarity"] = best_similarity
            result.at[idx, "family_assignment_reference_inchikey"] = result.at[
                ref_idx, "inchikey"
            ]

    for idx in result.index[result["scaffold_family_completed"].isna()]:
        result.at[idx, "scaffold_family_completed"] = broad_structural_family(
            result.at[idx, "_mol"]
        )
        result.at[idx, "scaffold_family_source"] = "deterministic_structural_rule"
        result.at[idx, "family_assignment_reference_inchikey"] = ""

    if result["scaffold_family_completed"].isna().any():
        raise RuntimeError("Scaffold-family completion unexpectedly left missing values")
    return result


def molecule_properties(mol: Chem.Mol) -> dict[str, float | int]:
    mw = float(Descriptors.MolWt(mol))
    logp = float(Descriptors.MolLogP(mol))
    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    return {
        "MolWt": mw,
        "MolLogP": logp,
        "TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
        "NumHDonors": hbd,
        "NumHAcceptors": hba,
        "NumRotatableBonds": int(Lipinski.NumRotatableBonds(mol)),
        "RingCount": int(Lipinski.RingCount(mol)),
        "NumAromaticRings": int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
        "HeavyAtomCount": int(mol.GetNumHeavyAtoms()),
        "LipinskiViolations": (
            int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
        ),
    }


def load_dataset() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST)
    if len(frame) != 955:
        raise ValueError(f"Expected current 955-molecule set, found {len(frame)}")
    enriched = complete_scaffold_families(frame)
    properties = pd.DataFrame(
        [molecule_properties(mol) for mol in enriched["_mol"]], index=enriched.index
    )
    enriched = pd.concat([enriched.drop(columns="_mol"), properties], axis=1)
    enriched["activity_class"] = np.where(
        enriched["audit_active_label"].eq(1), "Active", "Inactive"
    )
    enriched["label_source"] = np.select(
        [
            enriched["audit_has_exact_pic50"].eq(1),
            enriched["ic50_raw_uM"].astype(str).str.strip().str.upper().isin(["NS", "ND"]),
            enriched["audit_censor_type"].eq("right"),
        ],
        ["Exact IC50", "NS", "Right-censored"],
        default="Other classified",
    )
    # Six exact records are reported as mean ± SD strings. Their audited pIC50
    # values are exact point estimates even though the raw numeric helper column
    # is empty, so reconstruct the point estimate solely for visualisation.
    enriched["exact_ic50_uM_for_plot"] = enriched["ic50_uM_numeric"]
    reconstruct = (
        enriched["audit_has_exact_pic50"].eq(1)
        & enriched["exact_ic50_uM_for_plot"].isna()
        & enriched["audit_pic50"].notna()
    )
    enriched.loc[reconstruct, "exact_ic50_uM_for_plot"] = 10 ** (
        6.0 - enriched.loc[reconstruct, "audit_pic50"]
    )
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(TABLE_DIR / "chemical_space_dataset_current955.csv", index=False)
    enriched[
        [
            "inchikey",
            "canonical_smiles",
            "murcko_scaffold",
            "scaffold_family_original",
            "scaffold_family_completed",
            "scaffold_family_source",
            "family_assignment_similarity",
            "family_assignment_reference_inchikey",
        ]
    ].to_csv(TABLE_DIR / "scaffold_family_completion_audit.csv", index=False)
    return enriched


def family_counts(enriched: pd.DataFrame) -> pd.DataFrame:
    counts = (
        enriched.groupby(
            ["scaffold_family_completed", "activity_class"], observed=True
        )
        .size()
        .unstack(fill_value=0)
    )
    for column in ("Active", "Inactive"):
        if column not in counts:
            counts[column] = 0
    counts["Total"] = counts["Active"] + counts["Inactive"]
    counts["ActiveFraction"] = counts["Active"] / counts["Total"]
    return counts.sort_values(["Total", "Active"], ascending=False)


def draw_f1(enriched: pd.DataFrame) -> pd.DataFrame:
    counts = family_counts(enriched)
    ranked = counts.reset_index()
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["cumulative_fraction"] = ranked["Total"].cumsum() / len(enriched)
    ranked.to_csv(TABLE_DIR / "F1_scaffold_family_distribution_data.csv", index=False)
    top = counts.head(20).sort_values("Total")
    fig, axes = plt.subplots(
        1, 2, figsize=(13.6, 7.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.55, 1.0]}
    )
    y = np.arange(len(top))
    axes[0].barh(y, top["Inactive"], color=INACTIVE, label="Inactive")
    axes[0].barh(y, top["Active"], left=top["Inactive"], color=ACTIVE, label="Active")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(
        [textwrap.fill(str(v), 34) for v in top.index], fontsize=7.8
    )
    axes[0].set_xlabel("Number of molecules")
    axes[0].set_title("A  Top 20 scaffold families")
    axes[0].xaxis.grid(True, color=GRID, linewidth=0.5)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="lower right")
    for yi, total in zip(y, top["Total"]):
        axes[0].text(total + 0.5, yi, str(int(total)), va="center", fontsize=7.3)

    axes[1].plot(
        ranked["rank"], ranked["cumulative_fraction"] * 100,
        color=ACCENT, linewidth=2.2
    )
    coverage = float(ranked.iloc[min(19, len(ranked) - 1)]["cumulative_fraction"])
    axes[1].axvline(20, color=GREY, linestyle="--", linewidth=1)
    axes[1].scatter([20], [coverage * 100], color=ACCENT, s=42, zorder=3)
    axes[1].annotate(
        f"Top 20 coverage: {coverage:.1%}",
        xy=(20, coverage * 100),
        xytext=(max(25, len(ranked) * 0.18), max(10, coverage * 100 - 18)),
        arrowprops={"arrowstyle": "-", "color": GREY},
    )
    axes[1].set(xlabel="Scaffold-family rank", ylabel="Cumulative coverage (%)")
    axes[1].set_ylim(0, 102)
    axes[1].grid(True, color=GRID, linewidth=0.5)
    axes[1].set_title("B  Long-tail coverage")
    fig.suptitle(
        f"Scaffold-family distribution (n={len(enriched)}; families={len(counts)})",
        fontsize=13, fontweight="bold"
    )
    save_figure(fig, "F1_scaffold_family_distribution")
    return counts


def draw_f2(enriched: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    groups = (
        enriched.groupby("murcko_scaffold")
        .agg(
            count=("inchikey", "size"),
            active_count=("audit_active_label", "sum"),
            representative_inchikey=("inchikey", "first"),
            representative_smiles=("canonical_smiles", "first"),
            dominant_family=(
                "scaffold_family_completed",
                lambda values: values.value_counts().index[0],
            ),
        )
        .sort_values(["count", "active_count"], ascending=False)
    )
    # Acyclic compounds do not share a drawable Murcko core, so they do not
    # constitute one chemically meaningful structure panel.
    groups = groups.loc[groups.index != "[ACYCLIC]"].head(20)
    for rank, (scaffold, row) in enumerate(groups.iterrows(), 1):
        rows.append(
            {
                "rank": rank,
                "murcko_scaffold": scaffold,
                "count": int(row["count"]),
                "active_count": int(row["active_count"]),
                "inactive_count": int(row["count"] - row["active_count"]),
                "active_fraction": float(row["active_count"] / row["count"]),
                "dominant_family": row["dominant_family"],
                "representative_inchikey": row["representative_inchikey"],
                "representative_smiles": row["representative_smiles"],
            }
        )
    top = pd.DataFrame(rows)
    top.to_csv(TABLE_DIR / "F2_top20_murcko_scaffolds_data.csv", index=False)
    fig, axes = plt.subplots(5, 4, figsize=(13.2, 15.6), constrained_layout=True)
    for axis, row in zip(axes.ravel(), top.itertuples(index=False)):
        mol = Chem.MolFromSmiles(row.murcko_scaffold)
        image = Draw.MolToImage(mol, size=(700, 470), kekulize=True)
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(
            f"{row.rank}. n={row.count}; active={row.active_count}\n"
            f"{textwrap.fill(str(row.dominant_family), 30)}",
            fontsize=8.0, pad=3
        )
    fig.suptitle(
        "Twenty most frequent Bemis–Murcko scaffolds",
        fontsize=14, fontweight="bold"
    )
    fig.text(
        0.5, 0.002,
        "Ranking is based on exact scaffold frequency in the current 955-molecule set.",
        ha="center", fontsize=9, color=GREY
    )
    save_figure(fig, "F2_top20_murcko_scaffolds")
    return top


def overlaid_hist(
    ax: plt.Axes,
    data: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    bins: int | np.ndarray,
) -> None:
    for label, color in (("Inactive", INACTIVE), ("Active", ACTIVE)):
        values = data.loc[data["activity_class"].eq(label), column].dropna()
        ax.hist(
            values, bins=bins, density=True, alpha=0.5, color=color,
            edgecolor="white", linewidth=0.3, label=label
        )
        ax.axvline(values.median(), color=color, linestyle="--", linewidth=1.1)
    ax.set(title=title, xlabel=xlabel, ylabel="Density")
    ax.yaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def draw_f3(enriched: pd.DataFrame) -> None:
    raw_columns = [
        "inchikey", "activity_class", "label_source", "ic50_uM_numeric",
        "exact_ic50_uM_for_plot",
        "audit_pic50", "MolWt", "MolLogP", "TPSA", "NumHDonors",
        "NumHAcceptors", "NumRotatableBonds", "RingCount",
        "NumAromaticRings", "FractionCSP3", "HeavyAtomCount",
        "LipinskiViolations",
    ]
    enriched[raw_columns].to_csv(
        TABLE_DIR / "F3_physicochemical_activity_data.csv", index=False
    )
    fig, axes = plt.subplots(4, 3, figsize=(13.5, 14.8))
    panels = [
        ("MolWt", "A  Molecular weight", "Molecular weight (Da)", 26),
        ("MolLogP", "B  Lipophilicity", "MolLogP", 26),
        ("TPSA", "C  Polar surface area", "TPSA (Å²)", 26),
        ("NumHDonors", "D  Hydrogen-bond donors", "HBD count", np.arange(-0.5, 11.5, 1)),
        ("NumHAcceptors", "E  Hydrogen-bond acceptors", "HBA count", np.arange(-0.5, 21.5, 1)),
        ("NumRotatableBonds", "F  Molecular flexibility", "Rotatable bonds", np.arange(-0.5, 27.5, 1)),
        ("RingCount", "G  Ring count", "Ring count", np.arange(-0.5, 14.5, 1)),
        ("FractionCSP3", "H  Three-dimensionality proxy", "Fraction C(sp³)", 22),
    ]
    for axis, values in zip(axes.ravel()[:8], panels):
        overlaid_hist(axis, enriched, *values)

    violations = (
        enriched.groupby(["LipinskiViolations", "activity_class"]).size()
        .unstack(fill_value=0).reindex(range(5), fill_value=0)
    )
    ax = axes.ravel()[8]
    x = np.arange(5)
    ax.bar(x, violations["Inactive"], color=INACTIVE)
    ax.bar(x, violations["Active"], bottom=violations["Inactive"], color=ACTIVE)
    ax.set(title="I  Lipinski rule-of-five violations", xlabel="Violations", ylabel="Molecules")
    ax.set_xticks(x)

    ax = axes.ravel()[9]
    exact = enriched.loc[
        enriched["audit_has_exact_pic50"].eq(1)
        & enriched["exact_ic50_uM_for_plot"].gt(0)
    ]
    bins = np.logspace(
        np.log10(exact["exact_ic50_uM_for_plot"].min()),
        np.log10(exact["exact_ic50_uM_for_plot"].max()), 28
    )
    for label, color in (("Inactive", INACTIVE), ("Active", ACTIVE)):
        values = exact.loc[
            exact["activity_class"].eq(label), "exact_ic50_uM_for_plot"
        ]
        ax.hist(values, bins=bins, color=color, alpha=0.56, edgecolor="white")
    ax.axvline(50, color="#111827", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set(title=f"J  Exact IC50 records (n={len(exact)})", xlabel="IC50 (μM; log scale)", ylabel="Molecules")

    ax = axes.ravel()[10]
    counts = enriched["activity_class"].value_counts().reindex(["Inactive", "Active"])
    bars = ax.bar(counts.index, counts.values, color=[INACTIVE, ACTIVE], width=0.62)
    ax.bar_label(bars, padding=3)
    ax.set_ylim(0, max(counts) * 1.16)
    ax.set(title="K  Binary-label distribution", ylabel="Molecules")

    ax = axes.ravel()[11]
    source_order = ["Exact IC50", "NS", "Right-censored", "Other classified"]
    source_counts = enriched["label_source"].value_counts().reindex(source_order, fill_value=0)
    labels = ["Exact\nIC50", "NS", "Right-\ncensored", "Other"]
    bars = ax.bar(labels, source_counts.values, color=[ACCENT, SKY, PURPLE, GREY])
    ax.bar_label(bars, padding=3, fontsize=8)
    ax.set_ylim(0, max(source_counts) * 1.17)
    ax.set(title="L  Label-source distribution", ylabel="Molecules")

    for ax in axes.ravel()[8:]:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    fig.legend(
        handles=[
            Patch(facecolor=INACTIVE, label="Inactive"),
            Patch(facecolor=ACTIVE, label="Active"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2
    )
    fig.suptitle(
        "Physicochemical properties and activity labels",
        fontsize=14, fontweight="bold", y=0.995
    )
    fig.subplots_adjust(
        left=0.065, right=0.985, bottom=0.055, top=0.915,
        wspace=0.25, hspace=0.47
    )
    save_figure(fig, "F3_physicochemical_activity")


def write_summary(
    enriched: pd.DataFrame, family_table: pd.DataFrame, scaffold_table: pd.DataFrame
) -> None:
    exact = enriched.loc[enriched["audit_has_exact_pic50"].eq(1)]
    summary = {
        "source_manifest": str(MANIFEST.relative_to(ROOT)),
        "molecules": int(len(enriched)),
        "active": int(enriched["audit_active_label"].sum()),
        "inactive": int((1 - enriched["audit_active_label"]).sum()),
        "exact_ic50": int(len(exact)),
        "right_censored": int(enriched["audit_censor_type"].eq("right").sum()),
        "ns": int(enriched["ic50_raw_uM"].astype(str).str.upper().isin(["NS", "ND"]).sum()),
        "existing_scaffold_family": int(
            enriched["scaffold_family_source"].eq("existing_annotation").sum()
        ),
        "completed_scaffold_family": int(
            enriched["scaffold_family_source"].ne("existing_annotation").sum()
        ),
        "family_assignment_sources": {
            str(k): int(v)
            for k, v in enriched["scaffold_family_source"].value_counts().items()
        },
        "scaffold_families": int(len(family_table)),
        "murcko_scaffolds_including_acyclic": int(enriched["murcko_scaffold"].nunique()),
        "top20_family_coverage": float(family_table.head(20)["Total"].sum() / len(enriched)),
        "lipinski_zero_violation_fraction": float(
            enriched["LipinskiViolations"].eq(0).mean()
        ),
        "exact_ic50_uM": {
            "min": float(exact["exact_ic50_uM_for_plot"].min()),
            "median": float(exact["exact_ic50_uM_for_plot"].median()),
            "max": float(exact["exact_ic50_uM_for_plot"].max()),
        },
        "top20_scaffolds": scaffold_table.to_dict(orient="records"),
    }
    with (TABLE_DIR / "chemical_space_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def main() -> None:
    configure_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    enriched = load_dataset()
    families = draw_f1(enriched)
    scaffolds = draw_f2(enriched)
    draw_f3(enriched)
    write_summary(enriched, families, scaffolds)
    print(json.dumps({
        "molecules": len(enriched),
        "missing_completed_family": int(enriched["scaffold_family_completed"].isna().sum()),
        "assignment_sources": enriched["scaffold_family_source"].value_counts().to_dict(),
        "figure_dir": str(FIGURE_DIR),
        "table_dir": str(TABLE_DIR),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
