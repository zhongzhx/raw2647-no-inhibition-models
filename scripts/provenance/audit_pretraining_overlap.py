#!/usr/bin/env python3
"""Create a leakage-safe pretraining manifest and quantify structural overlap."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


RDLogger.DisableLog("rdApp.*")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scaffold_smiles(molecule: Chem.Mol) -> str:
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return "<acyclic>"
    # Scaffold identity is intentionally non-stereochemical. Removing residual
    # stereo markers also avoids RDKit canonicalization failures when side-chain
    # removal leaves an incomplete double-bond stereo definition.
    Chem.RemoveStereochemistry(scaffold)
    return Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretraining-index", required=True)
    parser.add_argument("--downstream-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-bits", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pretraining_path = Path(args.pretraining_index)
    downstream_path = Path(args.downstream_manifest)
    pretraining = pd.read_csv(pretraining_path)
    downstream = pd.read_csv(downstream_path)

    if pretraining["inchikey"].duplicated().any():
        raise RuntimeError("Pretraining index contains duplicate InChIKeys")
    if downstream["inchikey"].duplicated().any():
        raise RuntimeError("Downstream manifest contains duplicate InChIKeys")

    downstream_keys = set(downstream["inchikey"].astype(str))
    exact_overlap_mask = pretraining["inchikey"].astype(str).isin(downstream_keys)
    exact_overlap = pretraining.loc[exact_overlap_mask].copy()
    exact_overlap["exclusion_reason"] = "exact_downstream_inchikey"
    strict = pretraining.loc[~exact_overlap_mask].copy().reset_index(drop=True)
    strict.insert(0, "strict_pretraining_index", np.arange(len(strict), dtype=int))
    if set(strict["inchikey"].astype(str)).intersection(downstream_keys):
        raise AssertionError("Exact downstream overlap remains in strict manifest")

    exact_overlap.to_csv(
        output_dir / "excluded_exact_downstream_overlaps.csv", index=False
    )
    strict_path = output_dir / "strict_pretraining_manifest.csv"
    strict.to_csv(strict_path, index=False)

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=args.radius,
        fpSize=args.n_bits,
        includeChirality=True,
    )
    pre_fingerprints = []
    pre_scaffolds = []
    valid_pre_rows = []
    for index, row in strict.iterrows():
        molecule = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if molecule is None:
            raise ValueError(f"Invalid pretraining SMILES at strict row {index}")
        pre_fingerprints.append(generator.GetFingerprint(molecule))
        pre_scaffolds.append(scaffold_smiles(molecule))
        valid_pre_rows.append(index)
        if (index + 1) % 10000 == 0:
            print(f"fingerprinted_pretraining={index + 1}", flush=True)
    strict["bemis_murcko_scaffold"] = pre_scaffolds
    strict.to_csv(strict_path, index=False)

    nonacyclic_pre_scaffolds = {
        value for value in pre_scaffolds if value != "<acyclic>"
    }
    similarity_rows = []
    for index, row in downstream.iterrows():
        molecule = Chem.MolFromSmiles(str(row["canonical_smiles"]))
        if molecule is None:
            raise ValueError(f"Invalid downstream SMILES at row {index}")
        fingerprint = generator.GetFingerprint(molecule)
        similarities = DataStructs.BulkTanimotoSimilarity(
            fingerprint, pre_fingerprints
        )
        nearest_position = int(np.argmax(similarities))
        nearest = strict.iloc[nearest_position]
        scaffold = scaffold_smiles(molecule)
        similarity_rows.append(
            {
                "downstream_manifest_index": int(index),
                "downstream_inchikey": str(row["inchikey"]),
                "downstream_canonical_smiles": str(row["canonical_smiles"]),
                "downstream_scaffold": scaffold,
                "same_nonacyclic_scaffold_in_pretraining": int(
                    scaffold != "<acyclic>"
                    and scaffold in nonacyclic_pre_scaffolds
                ),
                "nearest_pretraining_strict_index": nearest_position,
                "nearest_pretraining_inchikey": str(nearest["inchikey"]),
                "nearest_pretraining_canonical_smiles": str(
                    nearest["canonical_smiles"]
                ),
                "nearest_tanimoto_morgan_r2_2048_chiral": float(
                    similarities[nearest_position]
                ),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"audited_downstream_similarity={index + 1}", flush=True)
    similarity = pd.DataFrame(similarity_rows)
    similarity.to_csv(
        output_dir / "downstream_pretraining_similarity_audit.csv", index=False
    )

    values = similarity["nearest_tanimoto_morgan_r2_2048_chiral"]
    downstream_scaffolds = set(
        similarity.loc[
            similarity["downstream_scaffold"].ne("<acyclic>"),
            "downstream_scaffold",
        ]
    )
    shared_scaffolds = downstream_scaffolds.intersection(
        nonacyclic_pre_scaffolds
    )
    summary = {
        "schema_version": 1,
        "fingerprint_definition": {
            "type": "Morgan",
            "radius": args.radius,
            "n_bits": args.n_bits,
            "include_chirality": True,
            "similarity": "Tanimoto",
        },
        "pretraining_original_rows": len(pretraining),
        "downstream_rows": len(downstream),
        "exact_inchikey_overlaps_excluded": len(exact_overlap),
        "strict_pretraining_rows": len(strict),
        "post_exclusion_exact_inchikey_overlaps": 0,
        "downstream_nonacyclic_scaffold_count": len(downstream_scaffolds),
        "strict_pretraining_nonacyclic_scaffold_count": len(
            nonacyclic_pre_scaffolds
        ),
        "shared_nonacyclic_scaffold_count": len(shared_scaffolds),
        "downstream_molecules_with_shared_nonacyclic_scaffold": int(
            similarity["same_nonacyclic_scaffold_in_pretraining"].sum()
        ),
        "downstream_acyclic_molecules": int(
            similarity["downstream_scaffold"].eq("<acyclic>").sum()
        ),
        "nearest_similarity_mean": float(values.mean()),
        "nearest_similarity_median": float(values.median()),
        "nearest_similarity_max": float(values.max()),
        "downstream_nearest_similarity_ge_0_8": int(values.ge(0.8).sum()),
        "downstream_nearest_similarity_ge_0_9": int(values.ge(0.9).sum()),
        "downstream_nearest_similarity_eq_1": int(
            np.isclose(values.to_numpy(float), 1.0).sum()
        ),
        "input_hashes": {
            "pretraining_index_sha256": sha256_file(pretraining_path),
            "downstream_manifest_sha256": sha256_file(downstream_path),
        },
        "output_hashes": {
            "strict_pretraining_manifest_sha256": sha256_file(strict_path),
            "excluded_exact_downstream_overlaps_sha256": sha256_file(
                output_dir / "excluded_exact_downstream_overlaps.csv"
            ),
            "downstream_pretraining_similarity_audit_sha256": sha256_file(
                output_dir / "downstream_pretraining_similarity_audit.csv"
            ),
        },
    }
    (output_dir / "pretraining_overlap_audit.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
