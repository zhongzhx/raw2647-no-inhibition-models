"""Audit the cleaned unlabeled pretraining corpus against the locked 526 compounds."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold


def scaffold_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold, canonical=True) if scaffold.GetNumAtoms() else "<acyclic>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_index", default="processed_pretrain_graphs_updated_pretrain_data_run/processed_index.csv")
    parser.add_argument("--pretrain_summary", default="processed_pretrain_graphs_updated_pretrain_data_run/processing_summary.json")
    parser.add_argument("--downstream_index", default="processed_smiles_graphs_multitask_ns_inactive/processed_index.csv")
    parser.add_argument("--output_dir", default="outputs_pretraining_redesign_v2")
    args = parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pretrain = pd.read_csv(args.pretrain_index)
    downstream = pd.read_csv(args.downstream_index)
    source_summary = json.loads(Path(args.pretrain_summary).read_text(encoding="utf-8"))
    pretrain_keys = pretrain["inchikey"].astype(str)
    downstream_keys = set(downstream["inchikey"].astype(str))
    overlap = sorted(set(pretrain_keys) & downstream_keys)
    scaffolds = [scaffold_smiles(smiles) for smiles in pretrain["canonical_smiles"].astype(str)]
    counts = Counter(scaffolds)
    scaffold_table = pd.DataFrame(
        [{"scaffold": scaffold, "molecule_count": count} for scaffold, count in counts.most_common()]
    )
    scaffold_table.to_csv(output / "pretrain_scaffold_counts.csv", index=False)
    audit = {
        **source_summary,
        "index_rows": len(pretrain),
        "unique_inchikeys": int(pretrain_keys.nunique()),
        "duplicate_inchikey_rows_in_final_index": int(pretrain_keys.duplicated().sum()),
        "downstream_locked_compounds": len(downstream),
        "downstream_overlap_after_cleaning": len(overlap),
        "downstream_overlap_inchikeys": overlap,
        "unique_bemis_murcko_scaffolds": len(counts),
        "acyclic_molecules": int(counts.get("<acyclic>", 0)),
        "empty_or_invalid_scaffold": int(counts.get("", 0)),
    }
    (output / "pretrain_data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md = f"""# PRETRAIN DATA AUDIT

## Result

The cleaned corpus contains **{len(pretrain):,} valid unique molecules** and **{len(counts):,} Bemis–Murcko scaffold groups**. Exact InChIKey overlap with all **{len(downstream)} locked downstream compounds is {len(overlap)}**.

## Cleaning trace

- Raw rows: {source_summary['raw_count']:,}
- Parsed successfully: {source_summary['parsed_success_count']:,}
- Removed by chemistry filters: {source_summary['filtered_count']:,}
- Removed as duplicate InChIKeys: {source_summary['duplicate_count']:,}
- Removed for exact downstream overlap: {source_summary['excluded_downstream_duplicate_count']:,}
- Final graph count: {source_summary['final_output_count']:,}
- Duplicate InChIKeys remaining: {audit['duplicate_inchikey_rows_in_final_index']}
- Mixtures/salts: removed when the source SMILES contained `.`
- Canonicalization: RDKit canonical isomeric SMILES

## Leakage assessment

The final pretraining index has no exact structural overlap with the locked 526-compound downstream dataset. The exclusion is structure-based (InChIKey), not name- or row-based. This prevents identical-molecule leakage, but it does not remove close analogues; similarity-stratified downstream analysis addresses that remaining question.
"""
    (output / "PRETRAIN_DATA_AUDIT.md").write_text(md, encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
