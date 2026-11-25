#!/usr/bin/env python3
"""Score user-supplied SMILES with the released fitted models."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts" / "provenance"))

import run_current_rebuild_models as runner
from formal_transfer_ablation_core import build_model, predict_model
from smiles_to_graph import smiles_to_pyg_data

MODELS = ("A0", "A1", "A2", "A3", "T1", "T3")


def predict(model_id, smiles):
    config_path = ROOT / "configs" / "current_rebuild_v3_locked_validation.json"
    config = json.loads(config_path.read_text())
    graphs = [smiles_to_pyg_data(s, strict=True) for s in smiles]
    weights = ROOT / "models"
    model_paths = weights / "locked_validation" / model_id
    scores = []

    if model_id.startswith("T"):
        features = runner.feature_matrices(graphs)
        x = features["flat" if model_id == "T1" else "rdkit"]
        for seed in config["model_seeds"]:
            model = joblib.load(model_paths / f"seed_{seed}" / "model.joblib")
            scores.append(model.predict_proba(x)[:, 1])
    else:
        if model_id in ("A2", "A3"):
            scaler = torch.load(
                weights / "descriptor_standardization.pt", map_location="cpu", weights_only=True
            )
            for graph in graphs:
                graph.global_feat = (graph.global_feat.float() - scaler["mean"]) / scaler["std"]

        args = runner.training_args(config, smoke=False)
        args.device = "cpu"
        spec = runner.model_specs(config)[model_id]
        checkpoint = weights / "pretrained" / "gine_encoder_best.pt"
        for seed in config["model_seeds"]:
            model, _, _, _ = build_model(spec, graphs[0], checkpoint, args, seed)
            state = torch.load(
                model_paths / f"seed_{seed}" / "model_state_dict.pt",
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(state)
            scores.append(predict_model(
                model, graphs, np.arange(len(graphs)), torch.device("cpu"), args.batch_size
            ))

    return np.mean(scores, axis=0).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("smiles", nargs="+", help="Quote each SMILES on the command line.")
    args = parser.parse_args()
    print(json.dumps(predict(args.model, args.smiles)))


if __name__ == "__main__":
    main()
