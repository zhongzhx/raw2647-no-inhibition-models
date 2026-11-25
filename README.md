# RAW264.7 nitric oxide inhibition models

Model code, configurations, and weights for *Molecular Representations for Scaffold-Aware Prediction of Nitric Oxide Inhibition by Natural Products and Related Derivatives*.

- `models/pretrained/` contains the pretrained GINE encoder and the parameters needed to load it.
- `models/locked_validation/` contains 12 fitted neural models (A0–A3) and six fitted random forests (T1 and T3), each at seeds 13, 42, and 2026.
- `models/descriptor_standardization.pt` contains the fitted descriptor transformation used by A2 and A3. `models/MANIFEST.csv` lists artifact hashes.
- `src/`, `scripts/`, and `configs/` contain the model implementation, feature generation, training and evaluation code, and hyperparameters.

Use Python 3.13 and the package versions in `requirements.txt`. The inference entry point takes user-supplied structures and prints the mean positive-class score across the three model seeds.

```sh
python -m pip install -r requirements.txt
python predict.py --model T3 'CCO'
```
