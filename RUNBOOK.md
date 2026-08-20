# RUNBOOK — model-privacy-attacks

## Prerequisites

- Python 3.9+
- pip
- Target model saved as pickle or scikit-learn artifact

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run Membership Inference Attack (MIA)

```bash
python -m privacy_attacks.mia --model models/target_model.pkl --train-data data/train.csv --test-data data/test.csv
```

Output: `results/mia_report.json` — contains per-sample membership scores and aggregate AUC.

## Run Model Inversion Attack

```bash
python -m privacy_attacks.inversion --model models/target_model.pkl --target-class 1 --output results/inversion/
```

Output: reconstructed feature vectors in `results/inversion/`. Visual plots saved as PNG if matplotlib is available.

## Interpret Results

| Metric | Meaning |
|--------|---------|
| MIA AUC > 0.5 | Model leaks membership info; closer to 1.0 = worse privacy |
| Inversion MSE | Lower MSE = attacker reconstructs training data more accurately |

- AUC ~0.5 → model is reasonably private
- AUC >0.7 → consider differential privacy or regularization

## Test

```bash
pytest tests/ -v
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: sklearn` | `pip install scikit-learn` |
| OOM on large datasets | Subsample with `--max-samples 10000` |
| Model pickle version mismatch | Retrain with current scikit-learn version |
| MIA AUC exactly 0.5 | Check that train/test splits are non-overlapping |
