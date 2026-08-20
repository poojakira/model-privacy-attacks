# RUNBOOK — model-privacy-attacks

## Prerequisites

- Python 3.9+
- pip
- Target model saved as pickle or scikit-learn artifact

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Run Membership Inference Attack (MIA)

The attacks are a library. `DirectMIA` takes a trained target model plus member
(training) and non-member (held-out) data.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from privacy_attacks.mia import DirectMIA

X, y = make_classification(n_samples=2000, n_features=20, random_state=0)
X_members, y_members = X[:1000], y[:1000]
X_nonmembers, y_nonmembers = X[1000:], y[1000:]

model = RandomForestClassifier(random_state=0).fit(X_members, y_members)

mia = DirectMIA().fit(model, X_members, y_members, X_nonmembers, y_nonmembers)
print(mia.evaluate(X_members, X_nonmembers, y_members, y_nonmembers))
```

`ShadowMIA` (shadow-model attack) and `ModelExtractionAttack` follow the same
fit/evaluate pattern. See `tests/` for complete examples.

## Interpret Results

| Metric | Meaning |
|--------|---------|
| MIA AUC > 0.5 | Model leaks membership info; closer to 1.0 = worse privacy |
| MIA AUC ~0.5 | Model is reasonably private |
| MIA AUC > 0.7 | Consider differential privacy (DP-SGD) or stronger regularization |

## Test

```bash
pytest tests/ -v
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ModuleNotFoundError: privacy_attacks` | Run `pip install -e ".[dev]"` from repo root |
| `attack_v19_core` test skipped | Optional dependency, not on PyPI — safe to ignore |
| MIA AUC exactly 0.5 | Check that member/non-member splits do not overlap |
| OOM on large datasets | Subsample before passing to `fit` |
