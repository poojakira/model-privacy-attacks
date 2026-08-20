# model-privacy-attacks

Membership inference and model inversion attacks measuring privacy leakage from ML models, with DP-SGD defense evaluation.

## Key Metrics

| Metric | Value |
|--------|-------|
| MIA AUC (heavily overfitted target) | 0.625 (direct), 0.568 (shadow) |
| MIA AUC (well-generalized target) | 0.499 — no detectable leakage |
| Model inversion quality | SSIM / PSNR measured |
| Shadow models | 4 (Shokri et al. 2017 methodology) |
| Defense tested | DP-SGD at configurable ε |
| Datasets | Adult Income (48,842 records), CIFAR-10 |

## Architecture

```
┌──────────────────┐     ┌───────────────────┐     ┌─────────────────┐
│  Target Model    │────▶│  Attack Module    │────▶│  Privacy Report │
│  (train + eval)  │     │  MIA / Inversion  │     │  AUC, SSIM/PSNR │
└──────────────────┘     └───────────────────┘     └─────────────────┘
         │                        │                         │
         ▼                        ▼                         ▼
  Controlled overfit         Shadow model            Quantified leakage
  levels (0–14% gap)        ensemble (×4)           vs. model condition
```

**Attack Implementations:**

| Attack | Paper | What It Measures |
|--------|-------|-----------------|
| Membership Inference (shadow) | Shokri et al. 2017 | Whether a sample was in training data |
| Membership Inference (direct) | Yeom et al. 2018 | Loss-threshold membership test |
| Model Inversion | Fredrikson et al. 2015 | Reconstruction of training inputs from gradients |

**Defense:**
- DP-SGD training at configurable privacy budget (ε)
- Gradient clipping + Gaussian noise during training
- Measures privacy-utility tradeoff: accuracy loss vs. MIA AUC reduction

## Key Findings

| Model Condition | Generalization Gap | Direct MIA AUC | Shadow MIA AUC |
|----------------|-------------------|----------------|----------------|
| Well-generalized | -0.3% | 0.499 | 0.500 |
| Moderate overfit | 3% | 0.510 | 0.497 |
| Heavy overfit | 14% | 0.625 | 0.568 |

Privacy leakage is measurable only when the target model severely overfits. Well-regularized models show no exploitable membership signal — consistent with theoretical expectations that memorization drives membership inference success.

Note: Published results (Shokri 2017) report AUC >0.9 using larger shadow model ensembles on more vulnerable targets. This implementation uses 4 shadow models and produces modest but directionally correct results.

## Quick Start

```bash
git clone https://github.com/poojakira/model-privacy-attacks.git && cd model-privacy-attacks
pip install -e ".[dev]"
```

The attacks are a library. Membership inference takes a trained target model plus
member (training) and non-member (held-out) samples:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from privacy_attacks.mia import DirectMIA

X, y = make_classification(n_samples=2000, n_features=20, random_state=0)
X_members, y_members = X[:1000], y[:1000]        # data the model trained on
X_nonmembers, y_nonmembers = X[1000:], y[1000:]  # held-out data

model = RandomForestClassifier(random_state=0).fit(X_members, y_members)

mia = DirectMIA().fit(model, X_members, y_members, X_nonmembers, y_nonmembers)
result = mia.evaluate(X_members, X_nonmembers, y_members, y_nonmembers)
print("MIA AUC:", result)   # ~0.70 — the model leaks membership signal
```

Run the full attack + defense test suite:

```bash
pytest tests/ -v
```

## Relevance to AI Security

Training data extraction is a primary concern for LLM deployments — memorization of PII, copyrighted content, and API keys. This implementation demonstrates the conditions under which membership inference succeeds and when it does not.

The key insight: generalization gap is the strongest predictor of privacy leakage. This directly informs deployment decisions:
- Proper regularization reduces privacy risk as a side effect
- DP-SGD provides formal guarantees at quantifiable utility cost
- Monitoring train/test divergence serves as a proxy for privacy risk

Understanding attack-defense dynamics is necessary for building ML systems that handle sensitive data responsibly.

## License

MIT
