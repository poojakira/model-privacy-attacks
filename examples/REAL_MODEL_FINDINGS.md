# Real-Model Validation of the Privacy-Certification Engine

We validated the certification engine on the **canonical membership-inference target**:
MNIST (70,000 real handwritten-digit images, 784 pixel features). We drew a small
**member (train) set of 2,000 images** and a **disjoint held-out non-member set of
2,000 images** from the same distribution (fixed seed 0), then trained two genuine
neural networks on the identical member set:

* an **overfit MLP** (`hidden_layer_sizes=(256, 256)`, `alpha=1e-6`) that memorizes, and
* a **regularized MLP** (`hidden_layer_sizes=(64,)`, `alpha=3.0`) that generalizes.

For each we ran the black-box `max_confidence` membership attack and certified it against
a **strong blind-baseline panel** (logistic regression + random forest + kNN, out-of-fold
over the raw pixels), taking the strongest baseline as the null floor.

## Results (actual numbers from `python examples/real_model_demo.py`, seed=0)

| model | train_acc | test_acc | naive_model_AUC | blind_AUC (panel) | delta_AUC (95% CI) | p_value | verdict |
|---|---|---|---|---|---|---|---|
| overfit MLP (256×256, alpha=1e-6) | 1.000 | 0.924 | 0.558 | 0.515 | 0.044 [0.019, 0.070] | 0.0010 | **CERTIFIED_LEAKAGE** |
| regularized MLP (64, alpha=3.0) | 0.958 | 0.894 | 0.527 | 0.515 | 0.012 [-0.013, 0.039] | 0.1789 | **INCONCLUSIVE** |

Blind panel per-baseline AUCs: `logreg 0.506, random_forest 0.515, knn 0.501` — all near
chance, confirming that raw pixels barely reveal membership, so the certified separation
is genuinely **model-attributable**, not a distribution artifact.

## Why this is the honest, product-grade result

- **The naive view cannot tell the models apart.** Raw attack AUCs are 0.558 vs 0.527 —
  a credulous tool would shrug at both. Only the null-calibrated certificate separates
  them: the overfit model clears the strong baseline with p = 0.001 and a delta-AUC CI
  strictly above zero; the regularized model's CI straddles zero (p = 0.18) and is left
  **INCONCLUSIVE** rather than falsely cleared or falsely flagged.
- **It survives a strong null.** An earlier version used a single linear blind baseline
  and certified leakage on the easy `load_digits` set. A skeptical review showed a linear
  baseline can miss non-linear input artifacts (an XOR of two features), so we replaced
  it with a panel and take the strongest baseline as the floor. Under that stronger null,
  marginal/easy-dataset "leakage" correctly collapses to INCONCLUSIVE, and only genuine
  memorization (the overfit MLP on MNIST) still certifies.
- **Every number is reproducible** (fixed seeds; the AUC difference is a paired bootstrap
  test) and the certificate carries a content checksum (SHA-256), upgradable to an HMAC
  signature via `PRIVACY_CERT_HMAC_KEY`.

## Caveats (stated plainly)

- The effect size is modest (delta ≈ 0.044) because MNIST is learnable even from 2,000
  samples, so the overfit model still generalizes (test acc 0.924). What separates the
  two models is **statistical significance against a strong null**, not a dramatic AUC
  gap — exactly the regime the engine is designed to adjudicate.
- `TPR@0.1% FPR` is reported as **unavailable** at this sample size: 2,000 non-members
  cannot resolve a 0.1% false-positive rate (it needs ≥ 5,000), so the engine omits it
  from the verdict rather than printing a misleading tail number.
- First run downloads MNIST via OpenML (~15 MB, then cached). The example is not part of
  the CI test suite for that reason.
