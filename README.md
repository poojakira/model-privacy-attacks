# model-privacy-attacks

Membership inference and model inversion attacks against ML classifiers, with DP-SGD defense measurement. Implements Shokri et al. 2017 and Fredrikson et al. 2015.

## What It Does

- **Membership inference** — shadow-model attack (Shokri 2017), measures whether a sample was in training
- **Model inversion** — gradient-based reconstruction (Fredrikson 2015), measures SSIM/PSNR
- **DP-SGD defense** — trains at configurable epsilon, shows privacy-utility tradeoff

## Actual Results vs. Paper Claims

On the Adult Income dataset (UCI, 48,842 records):

| Model Condition | Direct MIA AUC | Shadow MIA AUC | Notes |
|----------------|---------------|---------------|-------|
| Well-generalized (gap -0.3%) | 0.499 | 0.500 | No leakage detectable |
| Moderate overfit (gap 3%) | 0.510 | 0.497 | Barely above random |
| Heavily overfitted (gap 14%) | 0.625 | 0.568 | Leakage only with extreme overfitting |

Papers (Shokri 2017) report AUC >0.9 on heavily overfitted models with much larger shadow model ensembles. This implementation uses 4 shadow models and achieves modest results. The attacks work as described in the literature but only produce meaningful signal when the target model is severely overfitting.

The synthetic-data benchmark (1000 samples) shows MIA advantage of 0.42, but that result measures implementation correctness on toy data, not real privacy risk.

## Quick Start

```bash
git clone https://github.com/poojakira/model-privacy-attacks.git && cd model-privacy-attacks
pip install -e ".[dev]"
python -m privacy_attacks.evaluate --attack membership --dataset cifar10
```

## License

MIT.
