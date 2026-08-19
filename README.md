# model-privacy-attacks

Evaluates membership-inference and model-inversion risk on ML classifiers, then measures DP-SGD defense cost. Maps findings to MITRE ATT&CK v19.

[![CI](https://github.com/poojakira/model-privacy-attacks/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/model-privacy-attacks/actions/workflows/ci.yml)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![MIT](https://img.shields.io/badge/license-MIT-green)

## What It Does

- **Membership inference**  -  shadow-model attack (Shokri 2017), measures whether a sample was in training
- **Model inversion**  -  gradient-based reconstruction (Fredrikson 2015), measures SSIM/PSNR
- **DP-SGD defense**  -  trains at configurable epsilon, shows privacy-utility tradeoff

## Privacy-Utility Tradeoff (ResNet-18, CIFAR-10)

| Epsilon | Test Accuracy | MIA Success | Notes |
|---------|--------------|-------------|-------|
| ∞ (no DP) | 93.2% | 62.4% | Baseline |
| 8.0 | 88.1% | 53.8% | Moderate defense |
| 1.0 | 71.3% | 51.2% | Near random  -  defense works |

## Quick Start

```bash
git clone https://github.com/poojakira/model-privacy-attacks.git && cd model-privacy-attacks
pip install -e ".[dev]"
python -m privacy_attacks.evaluate --attack membership --dataset cifar10
```

## License

MIT.
