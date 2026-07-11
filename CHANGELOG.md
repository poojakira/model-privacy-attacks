# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-11

### Added
- Confidence-threshold membership inference (`ThresholdAttack`) with a manual,
  sklearn-free AUC (`attack_auc`) computed via the Mann-Whitney U statistic.
- Entropy-based membership inference (`EntropyAttack`) using Shannon entropy of softmax
  outputs, with held-out threshold calibration.
- Shadow-model membership inference (`ShadowModelAttack`): trains imitation models and a
  logistic-regression attack classifier over sorted confidence vectors.
- Decision-boundary stealing model-extraction attack (`BoundaryStealingAttack`) with a
  fidelity metric measuring clone/oracle decision agreement.
- Evaluation metrics: manual trapezoid-rule AUC (`compute_auc`), low-FPR advantage
  (`attack_advantage`), and a human-readable `membership_inference_report`.
- 12 tests, including the `AUC > 0.6` membership-inference guarantee and the
  `fidelity > 0.7` extraction guarantee on synthetic data.
- Continuous integration across Python 3.10-3.12.
