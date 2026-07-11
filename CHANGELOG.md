# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-07-11

### Added
- **Privacy Certification layer** (`privacy_attacks.certification`) that turns a raw
  attack score into a null-calibrated, reproducible, gate-able verdict:
  - Blind-baseline **panel** (logistic regression + random forest + kNN, out-of-fold);
    the strongest baseline is the null floor, so non-linear input artifacts cannot be
    mis-attributed as model leakage.
  - **Paired bootstrap** AUC-difference test with confidence interval and a one-sided
    p-value (measured type-I error ~2-3% at alpha=0.05).
  - Honest low-FPR handling: realized FPR, a reliability flag, and a bootstrapped
    difference CI; the metric is omitted from the verdict when the sample is too small.
  - Tamper-evidence: SHA-256 content checksum, upgradable to a keyed HMAC signature
    (`PRIVACY_CERT_HMAC_KEY`); `verify_integrity` fails closed against downgrade.
  - Framework-agnostic `TargetModel` adapter interface (`CallableTarget` ships today).
- **`privacy-certify` CLI** with a JSON/YAML policy gate and non-zero exit on violation,
  for use as a CI privacy-regression gate.
- Real-model validation on MNIST (`examples/real_model_demo.py`,
  `examples/REAL_MODEL_FINDINGS.md`) and design/threat-model docs
  (`docs/PRIVACY_CERTIFICATE.md`).
- 12 additional tests, including regressions for the two critical review findings
  (non-linear-artifact rejection and HMAC fail-closed verification).

### Changed
- Added `pyyaml` dependency (YAML policies) and a `privacy-certify` console script.

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
