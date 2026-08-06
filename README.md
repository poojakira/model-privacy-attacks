# model-privacy-attacks

[![CI](https://img.shields.io/github/actions/workflow/status/poojakira/model-privacy-attacks/ci.yml?branch=main&label=CI)](https://github.com/poojakira/model-privacy-attacks/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Educational implementation of ML privacy attacks and defenses. Demonstrates membership inference, model inversion, and differential privacy — attacks AND mitigations together.

> **Scope**: All results are on synthetic data (numpy seed=42). This measures implementation correctness, not real-world privacy leakage of any production model.

## Implemented Attacks

| Attack | Reference | Module | Metric |
|--------|-----------|--------|--------|
| Yeom loss-threshold MIA | Yeom et al., IEEE CSF 2018 | `src/privacy_attacks/mia/yeom_mia.py` | MIA advantage = 0.42 (baseline = 0.10) |
| Shokri shadow model MIA | Shokri et al., IEEE S&P 2017 | `src/privacy_attacks/mia/shadow_mia.py` | AUC on synthetic data |
| Direct confidence MIA | — | `src/privacy_attacks/mia/direct_mia.py` | TPR/FPR at threshold |
| Fredrikson model inversion | Fredrikson et al., ACM CCS 2015 | `src/privacy_attacks/inversion/fredrikson_inversion.py` | Confidence recovered |
| Model extraction | — | `src/privacy_attacks/extraction/extraction_attack.py` | Query budget |

## Defenses

| Defense | Module | Result |
|---------|--------|--------|
| DP-SGD with RDP accounting | `src/privacy_attacks/defenses/dp_sgd.py` | **epsilon=1.16 at sigma=4.0, delta=1e-5** |

Verify the epsilon claim: `python scripts/verify_epsilon.py` → writes `results/epsilon_verification.json`

## MIA Advantage — Honest Metric Explanation

**MIA advantage = TPR - FPR** (not AUC). This is the metric used in Yeom et al. (2018).

- Measured value: **0.42**
- Random baseline: **0.10** (class imbalance in synthetic data, not the theoretical 0.0)
- Excess advantage over baseline: **0.32**
- Risk level: HIGH (per `src/privacy_attacks/compliance_report.py`)

A model is considered private when its MIA advantage is at or below the random baseline. See `results/mia_advantage_report.json`.

## Compliance Report

```python
from privacy_attacks.compliance_report import generate_compliance_report, save_report

report = generate_compliance_report(
    mia_advantage=0.42,
    epsilon=1.16,
    delta=1e-5,
    model_name="my-classifier",
    random_baseline_advantage=0.10,
)
save_report(report, "results/compliance.json")
```

Outputs JSON with MIA advantage, epsilon, risk level, and EU AI Act articles triggered.

## Install

```bash
pip install -e ".[dev]"
```

## ATT&CK Mapping

| Technique | Description | Module |
|-----------|-------------|--------|
| T1685 | ML Privacy Attack | All MIA modules |
| T1689 | Model Inversion | `inversion/fredrikson_inversion.py` |
| T1682 | Model Extraction | `extraction/extraction_attack.py` |
