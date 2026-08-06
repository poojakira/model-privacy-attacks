# model-privacy-attacks

[![CI](https://github.com/poojakira/model-privacy-attacks/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/model-privacy-attacks/actions/workflows/ci.yml)
[![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SARIF](https://img.shields.io/badge/SARIF-Enabled-blueviolet)](https://docs.github.com/en/code-security/code-scanning)
[![NIST AI RMF](https://img.shields.io/badge/NIST%20AI%20RMF-Mapped-green)](https://airc.nist.gov/RMF_Overview)

[Live Dashboard](https://poojakira.github.io/model-privacy-attacks/)

---

## Privacy Risk Assessment Framework

This toolkit implements published privacy attacks — **Membership Inference (Yeom 2018, Shokri 2017), Model Inversion (Fredrikson 2015), and Attribute Inference** — as a structured **compliance assessment tool**. It enables teams to measure their model's privacy risk posture and map findings to EU AI Act Art. 10/15 and NIST AI RMF GOVERN/MANAGE functions.

This is not an offensive tool — it is the privacy equivalent of a penetration test: use it to find your own model's vulnerabilities before a regulator or adversary does.

**Exercised on synthetic and toy data.** See the Evidence Status section below for what is and is not claimed.

---

## Why This Matters

> A Membership Inference Attack (MIA) advantage of 0.42 against a 0.10 random baseline means an attacker can determine with **4× better-than-chance accuracy** whether a specific individual's data was used to train this model — a direct GDPR and EU AI Act violation risk. Models with MIA advantage > 0.20 are at material regulatory exposure.

This is not a theoretical concern. The EU AI Act (effective 2026) requires high-risk AI systems to demonstrate data governance controls under Art. 10. This toolkit provides the evidence.

---

## Compliance Mapping

| Attack | Risk Measured | EU AI Act Article | NIST AI RMF Function |
|--------|--------------|------------------|---------------------|
| Membership Inference (Yeom 2018) | Training data exposure — can attacker tell if a record was in training set? | Art. 10 (Data Governance) | MANAGE 2.2 |
| Membership Inference (Shokri 2017) | Model memorization — shadow model attack | Art. 10 | MANAGE 2.2 |
| Model Inversion (Fredrikson 2015) | PII reconstruction risk — can attacker recover training inputs from outputs? | Art. 15 (Accuracy & Robustness) | MANAGE 4.1 |
| DP-SGD Defense | Privacy budget (ε, δ) — quantifies formal privacy guarantee | Art. 10 | GOVERN 1.1 |
| Attribute Inference | Sensitive attribute leakage from model outputs | Art. 10 | MANAGE 2.2 |
| Model Extraction | Intellectual property theft + secondary attack surface | Art. 15 | MANAGE 4.2 |

---

## Audit Report Output

Run a privacy assessment and get a structured JSON compliance report:

```bash
python -m privacy_attacks.assess \
  --model-path ./my_model.pt \
  --dataset-path ./eval_data.csv \
  --output privacy_audit.json
```

Sample output:
```json
{
  "tool": "model-privacy-attacks",
  "version": "1.0.0",
  "assessment_date": "2026-08-05",
  "model_id": "resnet18-cifar10",
  "mia_advantage": 0.42,
  "threshold": 0.10,
  "risk_level": "HIGH",
  "findings": [
    {
      "attack": "membership_inference_yeom",
      "mia_advantage": 0.42,
      "random_baseline": 0.10,
      "severity": "HIGH",
      "eu_ai_act_articles_triggered": ["Art. 10"],
      "nist_ai_rmf_function": "MANAGE 2.2",
      "remediation_hint": "Apply DP-SGD with epsilon <= 1.0 or increase training regularization."
    }
  ],
  "eu_ai_act_articles_triggered": ["Art. 10", "Art. 15"],
  "recommended_defense": "DP-SGD with epsilon <= 1.0",
  "achieved_epsilon": 1.16,
  "sigma": 4.0,
  "severity_summary": {
    "CRITICAL": 0,
    "HIGH": 1,
    "MEDIUM": 0,
    "LOW": 0
  },
  "remediation_hints": [
    "Apply DP-SGD with epsilon <= 1.0 and sigma >= 1.5",
    "Reduce training epochs to limit memorization",
    "Add output perturbation (label smoothing, prediction confidence thresholding)"
  ]
}
```

---

## Developer Self-Service: Privacy Budget Calculator

Before training, estimate your privacy budget:

```python
from privacy_attacks.budget_calculator import privacy_budget_calculator

result = privacy_budget_calculator(
    dataset_size=50000,
    epochs=10,
    batch_size=256,
    sigma=1.0,
    delta=1e-5
)
print(result)
# {
#   "epsilon": 3.2,
#   "delta": 1e-5,
#   "risk_level": "MEDIUM",
#   "recommendation": "Increase sigma to >= 1.5 to reach epsilon <= 1.0 (LOW risk)",
#   "noise_multiplier": 1.0,
#   "effective_batch_size": 256
# }
```

This is the paved path: calculate your privacy exposure before training, not after.

---

## MITRE ATT&CK v19 Coverage

| Finding Type | Techniques (v19) |
|-------------|-----------------|
| membership_inference_success | T1005, T1213.002 |
| model_stealing_detected | T1005, T1114 |
| attribute_inference | T1552, T1213 |
| gradient_leakage | T1005, T1557 |
| model_inversion_pii | T1005, T1078 |
| **differential_privacy_bypass** | **T1685**, T1565 |
| federated_learning_poisoning | T1195, T1565 |
| api_probing_extraction | T1190, T1595 |

Export ATT&CK Navigator layer:
```bash
python -m attack_mapping.reporter --output navigator_layer.json
```

---

## Evidence Status

| Claim Area | Current Evidence |
|-----------|-----------------|
| Privacy attack implementations | Unit tests in `tests/test_privacy_attacks.py` exercise implemented attack paths on synthetic or toy data |
| ATT&CK v19 mapping | Mapping tests and reporter code present in repo |
| Compliance report generation | `generate_report()` in `src/privacy_attacks/report.py` — outputs structured JSON |
| Privacy budget calculator | `privacy_budget_calculator()` in `src/privacy_attacks/budget_calculator.py` |
| **Public benchmark metrics** | **No committed CIFAR-10 / ResNet18 / CelebA benchmark artifact.** Do not cite AUC/query-efficiency numbers from this README. |
| Production readiness | Not claimed. Real privacy-risk claims require target-model, dataset, split, and confidence-interval evidence. |

---

## Install

```bash
pip install model-privacy-attacks
```

From source:
```bash
git clone https://github.com/poojakira/model-privacy-attacks
cd model-privacy-attacks
pip install -e ".[dev]"
```

---

## Usage

```python
from privacy_attacks.mia import MembershipInferenceAttack
from privacy_attacks.report import generate_report

# Run membership inference attack
mia = MembershipInferenceAttack()
advantage = mia.evaluate(model, train_data, test_data)

# Generate compliance report
report = generate_report(
    model_id="my-model-v1",
    mia_advantage=advantage,
    mia_threshold=0.10,
    epsilon=1.16,
    sigma=4.0,
    dataset_size=50000
)
```

---

## License

MIT
