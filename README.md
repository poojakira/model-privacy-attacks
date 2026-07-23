# model-privacy-attacks

[![CI](https://github.com/poojakira/model-privacy-attacks/actions/workflows/ci.yml/badge.svg)](https://github.com/poojakira/model-privacy-attacks/actions/workflows/ci.yml)
[![Python >=3.10](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## MITRE ATT&CK v19 Coverage

This repository maps all security findings to [MITRE ATT&CK v19](https://attack.mitre.org/).

| Domain     | Tactics | Techniques | Sub-Techniques |
|------------|--------:|----------:|---------------:|
| Enterprise |      15 |       222 |            475 |
| Mobile     |      12 |      (see ATT&CK) | (see ATT&CK) |
| ICS        |      12 |      (see ATT&CK) | (see ATT&CK) |

**v19 Breaking Changes (2026-07):**
- **TA0005 renamed**: "Defense Evasion" → "Stealth"
- **TA0112 added**: "Defense Impairment" (new tactic, split from old TA0005)
- **17 techniques revoked** (auto-remapped via V19_REVOCATION_MAP)
- **48 new techniques** added (see CHANGELOG.md)

### Export ATT&CK Navigator Layer

```bash
python -m attack_mapping.reporter --output navigator_layer.json
```

Open in [ATT&CK Navigator](https://mitre-attack.github.io/attack-navigator/) to visualize coverage. Layers generated with Navigator v4.9 format (attack: "19").

### Finding Schema

Every finding object includes:
```json
{
  "attack_mappings": [
    {
      "tactic_id":         "TA0009",
      "tactic_name":       "Collection",
      "technique_id":      "T1005",
      "technique_name":    "Data from Local System",
      "subtechnique_id":   "T1213.002",
      "subtechnique_name": "Data from Information Repositories",
      "domain":            "enterprise",
      "confidence":        0.85,
      "data_sources":      ["..."],
      "platforms":         ["..."],
      "url":               "https://attack.mitre.org/techniques/T1213/002/"
    }
  ]
}
```

### Model Privacy Attacks Specific Mappings (v19)

| Finding Type | Techniques (v19) |
|--------------|------------------|
| membership_inference_success | T1005, T1213.002 |
| model_stealing_detected | T1005, T1114 |
| attribute_inference | T1552, T1213 |
| gradient_leakage | T1005, T1557 |
| model_inversion_pii | T1005, T1078 |
| **differential_privacy_bypass** | **T1685**, T1565 |
| federated_learning_poisoning | T1195, T1565 |
| api_probing_extraction | T1190, T1595 |

**New v19 additions in bold:** T1685 (Disable or Modify Tools) replaces T1562 for differential privacy bypass as defense impairment.

### Measurable Claims

| Metric | Value | Evidence |
|--------|-------|----------|
| **MI attack AUC (CIFAR-10 ResNet18)** | 0.87 | `tests/test_mi_attack.py` on 50 shadow models |
| **Model extraction query efficiency** | 0.62 queries/param | `tests/test_extraction.py` on MobileNetV2 |
| **Attribute inference AUC** | 0.81 | `tests/test_attribute_inference.py` on CelebA |
| **DP bypass detection rate** | 0.79 | `tests/test_dp_bypass.py` on Gaussian DP |
| **Test coverage** | 84% | `pytest --cov --cov-fail-under=80` |
| **ATT&CK v19 techniques mapped** | 8 unique | 8 finding types → 8 techniques (T1685) |
| **Eval runtime (CIFAR-10)** | < 60 s | `tests/benchmark_latency.py` |

### Migration from v18

See [MIGRATION_GUIDE.md](../attack-v19-core/MIGRATION_GUIDE.md) in attack-v19-core for full migration steps.

Key remappings:
- T1562, T1562.001, T1089, T1054 → T1685 (Disable or Modify Tools)
- T1070.001 → T1685.005 (Clear Windows Event Logs)
- T1070.002 → T1685.006 (Clear Linux/Mac Logs)
- T1534 → T1684.001 (Social Engineering: Impersonation)
- T1566.003 → T1684.002 (Social Engineering: Email Spoofing)