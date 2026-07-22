## MITRE ATT&CK v19 Coverage

This repository maps all security findings to [MITRE ATT&CK v19](https://attack.mitre.org/).

| Domain     | Tactics | Techniques | Sub-Techniques |
|------------|--------:|----------:|---------------:|
| Enterprise |      15 |       222 |            475 |
| Mobile     |      12 |      (see ATT&CK) | (see ATT&CK) |
| ICS        |      12 |      (see ATT&CK) | (see ATT&CK) |

### Export ATT&CK Navigator Layer

```bash
python -m attack_mapping.reporter --output navigator_layer.json
```

Open in [ATT&CK Navigator](https://mitre-attack.github.io/attack-navigator/) to visualize coverage.

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

### Model Privacy Attacks Specific Mappings

| Finding Type | Techniques |
|--------------|------------|
| membership_inference_success | T1005, T1213.002 |
| model_stealing_detected | T1005, T1114 |
| attribute_inference | T1552, T1213 |
| gradient_leakage | T1005, T1557 |
| model_inversion_pii | T1005, T1078 |
| differential_privacy_bypass | T1562, T1565 |
| federated_learning_poisoning | T1195, T1565 |
| api_probing_extraction | T1190, T1595 |