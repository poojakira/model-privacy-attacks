# Threat Model — model-privacy-attacks

**Version:** 1.0  
**Date:** 2026-08-05  
**ATT&CK version:** v19 (TA0005=Stealth, TA0112=Defense Impairment)  
**Scope:** Threats to the assessment framework itself, not to models being assessed.

---

## 1. System Overview

`model-privacy-attacks` is a privacy risk assessment CLI and library. It:

1. Accepts a trained ML model as input (serialized weights, ONNX, or scikit-learn object)
2. Runs membership inference, model inversion, attribute inference, and extraction attacks
3. Computes a differential privacy budget estimate
4. Generates a structured JSON compliance report mapping findings to EU AI Act and NIST AI RMF

**Trust boundary:** The framework runs locally in the assessor's environment. It does not send model weights, training data, or assessment results to any external service. Report output is written to local files.

---

## 2. Assets

| Asset | Confidentiality | Integrity | Availability |
|-------|----------------|-----------|--------------|
| Model weights (input) | HIGH — may encode training data | HIGH — corrupted weights yield invalid results | MEDIUM |
| Training data (input) | CRITICAL — personal data possible | HIGH | MEDIUM |
| Assessment report (output) | HIGH — reveals privacy posture | CRITICAL — false risk levels cause compliance failures | MEDIUM |
| Privacy budget calculation | MEDIUM | CRITICAL — epsilon underestimation is a safety failure | LOW |
| CI/CD pipeline configuration | MEDIUM | HIGH — bypass could suppress required gates | MEDIUM |

---

## 3. Threat Actors

| Actor | Motivation | Capability |
|-------|-----------|-----------|
| ML model owner seeking to hide privacy violations | Suppress HIGH/CRITICAL findings in audit report | Insider access; can modify model or input data |
| Supply chain attacker targeting the assessment tool | Corrupt epsilon calculation to underestimate risk | Can publish malicious dependency updates |
| Automated CI bypass | Skip privacy assessment gates entirely | Can modify workflow YAML or coverage thresholds |
| Adversarial model provider | Feed crafted model weights that cause assessment tool to crash or hang | External; can control model artifact |

---

## 4. Threat Enumeration (STRIDE)

### 4.1 Spoofing

**T-SPOOF-01: Forged model ID in report**  
An attacker with access to the assessment pipeline substitutes a benign model's `model_id` in the report while the assessment runs against a high-risk model.  
- **Mitigation:** Report generator should hash model weights and include the hash in the report. Downstream consumers should verify `model_hash` matches the deployed artifact.  
- **ATT&CK v19:** T1036 (Masquerading, TA0005 Stealth)  
- **Status:** Not yet implemented. Planned.

**T-SPOOF-02: Assessment date tampering**  
An attacker sets a backdated `assessment_date` to make a stale assessment appear current.  
- **Mitigation:** Assessment date defaults to `datetime.now(UTC)` and cannot be set externally via CLI. Internal callers can pass a custom date but this is audited via function signature.  
- **Status:** Mitigated in `report.py`.

---

### 4.2 Tampering

**T-TAMP-01: Adversarial model weights causing epsilon underestimation**  
A crafted model with artificially low training loss variance could cause the Yeom MIA to report near-zero advantage, masking high actual privacy risk.  
- **Impact:** Report returns `LOW` risk when model is genuinely high-risk.  
- **Mitigation:** Run multiple attack variants (Yeom + Shokri). Flag cases where loss variance is suspiciously low. Do not rely on a single attack primitive for compliance decisions.  
- **ATT&CK v19:** T1027/018 (Obfuscated Files — ML Model Obfuscation, TA0005 Stealth)  
- **Status:** Partial — multiple attack implementations exist; automated cross-check not yet implemented.

**T-TAMP-02: Report JSON manipulation after generation**  
An attacker with filesystem access modifies the JSON report to change `risk_level` from `HIGH` to `LOW` before it reaches the compliance dashboard.  
- **Mitigation:** Reports should be signed (HMAC or asymmetric signature) before being written to disk or uploaded. Signature verification should occur at the compliance dashboard ingestion point.  
- **Status:** Not yet implemented. Planned.

**T-TAMP-03: Dependency substitution (supply chain)**  
A malicious version of `numpy` or `scikit-learn` introduced via `pip install` could silently alter epsilon calculations or MIA advantage values.  
- **Mitigation:** Pin exact dependency versions in `pyproject.toml`. Run `pip-audit` in CI. SARIF upload for dependency CVEs.  
- **ATT&CK v19:** T1195.001 (Supply Chain Compromise — Develop Toolchain, TA0005)  
- **Status:** Dependency pinning in place. `pip-audit` step in CI planned.

---

### 4.3 Repudiation

**T-REPUD-01: Denial of assessment execution**  
A team claims a privacy assessment was run when it was not, or that the assessment used a different (more favorable) model version.  
- **Mitigation:** CI pipeline logs assessment execution with model path, hash, and timestamp. Reports include `model_id` field. Store assessment reports in immutable artifact storage (e.g., S3 with object lock or GitHub Actions artifacts).  
- **Status:** CI logging in place. Immutable storage not enforced.

---

### 4.4 Information Disclosure

**T-INFO-01: Training data leakage via assessment artifacts**  
Running model inversion or attribute inference attacks generates intermediate artifacts (reconstructed inputs, inferred attributes) that may contain personal data.  
- **Impact:** Compliance violation if artifacts are written to shared storage or included in CI logs.  
- **Mitigation:** Intermediate attack artifacts are not persisted to disk by default. Only the aggregated JSON report is written. Do not run assessments on production personal data without a data processing agreement.  
- **Status:** Mitigated by design — no intermediate artifact persistence.

**T-INFO-02: Report leakage via CI logs**  
Privacy assessment reports logged to stdout in CI may be captured in public CI logs.  
- **Mitigation:** Use `--output-file` flag to write reports to artifacts rather than stdout. Mask report output in CI using secret masking where applicable.  
- **Status:** `--output-file` flag planned. Currently stdout only.

---

### 4.5 Denial of Service

**T-DOS-01: Adversarial model causing assessment hang**  
A model crafted with pathological output distributions (e.g., all-zero logits) could cause shadow model training loops to run indefinitely.  
- **Mitigation:** Apply timeouts to all attack execution. Default timeout: 60 seconds per attack primitive.  
- **Status:** Not yet implemented. Planned.

---

### 4.6 Elevation of Privilege

**T-EOP-01: CI coverage gate bypass**  
An attacker modifies `.github/workflows/ci.yml` to remove the `--cov-fail-under=80` gate, allowing privacy assessment code with inadequate test coverage to merge.  
- **Mitigation:** Protect the default branch with required status checks. Require code owner review for workflow file changes.  
- **ATT&CK v19:** T1078 (Valid Accounts — CI/CD abuse, TA0005 Stealth)  
- **Status:** Branch protection recommended. Not enforced at repo level currently.

---

## 5. Residual Risks

| Threat ID | Residual Risk | Accepted? | Rationale |
|-----------|--------------|-----------|-----------|
| T-TAMP-01 | Adversarial model masking true risk | Accepted (partial) | Multi-attack approach reduces but does not eliminate this risk |
| T-TAMP-02 | Report manipulation post-generation | Not accepted | Report signing is on the roadmap |
| T-INFO-02 | Report leakage via CI logs | Accepted (low) | Assess only synthetic/toy data in CI; production assessments run locally |
| T-DOS-01 | Assessment hang | Accepted (low) | Affects only the assessor's local environment |

---

## 6. ATT&CK v19 Coverage

Threats above map to the following v19 techniques:

| Technique | ID | Tactic | Threat |
|-----------|----|--------|--------|
| Masquerading | T1036 | TA0005 Stealth | T-SPOOF-01 |
| ML Model Obfuscation | T1027/018 | TA0005 Stealth | T-TAMP-01 |
| Supply Chain Compromise — Toolchain | T1195.001 | TA0005 Stealth | T-TAMP-03 |
| Valid Accounts (CI/CD) | T1078 | TA0005 Stealth | T-EOP-01 |

All IDs validated against `V19_REVOCATION_MAP`. No deprecated technique IDs used.

---

## 7. Assumptions and Limitations

- This threat model covers the assessment framework itself, not the models or datasets being assessed.
- The framework does not connect to external networks during assessment. If a model's `forward()` method makes external calls, that is out of scope here and should be addressed by the model owner.
- All epsilon calculations are approximations. See `budget_calculator.py` docstring for the exact approximation used and its limitations versus the PRV accountant.
- This threat model was authored by the repository maintainer and has not undergone independent third-party review.

---

*See [SECURITY.md](SECURITY.md) for vulnerability disclosure procedures.*
