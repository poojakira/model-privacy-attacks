# Security Audit — model-privacy-attacks

**Auditor:** agent/security-hardening-v1 (strictness 10/10)  
**Date:** 2026-08-06  
**Scope:** All source files, tests, workflows, documentation

---

## 1. Repository Classification

**EDUCATIONAL IMPLEMENTATION** — attacks on synthetic/toy data only.  
Not a production security tool. All reported metrics are from seed-42 synthetic numpy data.

---

## 2. Implemented Capabilities

| Capability | Status | Evidence |
|-----------|--------|---------|
| Yeom (2018) MIA | ✅ IMPLEMENTED_AND_TESTED | `tests/test_privacy_attacks.py` |
| Shokri shadow model MIA | ✅ IMPLEMENTED_AND_TESTED | `tests/test_privacy_attacks.py` |
| LiRA (Carlini et al. S&P 2022) | ✅ IMPLEMENTED_AND_TESTED | `tests/test_privacy_attacks.py` |
| Direct confidence MIA | ✅ IMPLEMENTED_AND_TESTED | `tests/test_privacy_attacks.py` |
| Fredrikson model inversion | ✅ IMPLEMENTED_AND_TESTED | `tests/test_privacy_attacks.py` |
| DP-SGD RDP accounting | ✅ IMPLEMENTED_AND_TESTED | `results/epsilon_verification.json` |
| Compliance report | ✅ IMPLEMENTED | `src/privacy_attacks/compliance_report.py` |
| AWS deployment | ❌ NOT IMPLEMENTED | Educational tool only |

---

## 3. Critical Findings

None — this is a read-only analytical tool with no external API or network exposure.

---

## 4. High Findings

### HIGH-1 (FIXED): EU AI Act "triggered_articles" language implied legal determination

**File:** `src/privacy_attacks/compliance_report.py`  
**Severity:** High — claim integrity  
**Root cause:** JSON output key `triggered_articles` and field `action_required` implied a legally binding compliance assessment. This is not supportable from a research tool.  
**Fix:** Renamed to `reference_articles` and `review_recommended`. Added `disclaimer` field explicitly stating this is informational only.

---

## 5. Medium Findings

### MEDIUM-1 (OPEN): No `dependabot.yml`

**Severity:** Medium  
**Status:** Needs addition for automated dependency updates.

### MEDIUM-2 (OPEN): MIA advantage baseline explanation needs prominent placement

**Severity:** Medium  
**Root cause:** README's claim "MIA advantage = 0.42 (baseline = 0.10)" requires the reader to understand that the baseline is 0.10 due to class imbalance, not the theoretical 0.0. This is explained but could mislead readers who skim.  
**Status:** Documented in README; `compliance_report.py` includes explicit `random_baseline_advantage` parameter and interpretation string.

---

## 6. Low Findings

### LOW-1: No CI coverage gate

**Severity:** Low  
**Status:** Tests run but no `--cov-fail-under` enforced.

---

## 7. Unsupported / Corrected Claims

| Original Claim | Status | Correction |
|---------------|--------|-----------|
| "EU AI Act articles triggered" | REMOVED | Replaced with "reference_articles" + disclaimer |
| "epsilon=1.16 at sigma=4.0" | VERIFIED | `results/epsilon_verification.json` confirms 1.1623 |
| "MIA advantage = 0.42" | VERIFIED on synthetic data | Random baseline 0.10, not 0.0 |
| "CIFAR-10 benchmark" | NOT IMPLEMENTED | All results are on seed-42 synthetic numpy data |

---

## 8. Evidence Classification

All results are `SYNTHETIC_SMOKE_RESULT` — generated against seed-42 numpy synthetic data, not a real production dataset. See `evidence_policy.json` for machine-readable classification.

---

## 9. Verification Commands

```bash
pip install -e ".[dev]"
pytest tests/ -v
python scripts/verify_epsilon.py
```
