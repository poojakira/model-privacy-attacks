# Privacy Attack Simulation — Incident Report

| Field | Value |
| --- | --- |
| Report ID | MPA-2026-0711-01 |
| Date | 2026-07-11 |
| Author | poojakira |
| Classification | Internal / Defensive research |
| Tooling | `model-privacy-attacks` v0.1.0 |
| Environment | Python 3.12, scikit-learn ≥ 1.5, numpy ≥ 1.26 (offline, deterministic) |
| Targets | Synthetic classifiers trained locally (no third-party models) |

> **Scope note.** This is a *simulation* run against models trained on synthetic data
> for the purpose of measuring privacy leakage. No production system, real user data, or
> third-party model was accessed. It is an audit exercise, not a report of a real-world
> breach.

## 1. Summary

We ran the four attacks bundled in `model-privacy-attacks` against locally trained
target models to quantify how much they leak about their training data. Two findings are
material:

1. **Confidence and entropy signals leak strongly** on the synthetic targets
   (membership AUC 0.96 and 0.89 respectively).
2. **Shadow-model leakage is entirely driven by overfitting.** A well-generalizing
   linear model leaks nothing (AUC ≈ 0.51, indistinguishable from a coin flip), while an
   overfitting gradient-boosting model leaks clearly (AUC ≈ 0.75).

A separate model-extraction test showed a black-box classifier could be cloned to
**100% decision fidelity with 2,000 queries**.

## 2. Findings

### 2.1 Membership inference — confidence threshold (HIGH)

| Metric | Value |
| --- | --- |
| Attack AUC | **0.9637** |
| Optimal threshold | 0.7202 |
| Advantage (TPR @ 0.1% FPR) | 0.1060 |
| Verdict | Severe leakage — membership is highly recoverable |

An attacker observing only the maximum confidence score distinguishes members from
non-members with high reliability. Note the split between the average metric and the
low-FPR regime: even where AUC is very high, the strict TPR@0.1%FPR (≈10.6%) is the
number that matters for a real privacy incident, because it measures how many members an
attacker can expose with almost no false alarms.

### 2.2 Membership inference — prediction entropy (HIGH)

| Metric | Value |
| --- | --- |
| Attack AUC | **0.8889** |
| Mean entropy, members | 0.636 nats |
| Mean entropy, non-members | 0.947 nats |

Members produce visibly lower-entropy (more peaked) output distributions. The entropy
gap is a stable, exploitable signal.

### 2.3 Membership inference — shadow models (CONDITIONAL — depends on overfitting)

| Target model | Behaviour | Shadow attack AUC |
| --- | --- | --- |
| Logistic regression | Generalizes | **0.5143** (no leakage) |
| SVC (RBF) | Mild overfit | 0.6160 |
| MLP (64 hidden) | Overfit | 0.5905 |
| Gradient boosting | Memorizes | **0.7538** (clear leakage) |

**This is the most important finding.** The shadow attack does not "work" or "fail" in
the abstract — it recovers exactly the amount of leakage the target model creates by
memorizing. A model that generalizes is genuinely private against this attack. This
mirrors the 2025-26 literature warning that reported MI results are often an artifact of
overfit targets ([Blind Baselines Beat MIAs](https://arxiv.org/html/2406.16201v1);
[Revisiting LiRA](https://arxiv.org/html/2603.07567)).

### 2.4 Model extraction — decision-boundary stealing (HIGH)

Fidelity (agreement between the stolen clone and the black-box oracle) versus query
budget:

| Queries | Fidelity |
| --- | --- |
| 10 | 0.8783 |
| 50 | 0.9517 |
| 200 | 0.9617 |
| 1,000 | 0.9867 |
| 2,000 | **1.0000** |

A prediction API with no rate limiting is a free labelling oracle: ~2,000 queries were
enough to reconstruct the oracle's decisions perfectly with a logistic-regression clone.

## 3. Severity assessment

| Attack | Severity | Rationale |
| --- | --- | --- |
| Confidence threshold MI | High | AUC 0.96; trivial to run with only output scores |
| Entropy MI | High | AUC 0.89; needs only the softmax vector |
| Shadow-model MI | Medium (High if target overfits) | Leakage tracks overfitting; 0.51 → 0.75 |
| Boundary stealing | High | 100% fidelity clone from black-box queries |

## 4. Recommendations

- **Reduce overfitting.** The single most effective defence against MI here was
  generalization (regularization, early stopping, more data, dropout).
- **Do not expose raw confidence/probability vectors.** Return top-1 labels, or coarsen
  / round probabilities where the full vector is not required.
- **Rate-limit and monitor prediction APIs** to blunt extraction; watch for the
  broad, boundary-hugging query patterns extraction attacks generate.
- **Consider DP-SGD / differential privacy** where the training data is sensitive.
- **Audit with TPR@low-FPR, not just AUC.** A "safe-looking" average can still expose a
  handful of members with near certainty.

## 5. Reproduction

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest        # 12/12 tests pass, incl. the AUC > 0.6 and fidelity > 0.7 guarantees
```

All numbers above are deterministic (fixed seeds) and reproduce on any machine with the
pinned dependencies — no network, no model downloads.
