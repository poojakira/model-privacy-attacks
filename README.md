# model-privacy-attacks

> The question isn't whether your model leaks — it's *how much*.

A small, fully-offline, **deterministic** simulator for the two classic families of
inference-time privacy attacks: **membership inference** and **model extraction**. No
model downloads, no API keys, no internet. Every documented number is reproducible on a
laptop with `pytest`.

**Membership inference**, in one sentence: given a trained model and a data sample, an
attacker tries to decide whether that exact sample was in the model's training set.

## Why this exists

The attacks themselves are well understood. What is not well understood — and what the
2025-26 literature keeps flagging — is how to *evaluate* them honestly:

- [Membership Inference Attacks on LLMs are Rushing Nowhere](https://arxiv.org/html/2406.17975v3)
  (SaTML 2025): results are reported on post-hoc datasets without randomized controls.
- [Blind Baselines Beat Membership Inference Attacks for Foundation Models](https://arxiv.org/html/2406.16201v1):
  trivial baselines outperform sophisticated attacks on popular benchmarks.
- [Revisiting the LiRA Membership Inference Attack Under Realistic Assumptions](https://arxiv.org/html/2603.07567):
  prior work **overstated** effectiveness by calibrating thresholds on the target data,
  assuming balanced membership priors, and overlooking reproducibility.

This project takes the opposite stance on purpose: tiny synthetic problems, fixed seeds,
threshold calibration kept separate from the samples being scored, and metrics reported
the way the community now expects (AUC **and** TPR@low-FPR).

_Summaries above were rephrased from the cited sources for licensing compliance._

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## What's inside

| Attack | Module | Signal it exploits |
| --- | --- | --- |
| Confidence threshold | `membership_inference/threshold_attack.py` | Models are more confident on training data |
| Prediction entropy | `membership_inference/entropy_attack.py` | Members produce lower-entropy softmax vectors |
| Shadow models | `membership_inference/shadow_model_attack.py` | Imitation models teach an attack classifier |
| Boundary stealing | `model_extraction/boundary_stealing.py` | A prediction API is a free labelling oracle |

## Quick start

```python
import numpy as np
from privacy_attacks import ThresholdAttack, attack_auc, membership_inference_report

rng = np.random.default_rng(42)
member_scores    = np.clip(rng.normal(0.85, 0.08, 500), 0, 1)   # confident on training data
nonmember_scores = np.clip(rng.normal(0.60, 0.12, 500), 0, 1)   # less sure on unseen data

auc = attack_auc(member_scores, nonmember_scores)
print(membership_inference_report("threshold", auc, advantage=0.12))
```

Running the above prints an AUC of **0.9637** on this synthetic split — comfortably
above the documented `AUC > 0.6` guarantee that the test suite enforces
(`tests/test_threshold.py::test_auc_above_0_6_on_synthetic_data`).

## What an AUC of 0.5 means vs 0.8 means

AUC is the probability that a randomly chosen **member** is scored more "member-like"
than a randomly chosen **non-member**. Read it as a leakage dial, not a grade:

- **0.50** — a coin flip. The attacker learns *nothing*; members and non-members are
  indistinguishable. This is what a well-defended model should look like.
- **0.60** — weak but real. Over many samples the attacker does better than chance, so
  the model is leaking a little. Our threshold baseline lives here on hard problems.
- **0.80** — serious. Four times out of five the attacker ranks the true member higher.
  Membership is genuinely recoverable and the model is memorizing.
- **0.99** — the model has effectively published its training-set membership.

The catch: **AUC averages over the whole ROC curve**, so a model can post a "safe"
0.55 AUC while still letting an attacker identify a *handful* of members with near
certainty. That is why the modern metric is **TPR@low-FPR** — the true-positive rate
when you cap false positives at, say, 0.1%. A single confidently-exposed member can be a
privacy incident even when the average looks fine. `metrics.attack_advantage` reports
exactly this low-FPR regime, and `compute_auc` gives you the manual trapezoid AUC to
plot the full curve.

See [`INCIDENT_REPORT.md`](INCIDENT_REPORT.md) for a full reproducible audit run,
including the key finding that shadow-model leakage tracks overfitting (AUC 0.51 for a
model that generalizes vs 0.75 for one that memorizes).

## Ethics & scope

This is defensive tooling for privacy auditing and education. Run it against models you
own or are authorized to test, to measure and then *reduce* leakage.

## License

MIT
