# The Privacy Certificate

> A raw attack score tells you a number. A certificate tells you whether to believe it.

Most membership-inference tooling stops at "the attack got AUC 0.7." The 2025-26
literature is unambiguous that this number is frequently untrustworthy: it is inflated by
distribution artifacts ([Blind Baselines Beat MIAs](https://arxiv.org/html/2406.16201v1)),
by calibrating thresholds on the target ([Revisiting LiRA](https://arxiv.org/html/2603.07567)),
and by non-reproducibility ([MIAs are Rushing Nowhere](https://arxiv.org/html/2406.17975v3)).

The Privacy Certificate turns a raw attack signal into a **null-calibrated, reproducible,
gate-able verdict**. It is the difference between a research demo and something you can
put in a release pipeline.

## What it answers

1. **Does the model leak beyond a blind baseline?** We build the strongest "blind"
   attacker we can that never queries the model and looks only at the inputs (a panel of
   logistic regression, random forest, and kNN, evaluated out-of-fold). Its AUC is the
   floor. Leakage is the *difference* between the model-based attack and this floor.
2. **How confident are we?** A paired bootstrap resamples the same indices for both the
   model attack and the blind baseline, producing a confidence interval on the AUC
   difference and a one-sided p-value.
3. **Was the certificate altered?** A SHA-256 content checksum detects corruption;
   setting `PRIVACY_CERT_HMAC_KEY` upgrades it to a keyed HMAC signature.

## The verdict rule (stated plainly)

A single, honest, one-sided bootstrap test at level `alpha` (default 0.05):

- **CERTIFIED_LEAKAGE** — the model out-separates the strongest blind baseline with
  `p < alpha` and a positive delta. Leakage is attributable to the model.
- **NO_MODEL_LEAKAGE** — the `(1 - alpha)`-level delta-AUC CI lies entirely at or below
  zero. Any apparent separation is a distribution artifact.
- **INCONCLUSIVE** — not enough evidence either way at this sample size.

We do **not** dress this up as multiple independent tests; it is one test, and its
false-positive rate under the null was measured at ~2-3% (i.e. conservative at α=0.05).

## Threat model & honest limitations

**What it is:** a defensive auditing tool. It answers "does *this* attack signal, on
*this* member/non-member split, reflect model memorization rather than a dataset
artifact, with quantified uncertainty?"

**What it is not, and where it can be wrong:**

- **The verdict is only as strong as the blind-baseline panel.** If a real membership
  artifact is derivable from the inputs by some function *none* of the panel models can
  represent, the engine can still over-attribute leakage to the model. Mitigation:
  the panel includes a non-linear learner (random forest); you can extend it. The chosen
  baseline and its AUC are always reported so a weak floor is visible.
- **The blind baseline's own estimation variance is not fully propagated.** The paired
  bootstrap resamples data but reuses the once-fit out-of-fold baseline scores. This can
  make the delta CI slightly optimistic. Documented, not hidden.
- **The low-FPR (TPR@fpr) metric requires enough non-members.** Below `ceil(5 / fpr)`
  non-members the target FPR is unresolvable; the engine marks the metric **unavailable**
  and omits it from the verdict rather than printing a misleading tail number.
- **The unsigned checksum is corruption detection, not tamper-evidence.** Anyone can
  recompute a SHA-256 over public fields. Tamper-evidence requires the HMAC key. When a
  key is present at verification time, `verify_integrity` *requires* a valid HMAC and
  fails closed (no downgrade to an unsigned certificate).
- **This certifies an *attack outcome*, not a formal privacy guarantee.** It is not a
  differential-privacy proof. Causal `(ε)`-style auditing via injected canaries is the
  planned next layer (see roadmap).

## Roadmap (deferred, on purpose)

- **Causal ε-auditing** with injected canaries and single-training-run auditing
  ([Steinke, Nasr & Jagielski, 2023](https://arxiv.org/abs/2305.08846)) to emit an
  empirical differential-privacy lower bound.
- **Framework adapters** for HuggingFace / ONNX / remote prediction APIs (the
  `TargetModel` interface is stable today; only the callable adapter ships now).
- **Privacy regression tracking**: diff certificates across model versions in CI.
