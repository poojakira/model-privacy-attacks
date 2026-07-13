# model-privacy-attacks

> **Status: Work In Progress (scaffold only).**
> This repository is a placeholder. There is **no runnable code or test suite
> here yet** — do not rely on any capability described below until it lands.

Planned: a small, reproducible lab of **privacy attacks against ML models**,
intended for defensive research and evaluation on models you own or are
authorized to test.

## Intended scope (not yet implemented)

- **Membership inference** — determine whether a record was in the training set.
- **Model inversion** — reconstruct representative training features.
- **Attribute inference** — infer sensitive attributes from model outputs.
- **Evaluation harness** — ROC/AUC-based metrics rather than single-point claims.

## Current state

- No source modules, no tests, no packaging.
- Any test counts or coverage figures referenced elsewhere (e.g. a profile
  README) are **aspirational** and do not reflect this repository yet.

## If you are evaluating this repo

Treat it as **early WIP**. If it is not being actively developed it should be
marked **Archived** on GitHub to set expectations. Track progress via issues
and milestones before it is considered usable.

## Responsible use

When implemented, these techniques are for auditing the privacy of **your own**
or explicitly authorized models. Do not use them against systems you do not
have permission to test.


## Novelty & honesty (be skeptical)

There is nothing novel here yet — there is no code. When implemented, the
planned techniques (membership inference, model inversion, attribute inference)
are **well-established academic methods, not original research**; the value
would be a clean, reproducible harness, not new attacks.

To be credible, a privacy-attack tool must report against a **shared benchmark**
(e.g. an MIA/MLPrivacy auditing framework with known-member / known-nonmember
splits and DP-SGD models at known ε), because privacy-leakage numbers are
meaningless without a common baseline. Until that exists here, treat any claim
as aspirational.
