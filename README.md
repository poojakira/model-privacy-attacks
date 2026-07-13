# model-privacy-attacks

> ## Status: SPECIFICATION ONLY — not yet implemented
> This repository currently contains **only this design document**. There is no
> `src/` package, no `tests/`, and no `pyproject.toml` yet. Nothing here has been
> benchmarked. Any performance figures added later must be reproducible from a
> committed test on a fixed seed — no numbers are claimed until then.

A planned research toolkit for **measuring privacy leakage** from machine-learning
models: membership-inference attacks (MIA) and model-extraction attacks, with an
extension for large language models.

## Planned scope

| Attack | Target | Reference |
|--------|--------|-----------|
| Direct MIA (confidence thresholding) | sklearn classifiers | Shokri et al. (2017) |
| Shadow-model MIA | sklearn classifiers | Shokri et al. (2017) |
| Model extraction (substitute training) | sklearn classifiers | Tramèr et al. (2016) |
| Min-K% Prob MIA | LLMs (token log-probs) | Shi et al. (2024) |

## Design principles (for the eventual implementation)

- **Reproducibility first.** Every reported metric must be produced by a committed
  test on synthetic data with a fixed random seed, so CI can re-verify it. No
  metric appears in this README until such a test exists.
- **Black-box only.** Attacks assume query access to the target's outputs, not its
  weights or gradients.
- **Honest labelling.** Synthetic-data results measure implementation correctness,
  not real-world privacy risk, and will be labelled as such.

## Roadmap

1. `src/privacy_attacks/mia/` — direct + shadow membership inference on sklearn targets.
2. `src/privacy_attacks/extraction/` — substitute-model extraction + agreement metric.
3. `src/privacy_attacks/llm_mia/` — Min-K% Prob token-likelihood MIA.
4. `tests/` — synthetic, seed-pinned regression tests that produce the headline metrics.

## References

- Shokri et al. (2017), *Membership Inference Attacks Against ML Models*, IEEE S&P. <https://arxiv.org/abs/1610.05820>
- Tramèr et al. (2016), *Stealing ML Models via Prediction APIs*, USENIX Security. <https://arxiv.org/abs/1609.02943>
- Shi et al. (2024), *Detecting Pretraining Data from LLMs*, ICLR. <https://arxiv.org/abs/2310.16789>
