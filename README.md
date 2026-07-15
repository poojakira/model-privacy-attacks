# model-privacy-attacks

A research toolkit for **measuring privacy leakage** from machine-learning
models: membership-inference attacks (MIA) and model-extraction attacks, with an
extension for large language models (Min-K% Prob).

> **Honesty note.** Every metric in this README is produced by the committed,
> seed-pinned test suite (`tests/test_privacy_attacks.py`) on **synthetic data
> (seed 42)**. These numbers measure *implementation correctness*, not
> real-world privacy leakage. No number appears here that was not measured by a
> test in this repository. Dependency ranges are bounded to NumPy <3 and scikit-learn <2 to reduce major-version metric drift; this is not a lockfile.

---

## Contents

1. [Installation](#installation)
2. [Attacks](#attacks)
   - [Direct MIA](#direct-mia)
   - [Shadow-Model MIA](#shadow-model-mia)
   - [Model Extraction](#model-extraction)
   - [LLM MIA — Min-K% Prob](#llm-mia--min-k-prob)
3. [Measured Results](#measured-results-synthetic-seed-42)
4. [Reproducing](#reproducing)
5. [Regulatory Context](#regulatory-context)
6. [Project Structure](#project-structure)
7. [References](#references)

---

## Installation

```bash
git clone https://github.com/poojakira/model-privacy-attacks.git
cd model-privacy-attacks
python -m pip install -e .            # core: numpy + scikit-learn only
python -m python -m pip install -e ".[test]"     # test runner
```

The core library requires **only numpy and scikit-learn**. The test suite adds pytest only through the `test` extra —
no model downloads. The LLM demo can optionally use real GPT-2 log-probs:

```bash
python -m pip install -e ".[llm]"     # optional: transformers + torch
```

---

## Attacks

### Direct MIA

The **direct attack** (Shokri et al., 2017) exploits the tendency of a
classifier to assign higher confidence to samples it was trained on than to
held-out samples. A single threshold on the model's confidence for the true
class separates predicted members from non-members.

```python
from privacy_attacks.mia import DirectMIA

attack = DirectMIA(use_true_label=True)
attack.fit(target_model, X_members, y_members, X_nonmembers, y_nonmembers)
metrics = attack.evaluate(X_members, X_nonmembers, y_members, y_nonmembers)
print(metrics["auc"])
```

### Shadow-Model MIA

The **shadow-model attack** (Shokri et al., 2017) trains several shadow models
on public data of the same distribution as the target's training set. Because
the attacker controls each shadow's in/out split, it can build a labelled
`(confidence-vector -> member?)` dataset and train an **attack classifier** that
is then applied to the target's outputs.

```python
from privacy_attacks.mia import ShadowMIA

attack = ShadowMIA(n_shadow=4, shadow_model_cls="RandomForest", random_state=42)
attack.fit(X_public, y_public, target_model)
metrics = attack.evaluate(X_members, X_nonmembers)
```

### Model Extraction

Model extraction (Tramèr et al., 2016) reconstructs a functional copy of a
target model by querying it and training a local **substitute** on the
`(input, predicted-label)` pairs. Fidelity is reported as **agreement** — the
fraction of held-out inputs on which substitute and target predict the same
label.

```python
from privacy_attacks.extraction import ModelExtractionAttack

attack = ModelExtractionAttack(substitute_model_cls="DecisionTree", random_state=42)
attack.fit(target_model, X_query)
agreement = attack.agreement(target_model, X_eval)
```

### LLM MIA — Min-K% Prob

**Min-K% Prob** (Shi et al., 2024) is a reference-free, black-box membership
inference method for causal LMs. It needs only the **per-token log-probabilities**
that any causal LM can provide — no gradients, no shadow models, no reference
corpus.

**Core intuition:** for text seen during pre-training, the model assigns
relatively high log-probability even to the *rarest* tokens; for unseen text the
rare-token probabilities are lower. The score is the mean log-probability of the
`k%` lowest-probability tokens:

```
score(x) = mean{ log p(t_i) : t_i in the k%-lowest-prob tokens of x }
```

Higher (less negative) → predicted **member**; lower → **non-member**.

```python
from privacy_attacks.llm_mia import TokenLikelihoodMIA, TokenLikelihoodMIAConfig

config = TokenLikelihoodMIAConfig(k_percent=0.20, min_tokens=10, device="cpu")
mia = TokenLikelihoodMIA(config=config, threshold=-3.0)

result = mia.predict_from_log_probs(text="Your text here", token_log_probs=[...])
print(result.predicted_member, result.confidence)

metrics = mia.evaluate_auc(member_results, nonmember_results)
```

A runnable demo (synthetic by default, GPT-2 with `--real`) is in
`examples/llm_mia_demo.py`:

```bash
python examples/llm_mia_demo.py            # synthetic log-probs, no download
python examples/llm_mia_demo.py --real     # real GPT-2 (needs .[llm])
```

#### Extracting per-token log-probs from a HuggingFace model

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2").eval()

input_ids = tokenizer("The quick brown fox.", return_tensors="pt")["input_ids"]
with torch.no_grad():
    logits = model(input_ids).logits
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
token_log_probs = [
    log_probs[0, i - 1, input_ids[0, i]].item()
    for i in range(1, input_ids.shape[1])
]
```

---

## Measured Results (synthetic, seed 42)

All figures below are **reproduced by `tests/test_privacy_attacks.py`** and
measure implementation correctness on synthetic data with `random_state=42`.
The baseline for every AUC is 0.50 (random guessing).

| Attack | Metric | Value | Setup |
|--------|--------|:-----:|-------|
| Direct MIA | ROC AUC | **0.7709** | sklearn RandomForest target (100 trees), true-class confidence, 1000 member / 1000 non-member, 20 features |
| Direct MIA | Accuracy | 0.698 | threshold selected on reference non-members |
| Shadow MIA | ROC AUC | **0.7680** | RandomForest target + 4 RandomForest shadows, RandomForest attack classifier, 1000 / 1000 |
| Shadow MIA | Accuracy | 0.6905 | 0.5 decision threshold |
| Model Extraction | Agreement | **0.892** | RandomForest target, DecisionTree substitute, 2000 query / 500 eval samples |
| LLM MIA (Min-20% Prob) | ROC AUC | **0.9599** | synthetic per-token log-probs, 100 member / 100 non-member texts, 50 tokens each, overlapping Gaussians |
| LLM MIA (Min-20% Prob) | TPR @ 1% FPR | 0.47 | same synthetic set |

> The LLM MIA figure is a correctness check of the Min-K% Prob scoring and AUC
> logic on synthetic log-probs; it is **not** a benchmark of any real model.
> For published results on real models, see Shi et al. (2024) on the
> [WikiMIA benchmark](https://huggingface.co/datasets/swj0419/WikiMIA)
> (e.g. GPT-NeoX-20B ≈ 0.69 AUC) — **not reproduced in this repository**
> because it requires model downloads outside this toolkit's scope.

---

## Reproducing

```bash
python -m pip install -e ".[test]"
pytest tests/test_privacy_attacks.py -v
# or see the metrics printed on stdout:
pytest tests/test_privacy_attacks.py -s
```

The suite is fully deterministic (`SEED = 42`); the numbers above are the exact
values it emits. Re-running regenerates them.

---

## Regulatory Context

### EU AI Act — Article 10 (Data Governance)

Article 10 requires providers of high-risk AI systems to implement data
governance practices covering training-data characteristics, provenance, and
bias detection. Membership inference is directly relevant: an MIA that succeeds
with AUC well above 0.5 demonstrates that training data can be re-identified
from a deployed model, with implications under GDPR Article 5(1)(f) and Article
25. Teams should treat MIA evaluation as part of pre-deployment conformity
assessment — documenting the attack surface, the measured AUC on representative
held-out data, and any mitigations (differential privacy, deduplication, output
perturbation).

### NIST AI RMF — GOVERN-1.6 (Privacy Risk Documentation)

GOVERN-1.6 calls for policies that explicitly address privacy risks from AI
development and deployment, documented (not assumed absent) and tracked across
the lifecycle. Membership inference and model extraction translate the abstract
risk of "training-data leakage" into measurable quantities (AUC, agreement) that
belong in risk registers and model cards. Run these attacks as part of the MAP
and MEASURE functions and record results with the provenance labels used above.

---

## Project Structure

```
model-privacy-attacks/
├── src/privacy_attacks/
│   ├── __init__.py
│   ├── mia/
│   │   ├── direct_mia.py          # DirectMIA
│   │   └── shadow_mia.py          # ShadowMIA
│   ├── extraction/
│   │   └── extraction_attack.py   # ModelExtractionAttack
│   └── llm_mia/
│       └── token_likelihood_mia.py # TokenLikelihoodMIA (Min-K% Prob)
├── examples/
│   └── llm_mia_demo.py            # synthetic + optional GPT-2 demo
├── tests/
│   └── test_privacy_attacks.py    # seed-42 synthetic regression tests
├── pyproject.toml
└── README.md
```

---

## References

- Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017). Membership
  inference attacks against machine learning models. *IEEE S&P*.
  <https://arxiv.org/abs/1610.05820>
- Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T. (2016).
  Stealing machine learning models via prediction APIs. *USENIX Security*.
  <https://arxiv.org/abs/1609.02943>
- Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D., &
  Zettlemoyer, L. (2024). Detecting pretraining data from large language models.
  *ICLR*. <https://arxiv.org/abs/2310.16789>
- EU Artificial Intelligence Act (2024). Regulation (EU) 2024/1689, Article 10.
  <https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689>
- NIST (2023). AI Risk Management Framework (AI RMF 1.0), GOVERN-1.6.
  <https://airc.nist.gov/Docs/1>
