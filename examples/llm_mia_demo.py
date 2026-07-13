"""End-to-end Min-K% Prob demo.

By default this runs on **synthetic** per-token log-probs so it needs only
numpy + scikit-learn (no model download).  If ``transformers`` and ``torch`` are
installed, pass ``--real`` to score real GPT-2 log-probs instead.

    python examples/llm_mia_demo.py            # synthetic (default)
    python examples/llm_mia_demo.py --real     # GPT-2 (needs `pip install .[llm]`)

The synthetic numbers measure *implementation correctness* of the Min-K% Prob
scoring/AUC logic, not real-world memorisation.
"""

from __future__ import annotations

import argparse

import numpy as np

from privacy_attacks.llm_mia import TokenLikelihoodMIA, TokenLikelihoodMIAConfig

SEED = 42


def synthetic_log_probs(n_texts, tokens_per_text, mean, std, rng):
    out = []
    for _ in range(n_texts):
        logp = np.minimum(rng.normal(mean, std, tokens_per_text), -1e-6)
        out.append(logp.tolist())
    return out


def run_synthetic() -> None:
    rng = np.random.default_rng(SEED)
    mia = TokenLikelihoodMIA(
        config=TokenLikelihoodMIAConfig(k_percent=0.20, min_tokens=10),
        threshold=-3.0,
    )
    members = synthetic_log_probs(100, 50, mean=-2.8, std=1.5, rng=rng)
    nonmembers = synthetic_log_probs(100, 50, mean=-3.6, std=1.5, rng=rng)

    member_results = [mia.predict_from_log_probs(f"m{i}", lp) for i, lp in enumerate(members)]
    non_results = [mia.predict_from_log_probs(f"n{i}", lp) for i, lp in enumerate(nonmembers)]

    metrics = mia.evaluate_auc(member_results, non_results, dataset="synthetic")
    print("Min-K% Prob on synthetic log-probs (seed 42):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")


def run_real() -> None:  # pragma: no cover - requires optional heavy deps
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    mia = TokenLikelihoodMIA(
        config=TokenLikelihoodMIAConfig(k_percent=0.20, min_tokens=10),
        threshold=-3.0,
    )

    def token_log_probs(text: str) -> list[float]:
        input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
        with torch.no_grad():
            logits = model(input_ids).logits
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return [
            log_probs[0, i - 1, input_ids[0, i]].item()
            for i in range(1, input_ids.shape[1])
        ]

    text = "The quick brown fox jumps over the lazy dog."
    result = mia.predict_from_log_probs(text, token_log_probs(text))
    print(f"text={text!r}")
    print(f"  score={result.score:.4f} member={result.predicted_member} "
          f"confidence={result.confidence:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real", action="store_true",
        help="Score real GPT-2 log-probs (requires transformers + torch).",
    )
    args = parser.parse_args()
    if args.real:
        run_real()
    else:
        run_synthetic()


if __name__ == "__main__":
    main()
