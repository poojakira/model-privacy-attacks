"""Min-K% Prob membership inference for causal language models.

Min-K% Prob (Shi et al., 2024) is a reference-free, black-box membership
inference method.  It requires only the per-token log-probabilities that any
causal LM can provide -- no gradients, no shadow models, no reference corpus.

Core intuition
--------------
For text seen during pre-training, the model assigns relatively high
log-probability even to the *rarest* tokens in the sequence.  For unseen text,
those rare-token probabilities are lower.  The score is therefore the mean
log-probability of the ``k%`` lowest-probability tokens::

    score(x) = mean{ log p(t_i) : t_i in the k%-lowest-prob tokens of x }

A higher (less negative) score  -> predicted *member*.
A lower (more negative)  score  -> predicted *non-member*.

Reference
---------
Shi, W., Ajith, A., Xia, M., Huang, Y., Liu, D., Blevins, T., Chen, D., &
Zettlemoyer, L. (2024). Detecting pretraining data from large language models.
ICLR. https://arxiv.org/abs/2310.16789
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score


@dataclass
class TokenLikelihoodMIAConfig:
    """Configuration for :class:`TokenLikelihoodMIA`.

    Parameters
    ----------
    k_percent:
        Fraction (in ``(0, 1]``) of lowest-probability tokens to average over.
        ``0.20`` reproduces the "Min-20% Prob" setting.
    min_tokens:
        Minimum number of tokens a text must have to be scored.
    device:
        Informational only; retained for API compatibility with LLM backends.
    """

    k_percent: float = 0.20
    min_tokens: int = 10
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not 0.0 < self.k_percent <= 1.0:
            raise ValueError("k_percent must be in (0, 1].")
        if self.min_tokens < 1:
            raise ValueError("min_tokens must be >= 1.")


@dataclass
class MinKProbResult:
    """Result of scoring a single text.

    Attributes
    ----------
    text:
        The scored text (may be empty if only log-probs were supplied).
    score:
        The Min-K% Prob score (mean log-prob of the k% lowest tokens).
    predicted_member:
        Whether ``score >= threshold``.
    confidence:
        A monotonic ``[0, 1)`` confidence derived from the score's margin above
        the threshold; useful for ranking, not calibrated as a probability.
    n_tokens:
        Number of per-token log-probs used.
    n_min_tokens:
        Number of lowest-probability tokens averaged for the score.
    """

    text: str
    score: float
    predicted_member: bool
    confidence: float
    n_tokens: int
    n_min_tokens: int
    extra: dict = field(default_factory=dict)


class TokenLikelihoodMIA:
    """Min-K% Prob membership-inference attack from per-token log-probs.

    Parameters
    ----------
    config:
        A :class:`TokenLikelihoodMIAConfig`.  Defaults are used if omitted.
    threshold:
        Decision threshold on the Min-K% Prob score.  Scores at or above this
        value are predicted to be members.
    """

    def __init__(
        self,
        config: Optional[TokenLikelihoodMIAConfig] = None,
        threshold: float = -3.0,
    ) -> None:
        self.config = config or TokenLikelihoodMIAConfig()
        self.threshold = float(threshold)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def min_k_score(self, token_log_probs: Sequence[float]) -> float:
        """Compute the Min-K% Prob score from a sequence of token log-probs."""
        logp = np.asarray(list(token_log_probs), dtype=float)
        if logp.size < self.config.min_tokens:
            raise ValueError(
                f"Need at least {self.config.min_tokens} tokens, " f"got {logp.size}."
            )
        k = max(1, int(np.ceil(self.config.k_percent * logp.size)))
        # The k smallest (lowest-probability) log-probs.
        lowest = np.sort(logp)[:k]
        return float(np.mean(lowest))

    def _n_min_tokens(self, n_tokens: int) -> int:
        return max(1, int(np.ceil(self.config.k_percent * n_tokens)))

    def predict_from_log_probs(
        self,
        text: str,
        token_log_probs: Sequence[float],
    ) -> MinKProbResult:
        """Score a single text given its per-token log-probs."""
        logp = list(token_log_probs)
        score = self.min_k_score(logp)
        predicted_member = score >= self.threshold
        # Squash the signed margin into (0, 1) via a logistic transform.
        confidence = float(1.0 / (1.0 + np.exp(-(score - self.threshold))))
        return MinKProbResult(
            text=text,
            score=score,
            predicted_member=predicted_member,
            confidence=confidence,
            n_tokens=len(logp),
            n_min_tokens=self._n_min_tokens(len(logp)),
        )

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate_auc(
        self,
        member_results: Sequence[MinKProbResult],
        nonmember_results: Sequence[MinKProbResult],
        dataset: str = "provided_by_caller",
    ) -> dict[str, float | int | str]:
        """Compute ranking AUC and TPR @ 1% FPR from scored results.

        A higher score indicates a member, so scores are used directly as the
        positive-class ranking signal.
        """
        member_scores = np.array([r.score for r in member_results], dtype=float)
        non_scores = np.array([r.score for r in nonmember_results], dtype=float)

        scores = np.concatenate([member_scores, non_scores])
        labels = np.concatenate(
            [np.ones(len(member_scores)), np.zeros(len(non_scores))]
        )
        auc = float(roc_auc_score(labels, scores))
        tpr_at_1 = self._tpr_at_fpr(member_scores, non_scores, target_fpr=0.01)
        return {
            "auc": auc,
            "tpr_at_1pct_fpr": tpr_at_1,
            "n_members": int(len(member_scores)),
            "n_nonmembers": int(len(non_scores)),
            "method": "min_k_prob",
            "dataset": dataset,
        }

    @staticmethod
    def _tpr_at_fpr(
        member_scores: np.ndarray,
        non_scores: np.ndarray,
        target_fpr: float = 0.01,
    ) -> float:
        """True-positive rate at the largest threshold with FPR <= target."""
        if len(non_scores) == 0 or len(member_scores) == 0:
            return 0.0
        # Threshold chosen from the non-member score quantile: allow at most
        # ``target_fpr`` of non-members above it.
        thr = np.quantile(non_scores, 1.0 - target_fpr)
        return float(np.mean(member_scores >= thr))
