"""Entropy-based membership inference.

Core intuition
--------------
Confidence collapses to a single number, but the *shape* of the full softmax vector
carries more signal. A model that has memorized a training sample produces a peaked
distribution -- low Shannon entropy. On unseen data the distribution is flatter --
higher entropy. So: low entropy => likely member.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class EntropyAttack:
    """Predict membership from the Shannon entropy of softmax outputs."""

    def __init__(self, threshold: float = 0.5) -> None:
        # Entropy is in nats (natural log). A one-hot vector has entropy 0; a uniform
        # distribution over k classes has entropy ln(k).
        self.threshold = float(threshold)

    def entropy(self, probs: Sequence[float]) -> float:
        """Shannon entropy ``-sum(p * ln p)`` in nats, treating ``0 * ln 0`` as 0."""
        total = 0.0
        for p in probs:
            p = float(p)
            if p > 0.0:
                total -= p * math.log(p)
        return total

    def infer(self, softmax_outputs: Sequence[Sequence[float]]) -> list[bool]:
        """Return ``True`` (member) when a sample's entropy is below the threshold."""
        return [self.entropy(out) < self.threshold for out in softmax_outputs]

    def calibrate_threshold(
        self,
        member_outputs: Sequence[Sequence[float]],
        nonmember_outputs: Sequence[Sequence[float]],
    ) -> float:
        """Pick the entropy threshold maximizing balanced accuracy on reference data.

        A member is predicted when ``entropy < threshold``. As with the confidence
        attack, calibration data must be held out from the samples later scored.
        """
        member_entropy = [self.entropy(o) for o in member_outputs]
        nonmember_entropy = [self.entropy(o) for o in nonmember_outputs]
        total = len(member_entropy) + len(nonmember_entropy)
        if total == 0:
            return self.threshold

        observed = sorted(set(member_entropy + nonmember_entropy))
        candidates: list[float] = []
        for i, value in enumerate(observed):
            candidates.append(value)
            if i + 1 < len(observed):
                candidates.append((value + observed[i + 1]) / 2.0)
        candidates.append(observed[-1] + 1e-9)  # classify everything as member

        best_threshold = self.threshold
        best_accuracy = -1.0
        for t in candidates:
            true_pos = sum(1 for e in member_entropy if e < t)
            true_neg = sum(1 for e in nonmember_entropy if e >= t)
            accuracy = (true_pos + true_neg) / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t

        self.threshold = best_threshold
        return best_threshold
