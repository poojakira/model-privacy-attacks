"""Threshold-based membership inference.

Core intuition
--------------
A model tends to be *more confident* on samples it was trained on than on samples it
has never seen. If the maximum softmax probability for a sample exceeds a threshold, we
guess it was a training member.

This is the oldest and simplest membership signal (Shokri et al., 2017 / Yeom et al.,
2018). It is weak on its own but it is the honest baseline every stronger attack must
beat, and it is trivially reproducible offline.
"""

from __future__ import annotations

from collections.abc import Sequence


class ThresholdAttack:
    """Predict membership from per-sample maximum confidence scores."""

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = float(threshold)

    def infer(self, confidence_scores: Sequence[float]) -> list[bool]:
        """Return ``True`` (member) when a sample's confidence exceeds the threshold.

        Parameters
        ----------
        confidence_scores:
            One maximum-confidence value per sample (``max(softmax)``).
        """
        return [float(score) > self.threshold for score in confidence_scores]

    def find_optimal_threshold(
        self,
        member_scores: Sequence[float],
        nonmember_scores: Sequence[float],
    ) -> float:
        """Sweep candidate thresholds and keep the one maximizing balanced accuracy.

        The threshold is chosen on the *provided* score sets. In a faithful audit these
        must come from shadow/reference data, never from the target samples being
        scored -- calibrating on the target is the leakage that inflates published
        attack numbers (see Revisiting LiRA, arXiv:2603.07567).
        """
        member_scores = [float(s) for s in member_scores]
        nonmember_scores = [float(s) for s in nonmember_scores]
        total = len(member_scores) + len(nonmember_scores)
        if total == 0:
            return self.threshold

        # Candidate thresholds: every observed score plus midpoints between neighbours.
        observed = sorted(set(member_scores + nonmember_scores))
        candidates: list[float] = []
        for i, value in enumerate(observed):
            candidates.append(value)
            if i + 1 < len(observed):
                candidates.append((value + observed[i + 1]) / 2.0)
        candidates.append(observed[0] - 1e-9)  # classify everything as member

        best_threshold = self.threshold
        best_accuracy = -1.0
        for t in candidates:
            true_pos = sum(1 for s in member_scores if s > t)
            true_neg = sum(1 for s in nonmember_scores if s <= t)
            accuracy = (true_pos + true_neg) / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t

        self.threshold = best_threshold
        return best_threshold


def attack_auc(
    member_scores: Sequence[float],
    nonmember_scores: Sequence[float],
) -> float:
    """Area under the ROC curve, computed manually (no sklearn) to show the math.

    AUC equals the probability that a randomly chosen member is scored higher than a
    randomly chosen non-member. That is exactly the normalized Mann-Whitney U statistic:
    count every (member, non-member) pair, award 1 when the member scores higher and 0.5
    on a tie, then divide by the number of pairs.
    """
    members = [float(s) for s in member_scores]
    nonmembers = [float(s) for s in nonmember_scores]
    n_members = len(members)
    n_nonmembers = len(nonmembers)
    if n_members == 0 or n_nonmembers == 0:
        return 0.5

    wins = 0.0
    for m in members:
        for nm in nonmembers:
            if m > nm:
                wins += 1.0
            elif m == nm:
                wins += 0.5
    return wins / (n_members * n_nonmembers)
