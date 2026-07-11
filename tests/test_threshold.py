"""Tests for the confidence-threshold membership inference attack."""

from __future__ import annotations

import numpy as np

from privacy_attacks.membership_inference.threshold_attack import (
    ThresholdAttack,
    attack_auc,
)


def test_high_confidence_classified_as_member():
    attack = ThresholdAttack(threshold=0.5)
    assert attack.infer([0.99, 0.8, 0.6]) == [True, True, True]


def test_low_confidence_classified_as_nonmember():
    attack = ThresholdAttack(threshold=0.5)
    assert attack.infer([0.4, 0.5, 0.1]) == [False, False, False]


def test_auc_above_0_6_on_synthetic_data():
    """The documented guarantee: on separable synthetic data AUC clears 0.6."""
    rng = np.random.default_rng(42)
    # Members are more confident (higher mean) than non-members.
    member_scores = np.clip(rng.normal(0.85, 0.08, size=500), 0.0, 1.0)
    nonmember_scores = np.clip(rng.normal(0.60, 0.12, size=500), 0.0, 1.0)

    auc = attack_auc(member_scores, nonmember_scores)
    assert auc > 0.6, f"expected AUC > 0.6, got {auc:.4f}"
