"""Tests for the entropy-based membership inference attack."""

from __future__ import annotations

import math

from privacy_attacks.membership_inference.entropy_attack import EntropyAttack


def test_low_entropy_is_member():
    attack = EntropyAttack(threshold=0.5)
    # A peaked distribution has low entropy -> predicted member.
    assert attack.infer([[0.95, 0.03, 0.02]]) == [True]


def test_uniform_distribution_is_nonmember():
    attack = EntropyAttack(threshold=0.5)
    # A uniform distribution has maximal entropy -> predicted non-member.
    assert attack.infer([[0.25, 0.25, 0.25, 0.25]]) == [False]


def test_entropy_of_one_hot_is_zero():
    attack = EntropyAttack()
    assert attack.entropy([1.0, 0.0, 0.0]) == 0.0
    # And a uniform vector over k classes has entropy ln(k).
    assert math.isclose(attack.entropy([0.5, 0.5]), math.log(2), rel_tol=1e-9)
