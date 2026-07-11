"""Membership inference attacks.

Every attack answers one question: given a model's response to a sample, was that
sample part of the model's training set?
"""

from privacy_attacks.membership_inference.entropy_attack import EntropyAttack
from privacy_attacks.membership_inference.shadow_model_attack import ShadowModelAttack
from privacy_attacks.membership_inference.threshold_attack import (
    ThresholdAttack,
    attack_auc,
)

__all__ = ["ThresholdAttack", "attack_auc", "EntropyAttack", "ShadowModelAttack"]
