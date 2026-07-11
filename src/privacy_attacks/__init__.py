"""Offline, deterministic membership-inference and model-extraction attack simulator.

The public API intentionally exposes small, dependency-light attack objects so that
attack metrics (AUC, TPR@low-FPR, extraction fidelity) can be reproduced locally with
no external model or API access.
"""

from privacy_attacks.membership_inference.entropy_attack import EntropyAttack
from privacy_attacks.membership_inference.shadow_model_attack import ShadowModelAttack
from privacy_attacks.membership_inference.threshold_attack import (
    ThresholdAttack,
    attack_auc,
)
from privacy_attacks.metrics import (
    attack_advantage,
    compute_auc,
    membership_inference_report,
)
from privacy_attacks.model_extraction.boundary_stealing import BoundaryStealingAttack

__all__ = [
    "ThresholdAttack",
    "attack_auc",
    "EntropyAttack",
    "ShadowModelAttack",
    "BoundaryStealingAttack",
    "compute_auc",
    "attack_advantage",
    "membership_inference_report",
]

__version__ = "0.1.0"
