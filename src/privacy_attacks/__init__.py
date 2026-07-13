"""privacy_attacks: black-box privacy attacks on machine-learning models.

Sub-packages
------------
- ``mia``       : classical membership-inference attacks on sklearn classifiers.
- ``extraction``: model-extraction (substitute-training) attack.
- ``llm_mia``   : Min-K% Prob membership inference for causal LLMs.

All attacks are *black-box*: they assume query access to a target model's
outputs (probabilities / labels / per-token log-probs), not its weights or
gradients.
"""

from __future__ import annotations

from privacy_attacks.extraction import ModelExtractionAttack
from privacy_attacks.llm_mia import (
    MinKProbResult,
    TokenLikelihoodMIA,
    TokenLikelihoodMIAConfig,
)
from privacy_attacks.mia import DirectMIA, ShadowMIA

__all__ = [
    "DirectMIA",
    "ShadowMIA",
    "ModelExtractionAttack",
    "TokenLikelihoodMIA",
    "TokenLikelihoodMIAConfig",
    "MinKProbResult",
]

__version__ = "0.1.0"
