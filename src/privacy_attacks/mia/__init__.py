"""Classical membership-inference attacks on sklearn classifiers."""

from __future__ import annotations

from privacy_attacks.mia.direct_mia import DirectMIA
from privacy_attacks.mia.shadow_mia import ShadowMIA

__all__ = ["DirectMIA", "ShadowMIA"]
