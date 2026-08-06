"""
src/privacy_attacks/inversion/fredrikson_inversion.py
──────────────────────────────────────────────────────────────────────────────
Fredrikson et al. (2015) model inversion attack.

Reference
---------
Fredrikson, M., Jha, S., & Ristenpart, T. (2015).
Model Inversion Attacks that Exploit Confidence Information and Basic
Countermeasures. ACM CCS 2015.
https://dl.acm.org/doi/10.1145/2810103.2813677

Attack model
------------
Given black-box access to a classification model f(x) -> confidence_vector,
an adversary who knows a target's class label y* can reconstruct a feature
vector x* that the model is confident belongs to class y*:

    x* = argmax_x  f(x)[y*]   subject to  x in feasible_domain

This implementation uses gradient ascent (for differentiable models) or
hill-climbing (for black-box models) to find x*.

Scope and limitations
---------------------
This implementation operates on tabular / low-dimensional feature spaces
(matching the original paper's pharmacogenetics setting). Image-space
inversion (e.g. Yang et al. 2019) requires a generative model and is out
of scope here.

The attack is exercised on synthetic/toy data. No real patient records or
proprietary model weights are used.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol

import numpy as np


class _ConfidenceModel(Protocol):
    """A model that returns a confidence vector over classes."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...  # pragma: no cover


class FredriksonInversion:
    """Confidence-based model inversion attack (Fredrikson et al., 2015).

    Reconstructs a feature vector x* for a target class using hill-climbing
    optimisation over the model's confidence output.

    Parameters
    ----------
    target_class:
        The class label the adversary wants to reconstruct.
    n_restarts:
        Number of random restarts (best result across restarts is returned).
    max_iter:
        Maximum hill-climbing iterations per restart.
    step_size:
        Perturbation magnitude per hill-climbing step.
    n_features:
        Dimensionality of the input space.
    feature_bounds:
        Optional (low, high) tuple clamping feature values.
        Defaults to (0.0, 1.0).
    seed:
        Random seed for reproducibility.

    Example
    -------
    >>> attack = FredriksonInversion(target_class=1, n_features=10)
    >>> x_reconstructed, confidence = attack.run(model)
    >>> print(f"Reconstructed with confidence {confidence:.3f}")
    """

    def __init__(
        self,
        target_class: int,
        n_features: int,
        n_restarts: int = 5,
        max_iter: int = 500,
        step_size: float = 0.01,
        feature_bounds: tuple[float, float] = (0.0, 1.0),
        seed: int = 42,
    ) -> None:
        self.target_class = target_class
        self.n_features = n_features
        self.n_restarts = n_restarts
        self.max_iter = max_iter
        self.step_size = step_size
        self.feature_bounds = feature_bounds
        self._rng = np.random.default_rng(seed)

    def _confidence(self, model: _ConfidenceModel, x: np.ndarray) -> float:
        """Return model confidence for target_class at input x."""
        proba = model.predict_proba(x.reshape(1, -1))[0]
        return float(proba[self.target_class])

    def _hill_climb(
        self,
        model: _ConfidenceModel,
        x_init: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Hill-climbing optimisation from x_init.

        At each step, randomly perturb one feature. Accept the perturbation if
        the target-class confidence improves.
        """
        lo, hi = self.feature_bounds
        x = np.clip(x_init.copy(), lo, hi)
        best_conf = self._confidence(model, x)

        for _ in range(self.max_iter):
            # Perturb a random feature
            feature_idx = int(self._rng.integers(0, self.n_features))
            delta = self._rng.uniform(-self.step_size, self.step_size)
            x_candidate = x.copy()
            x_candidate[feature_idx] = np.clip(
                x_candidate[feature_idx] + delta, lo, hi
            )
            candidate_conf = self._confidence(model, x_candidate)
            if candidate_conf > best_conf:
                x = x_candidate
                best_conf = candidate_conf

        return x, best_conf

    def run(
        self, model: _ConfidenceModel
    ) -> tuple[np.ndarray, float]:
        """Run the inversion attack.

        Parameters
        ----------
        model:
            Target model with ``predict_proba`` interface.

        Returns
        -------
        tuple[np.ndarray, float]
            ``(x_reconstructed, best_confidence)`` where x_reconstructed is
            the feature vector that maximises target-class confidence.
        """
        lo, hi = self.feature_bounds
        best_x: np.ndarray | None = None
        best_conf = -1.0

        for _ in range(self.n_restarts):
            x_init = self._rng.uniform(lo, hi, size=self.n_features)
            x_candidate, conf = self._hill_climb(model, x_init)
            if conf > best_conf:
                best_conf = conf
                best_x = x_candidate

        assert best_x is not None
        return best_x, best_conf

    def inversion_report(
        self, model: _ConfidenceModel
    ) -> dict[str, Any]:
        """Run attack and return a structured report dict."""
        x_reconstructed, best_confidence = self.run(model)
        lo, hi = self.feature_bounds
        return {
            "attack": "FredriksonInversion",
            "reference": "Fredrikson et al., ACM CCS 2015",
            "target_class": self.target_class,
            "n_features": self.n_features,
            "best_confidence": round(best_confidence, 4),
            "n_restarts": self.n_restarts,
            "max_iter": self.max_iter,
            "x_reconstructed": x_reconstructed.tolist(),
            "notes": (
                "Hill-climbing inversion on tabular feature space. "
                "Higher confidence = more successful reconstruction. "
                "Baseline: random x achieves ~1/n_classes confidence."
            ),
        }
