"""Target-model adapters -- the seam that lets certification touch *real* models.

The certification engine never imports a deep-learning framework. It only needs a way
to turn inputs into per-sample membership *signals*. Anything that can produce
class probabilities (a scikit-learn estimator, a PyTorch/HF wrapper, or an HTTP
prediction endpoint) can be adapted here.

Two attack signals are provided out of the box, both black-box (probabilities only):

* ``max_confidence`` -- the maximum softmax probability (Yeom et al., 2018).
* ``negative_entropy`` -- minus the Shannon entropy of the output vector; higher means
  more peaked, i.e. more member-like.

Framework-specific adapters (HuggingFace, ONNX, remote API) subclass ``TargetModel``;
the interface is stable even though only the callable adapter ships today.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class TargetModel(Protocol):
    """Anything that maps a batch of inputs to a batch of class-probability rows."""

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class CallableTarget:
    """Wrap a plain callable ``f(X) -> probs`` as a :class:`TargetModel`.

    This is the black-box seam: ``f`` can be a local model, or a function that calls a
    remote prediction API and returns the returned probability rows.
    """

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        self._fn = fn

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self._fn(np.asarray(X, dtype=float)), dtype=float)


def max_confidence(model: TargetModel, X: np.ndarray) -> np.ndarray:
    """Per-sample maximum class probability."""
    proba = np.asarray(model.predict_proba(np.asarray(X, dtype=float)), dtype=float)
    return proba.max(axis=1)


def negative_entropy(model: TargetModel, X: np.ndarray) -> np.ndarray:
    """Per-sample negative Shannon entropy (higher = more confident = more member-like)."""
    proba = np.asarray(model.predict_proba(np.asarray(X, dtype=float)), dtype=float)
    eps = 1e-12
    entropy = -np.sum(proba * np.log(proba + eps), axis=1)
    return -entropy


ATTACK_SIGNALS: dict[str, Callable[[TargetModel, np.ndarray], np.ndarray]] = {
    "max_confidence": max_confidence,
    "negative_entropy": negative_entropy,
}


def attack_scores(
    model: TargetModel,
    member_X: np.ndarray,
    nonmember_X: np.ndarray,
    signal: str = "max_confidence",
) -> np.ndarray:
    """Return the chosen attack signal for ``[members..., non-members...]``."""
    if signal not in ATTACK_SIGNALS:
        raise ValueError(
            f"unknown signal {signal!r}; choose from {sorted(ATTACK_SIGNALS)}"
        )
    fn = ATTACK_SIGNALS[signal]
    member_scores = fn(model, np.asarray(member_X, dtype=float))
    nonmember_scores = fn(model, np.asarray(nonmember_X, dtype=float))
    return np.concatenate([member_scores, nonmember_scores])
