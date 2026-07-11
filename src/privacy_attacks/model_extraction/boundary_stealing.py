"""Decision-boundary stealing (functionality extraction).

Core intuition
--------------
A prediction API is a free oracle. Query it enough times and label the responses, and
you can fit your own model on those (input, predicted-label) pairs. The clone never sees
the original weights or training data -- only outputs -- yet it can reproduce the
original's decisions. This is the "functionality extraction" branch of the model-
extraction taxonomy (arXiv:2506.22521).

The attack is most sample-efficient when queries are concentrated near the decision
boundary, where each label flip is maximally informative.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

QueryFn = Callable[[np.ndarray], np.ndarray]


class BoundaryStealingAttack:
    """Clone a black-box classifier by querying it near the decision boundary."""

    def __init__(self, n_queries: int = 1000, random_state: int = 0) -> None:
        self.n_queries = int(n_queries)
        self.random_state = int(random_state)
        self.stolen_model: LogisticRegression | None = None

    def generate_queries(self, X_seed: Sequence[Sequence[float]], n: int) -> np.ndarray:
        """Generate ``n`` synthetic query points around the seed distribution.

        We resample seed points and add Gaussian jitter scaled to each feature's spread.
        The jitter pushes samples across the (unknown) boundary so the oracle's labels
        trace its shape, rather than clustering safely inside one class.
        """
        X_seed = np.asarray(X_seed, dtype=float)
        rng = np.random.default_rng(self.random_state)
        mean = X_seed.mean(axis=0)
        std = X_seed.std(axis=0)
        std = np.where(std > 0, std, 1.0)

        base = X_seed[rng.integers(0, len(X_seed), size=n)]
        jitter = rng.normal(0.0, std, size=(n, X_seed.shape[1]))
        # A fraction of queries are drawn broadly around the mean to cover the space.
        broad = rng.normal(mean, 2.0 * std, size=(n, X_seed.shape[1]))
        mask = (rng.random(n) < 0.5)[:, None]
        return np.where(mask, base + jitter, broad)

    def steal(
        self,
        query_fn: QueryFn,
        X_seed: Sequence[Sequence[float]],
    ) -> LogisticRegression:
        """Query the black-box ``query_fn`` and fit a logistic-regression clone.

        ``query_fn`` takes a 2D array of inputs and returns a 2D array of softmax rows.
        """
        queries = self.generate_queries(X_seed, self.n_queries)
        probabilities = np.asarray(query_fn(queries), dtype=float)
        stolen_labels = probabilities.argmax(axis=1)

        clone = LogisticRegression(max_iter=1000)
        # If the oracle only ever returned one class on our queries, fall back to a
        # trivial constant predictor to avoid a fit error.
        if len(np.unique(stolen_labels)) < 2:
            clone = _ConstantClassifier(int(stolen_labels[0]))
        else:
            clone.fit(queries, stolen_labels)
        self.stolen_model = clone
        return clone

    def fidelity(
        self,
        stolen_model: LogisticRegression,
        query_fn: QueryFn,
        X_test: Sequence[Sequence[float]],
    ) -> float:
        """Agreement rate between the stolen clone and the original on ``X_test``.

        Fidelity measures whether the clone *makes the same decisions* as the oracle,
        which is the extraction objective -- distinct from raw task accuracy.
        """
        X_test = np.asarray(X_test, dtype=float)
        original = np.asarray(query_fn(X_test), dtype=float).argmax(axis=1)
        cloned = np.asarray(stolen_model.predict(X_test))
        return float(np.mean(original == cloned))


class _ConstantClassifier:
    """Degenerate fallback used when the oracle returns a single class."""

    def __init__(self, label: int) -> None:
        self.label = int(label)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(np.asarray(X)), self.label, dtype=int)
