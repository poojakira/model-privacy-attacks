"""Tests for the decision-boundary stealing (model extraction) attack."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from privacy_attacks.model_extraction.boundary_stealing import BoundaryStealingAttack

# A single, moderately-overlapping linear problem shared by the tests. The overlap
# matters: it forces the clone to actually accumulate queries to pin down the boundary,
# which is what makes "more queries -> higher fidelity" a real, observable relationship
# rather than an artifact of an easily separable dataset.
_DATA_KWARGS = dict(
    n_samples=600,
    n_features=6,
    n_informative=4,
    n_redundant=0,
    class_sep=0.9,
    flip_y=0.0,
)


def _make_oracle():
    """A black-box logistic-regression oracle exposing hard labels (one-hot)."""
    X, y = make_classification(random_state=0, **_DATA_KWARGS)
    oracle = LogisticRegression(max_iter=1000).fit(X, y)

    def query_fn(inputs: np.ndarray) -> np.ndarray:
        preds = oracle.predict(np.asarray(inputs, dtype=float))
        one_hot = np.zeros((len(preds), 2), dtype=float)
        one_hot[np.arange(len(preds)), preds.astype(int)] = 1.0
        return one_hot

    return X, query_fn


def _test_points():
    X, _ = make_classification(random_state=1, **_DATA_KWARGS)
    return X


def test_stolen_model_achieves_fidelity_above_0_7():
    X_seed, query_fn = _make_oracle()
    attack = BoundaryStealingAttack(n_queries=1500)
    stolen = attack.steal(query_fn, X_seed)
    fidelity = attack.fidelity(stolen, query_fn, _test_points())
    assert fidelity > 0.7, f"expected fidelity > 0.7, got {fidelity:.4f}"


def test_query_count_affects_fidelity():
    X_seed, query_fn = _make_oracle()
    X_test = _test_points()

    starved = BoundaryStealingAttack(n_queries=10)
    generous = BoundaryStealingAttack(n_queries=2000)

    fid_starved = starved.fidelity(starved.steal(query_fn, X_seed), query_fn, X_test)
    fid_generous = generous.fidelity(generous.steal(query_fn, X_seed), query_fn, X_test)

    # A generous query budget converges on the boundary; a starved one cannot.
    assert fid_generous >= fid_starved


def test_fidelity_metric_range():
    X_seed, query_fn = _make_oracle()
    attack = BoundaryStealingAttack(n_queries=500)
    stolen = attack.steal(query_fn, X_seed)
    fidelity = attack.fidelity(stolen, query_fn, _test_points())
    assert 0.0 <= fidelity <= 1.0
