"""Tests for the shadow-model membership inference attack."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from privacy_attacks.membership_inference.shadow_model_attack import ShadowModelAttack


def _synthetic_data():
    return make_classification(
        n_samples=400,
        n_features=8,
        n_informative=5,
        n_classes=3,
        random_state=0,
    )


def _model_fn():
    return LogisticRegression(max_iter=500)


def test_shadow_attack_builds_dataset():
    X, y = _synthetic_data()
    attack = ShadowModelAttack(n_shadow=3)
    models = attack.train_shadow_models(X, y, _model_fn)

    attack_X, attack_y = attack.build_attack_dataset(
        models, attack.shadow_train_splits, attack.shadow_test_splits
    )
    assert attack_X.shape[0] == attack_y.shape[0]
    assert attack_X.shape[0] > 0
    # Both members (1) and non-members (0) must be represented.
    assert set(np.unique(attack_y).tolist()) == {0, 1}


def test_attack_classifier_trains():
    X, y = _synthetic_data()
    attack = ShadowModelAttack(n_shadow=3)
    models = attack.train_shadow_models(X, y, _model_fn)
    attack_X, attack_y = attack.build_attack_dataset(
        models, attack.shadow_train_splits, attack.shadow_test_splits
    )
    classifier = attack.train_attack_classifier(attack_X, attack_y)
    assert classifier is attack.attack_classifier
    assert attack.attack_classifier is not None


def test_infer_returns_bool_list():
    X, y = _synthetic_data()
    attack = ShadowModelAttack(n_shadow=3)
    models = attack.train_shadow_models(X, y, _model_fn)
    attack_X, attack_y = attack.build_attack_dataset(
        models, attack.shadow_train_splits, attack.shadow_test_splits
    )
    attack.train_attack_classifier(attack_X, attack_y)

    target_confidences = models[0].predict_proba(X[:10])
    result = attack.infer(target_confidences)
    assert isinstance(result, list)
    assert len(result) == 10
    assert all(isinstance(flag, bool) for flag in result)
