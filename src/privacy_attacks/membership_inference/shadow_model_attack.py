"""Shadow-model membership inference (Shokri et al., 2017).

Core intuition
--------------
We cannot see inside the target model, but we can *imitate* it. Train several "shadow"
models on data from the same distribution, where we know exactly which points were in
each shadow's training set. Each shadow's outputs on its own members vs. non-members
become a labelled dataset. A small attack classifier learns the boundary "this
confidence vector looks like a member" and is then applied to the target model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


class ShadowModelAttack:
    """Learn a membership classifier from imitation ("shadow") models."""

    def __init__(self, n_shadow: int = 4, random_state: int = 0) -> None:
        self.n_shadow = int(n_shadow)
        self.random_state = int(random_state)
        self.shadow_models: list[Any] = []
        self.shadow_train_splits: list[np.ndarray] = []
        self.shadow_test_splits: list[np.ndarray] = []
        self.attack_classifier: LogisticRegression | None = None
        self._n_features: int | None = None

    @staticmethod
    def _confidence_vector(model: Any, samples: np.ndarray) -> np.ndarray:
        """Sorted-descending predict_proba, so the vector is label-order invariant."""
        proba = np.asarray(model.predict_proba(samples), dtype=float)
        return -np.sort(-proba, axis=1)

    def train_shadow_models(
        self,
        X: Sequence[Sequence[float]],
        y: Sequence[int],
        model_fn: Callable[[], Any],
    ) -> list[Any]:
        """Train ``n_shadow`` models, each on a random half of ``(X, y)``.

        The unused half of each split is recorded as that shadow's non-members. Returns
        the list of fitted shadow models and stores the member/non-member splits.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        half = max(1, n // 2)

        self.shadow_models = []
        self.shadow_train_splits = []
        self.shadow_test_splits = []
        for _ in range(self.n_shadow):
            order = rng.permutation(n)
            train_idx, test_idx = order[:half], order[half:]
            model = model_fn()
            model.fit(X[train_idx], y[train_idx])
            self.shadow_models.append(model)
            self.shadow_train_splits.append(X[train_idx])
            self.shadow_test_splits.append(X[test_idx])
        return self.shadow_models

    def build_attack_dataset(
        self,
        shadow_models: Sequence[Any],
        X_train_splits: Sequence[np.ndarray],
        X_test_splits: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build ``(confidence_vector, in/out)`` pairs from the shadow models.

        ``in`` (label 1) means the sample was a member of that shadow's training set;
        ``out`` (label 0) means it was held out.
        """
        features: list[np.ndarray] = []
        labels: list[int] = []
        for model, members, nonmembers in zip(
            shadow_models, X_train_splits, X_test_splits
        ):
            members = np.asarray(members, dtype=float)
            nonmembers = np.asarray(nonmembers, dtype=float)
            if len(members):
                features.append(self._confidence_vector(model, members))
                labels.extend([1] * len(members))
            if len(nonmembers):
                features.append(self._confidence_vector(model, nonmembers))
                labels.extend([0] * len(nonmembers))

        attack_X = np.vstack(features)
        attack_y = np.asarray(labels)
        self._n_features = attack_X.shape[1]
        return attack_X, attack_y

    def train_attack_classifier(
        self,
        attack_X: np.ndarray,
        attack_y: np.ndarray,
    ) -> LogisticRegression:
        """Fit the logistic-regression attack classifier on the shadow dataset."""
        attack_X = np.asarray(attack_X, dtype=float)
        attack_y = np.asarray(attack_y)
        self._n_features = attack_X.shape[1]
        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(attack_X, attack_y)
        self.attack_classifier = classifier
        return classifier

    def infer(self, target_confidences: Sequence[Sequence[float]]) -> list[bool]:
        """Predict membership for target-model confidence vectors.

        ``target_confidences`` may be raw softmax vectors; they are sorted descending to
        match the shadow feature layout.
        """
        if self.attack_classifier is None:
            raise RuntimeError("Call train_attack_classifier before infer().")
        features = -np.sort(-np.asarray(target_confidences, dtype=float), axis=1)
        predictions = self.attack_classifier.predict(features)
        return [bool(p) for p in predictions]
