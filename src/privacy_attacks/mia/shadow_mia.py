"""Shadow-model membership-inference attack.

The attacker trains several *shadow* models on data drawn from the same
distribution as the target's training set.  For each shadow model it knows the
ground-truth membership (which samples were in the shadow's training set), so it
can build a labelled dataset of (confidence-vector -> member?) pairs and train an
*attack classifier* on it.  The attack classifier is then applied to confidence
vectors produced by the real target model.

Reference
---------
Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).
Membership inference attacks against machine learning models. IEEE S&P.
https://arxiv.org/abs/1610.05820
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier


class _ProbaModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


_MODEL_FACTORY = {
    "RandomForest": lambda rs: RandomForestClassifier(
        n_estimators=100, random_state=rs
    ),
    "DecisionTree": lambda rs: DecisionTreeClassifier(random_state=rs),
    "LogisticRegression": lambda rs: LogisticRegression(max_iter=1000),
}


def _make_model(name: str, random_state: Optional[int]):
    if name not in _MODEL_FACTORY:
        raise ValueError(
            f"Unknown model '{name}'. Choose from {sorted(_MODEL_FACTORY)}."
        )
    return _MODEL_FACTORY[name](random_state)


class ShadowMIA:
    """Shadow-model membership-inference attack.

    Parameters
    ----------
    n_shadow:
        Number of shadow models to train.
    shadow_model_cls:
        Name of the sklearn classifier used for shadow models.  Should match
        the target's architecture family for best results.
    attack_model_cls:
        Name of the sklearn classifier used as the attack (meta) classifier.
    random_state:
        Seed controlling shadow-data splits and model initialisation.
    """

    def __init__(
        self,
        n_shadow: int = 4,
        shadow_model_cls: str = "RandomForest",
        attack_model_cls: str = "RandomForest",
        random_state: Optional[int] = None,
    ) -> None:
        self.n_shadow = n_shadow
        self.shadow_model_cls = shadow_model_cls
        self.attack_model_cls = attack_model_cls
        self.random_state = random_state

        self.attack_model_ = None
        self.target_model_: Optional[_ProbaModel] = None
        self.n_features_: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Feature construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _features(model: _ProbaModel, X: np.ndarray) -> np.ndarray:
        """Attack features: per-sample class probabilities sorted descending.

        Sorting makes the feature representation invariant to class identity,
        which is what lets a single attack classifier generalise across classes.
        """
        proba = np.asarray(model.predict_proba(X))
        return -np.sort(-proba, axis=1)

    # ------------------------------------------------------------------ #
    # Fit
    # ------------------------------------------------------------------ #
    def fit(
        self,
        X_public: np.ndarray,
        y_public: np.ndarray,
        target_model: _ProbaModel,
    ) -> "ShadowMIA":
        """Train shadow models + attack classifier and register the target."""
        X_public = np.asarray(X_public)
        y_public = np.asarray(y_public)
        rng = np.random.default_rng(self.random_state)

        attack_X: list[np.ndarray] = []
        attack_y: list[np.ndarray] = []

        n = X_public.shape[0]
        for i in range(self.n_shadow):
            # Random in/out split of the public data for this shadow model.
            perm = rng.permutation(n)
            half = n // 2
            in_idx, out_idx = perm[:half], perm[half:]

            seed = None if self.random_state is None else self.random_state + i
            shadow = _make_model(self.shadow_model_cls, seed)
            shadow.fit(X_public[in_idx], y_public[in_idx])

            feat_in = self._features(shadow, X_public[in_idx])
            feat_out = self._features(shadow, X_public[out_idx])

            attack_X.append(feat_in)
            attack_y.append(np.ones(len(in_idx)))
            attack_X.append(feat_out)
            attack_y.append(np.zeros(len(out_idx)))

        X_attack = np.concatenate(attack_X, axis=0)
        y_attack = np.concatenate(attack_y, axis=0)
        self.n_features_ = X_attack.shape[1]

        self.attack_model_ = _make_model(self.attack_model_cls, self.random_state)
        self.attack_model_.fit(X_attack, y_attack)
        self.target_model_ = target_model
        return self

    # ------------------------------------------------------------------ #
    # Predict / evaluate
    # ------------------------------------------------------------------ #
    def _check_fitted(self) -> None:
        if self.attack_model_ is None or self.target_model_ is None:
            raise RuntimeError("ShadowMIA must be fitted before use.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Probability that each row of ``X`` was a member of the target's data."""
        self._check_fitted()
        feats = self._features(self.target_model_, X)
        proba = self.attack_model_.predict_proba(feats)
        # Index of the positive ("member") class.
        classes = list(self.attack_model_.classes_)
        pos = classes.index(1.0) if 1.0 in classes else classes.index(1)
        return proba[:, pos]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict membership (``True`` = predicted member) for each row of ``X``."""
        return self.predict_proba(X) >= 0.5

    def evaluate(
        self, X_members: np.ndarray, X_nonmembers: np.ndarray
    ) -> dict[str, float]:
        """Compute attack AUC and accuracy on a labelled member/non-member set."""
        self._check_fitted()
        member_scores = self.predict_proba(X_members)
        non_scores = self.predict_proba(X_nonmembers)

        scores = np.concatenate([member_scores, non_scores])
        labels = np.concatenate(
            [np.ones(len(member_scores)), np.zeros(len(non_scores))]
        )
        auc = float(roc_auc_score(labels, scores))
        preds = (scores >= 0.5).astype(int)
        accuracy = float(np.mean(preds == labels.astype(int)))
        return {
            "auc": auc,
            "accuracy": accuracy,
            "n_shadow": int(self.n_shadow),
            "n_members": int(len(member_scores)),
            "n_nonmembers": int(len(non_scores)),
        }
