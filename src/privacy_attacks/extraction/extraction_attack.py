"""Model-extraction attack via substitute-model training.

The attacker queries a target model on a pool of inputs, records the returned
labels (or probabilities), and trains a local *substitute* model to imitate the
target.  Fidelity is measured by *agreement*: the fraction of held-out inputs on
which the substitute and target predict the same label.

Reference
---------
Tramèr, F., Zhang, F., Juels, A., Reiter, M. K., & Ristenpart, T. (2016).
Stealing machine learning models via prediction APIs. USENIX Security.
https://arxiv.org/abs/1609.02943
"""

from __future__ import annotations

from typing import Any, Optional, Protocol

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


class _PredictModel(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


_MODEL_FACTORY = {
    "DecisionTree": lambda rs: DecisionTreeClassifier(random_state=rs),
    "RandomForest": lambda rs: RandomForestClassifier(
        n_estimators=100, random_state=rs
    ),
    "LogisticRegression": lambda rs: LogisticRegression(max_iter=1000),
    "MLP": lambda rs: MLPClassifier(max_iter=500, random_state=rs),
}


class ModelExtractionAttack:
    """Substitute-model extraction attack.

    Parameters
    ----------
    substitute_model_cls:
        Name of the sklearn classifier trained to imitate the target.
    random_state:
        Seed for the substitute model.
    """

    def __init__(
        self,
        substitute_model_cls: str = "DecisionTree",
        random_state: Optional[int] = None,
    ) -> None:
        if substitute_model_cls not in _MODEL_FACTORY:
            raise ValueError(
                f"Unknown model '{substitute_model_cls}'. "
                f"Choose from {sorted(_MODEL_FACTORY)}."
            )
        self.substitute_model_cls = substitute_model_cls
        self.random_state = random_state
        self.substitute_model_: Any = None

    def fit(
        self, target_model: _PredictModel, X_query: np.ndarray
    ) -> "ModelExtractionAttack":
        """Query ``target_model`` on ``X_query`` and train the substitute."""
        X_query = np.asarray(X_query)
        y_target = np.asarray(target_model.predict(X_query))
        self.substitute_model_ = _MODEL_FACTORY[self.substitute_model_cls](
            self.random_state
        )
        self.substitute_model_.fit(X_query, y_target)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels using the extracted substitute model."""
        if self.substitute_model_ is None:
            raise RuntimeError("Attack must be fitted before calling predict.")
        return self.substitute_model_.predict(X)

    def agreement(self, target_model: _PredictModel, X_eval: np.ndarray) -> float:
        """Fraction of ``X_eval`` where substitute and target labels match."""
        if self.substitute_model_ is None:
            raise RuntimeError("Attack must be fitted before measuring agreement.")
        X_eval = np.asarray(X_eval)
        target_pred = np.asarray(target_model.predict(X_eval))
        sub_pred = self.substitute_model_.predict(X_eval)
        return float(np.mean(target_pred == sub_pred))
