"""
src/privacy_attacks/mia/yeom_mia.py
──────────────────────────────────────────────────────────────────────────────
Yeom 2018 membership inference attack.

Reference
---------
Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018).
Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.
IEEE Computer Security Foundations Symposium (CSF).
https://arxiv.org/abs/1709.01604

Attack model
------------
The Yeom attack exploits a simple observation: a model trained on a sample x
will typically assign a higher loss to non-members than to members (because it
has "memorised" the members to some degree). The attack uses a single threshold
on the per-sample loss to classify membership:

    member_pred(x) = 1  if  loss(model, x) <= threshold
                     0  otherwise

The default threshold is the mean training loss, as proposed in the original
paper. This is the simplest possible MIA  --  it achieves non-trivial advantage
on overfit models without requiring shadow models or confidence scores.

Metrics
-------
MIA Advantage = TPR - FPR  (also called "balanced advantage" or "epsilon-advantage")
Random baseline advantage = 0.0 (a random classifier achieves 0 advantage).
A well-regularised model should have advantage close to 0.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from sklearn.metrics import roc_auc_score


class _LossModel(Protocol):
    """Minimal protocol: model exposes per-sample loss."""

    def per_sample_loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray: ...  # pragma: no cover


class YeomMIA:
    """Yeom et al. (2018) loss-threshold membership inference attack.

    Parameters
    ----------
    threshold:
        Loss threshold. Samples with loss <= threshold are predicted as
        members. If None, set to mean training loss during :meth:`fit`.

    Example
    -------
    >>> attack = YeomMIA()
    >>> attack.fit(model, X_train, y_train)
    >>> advantage = attack.advantage(X_train, y_train, X_test, y_test)
    >>> print(f"MIA advantage: {advantage:.3f}  (random baseline: 0.0)")
    """

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold
        self._fitted_threshold: float | None = threshold

    def fit(
        self,
        model: _LossModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> YeomMIA:
        """Calibrate the loss threshold on training data.

        Parameters
        ----------
        model:
            Trained model exposing ``per_sample_loss(X, y) -> np.ndarray``.
        X_train:
            Training set features used to train the model.
        y_train:
            Training set labels.

        Returns
        -------
        YeomMIA
            self (for chaining).
        """
        train_losses = model.per_sample_loss(X_train, y_train)
        if self.threshold is None:
            # Default: mean training loss as per Yeom et al. §4.1
            self._fitted_threshold = float(np.mean(train_losses))
        else:
            self._fitted_threshold = self.threshold
        return self

    def predict(
        self,
        model: _LossModel,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Predict membership for each sample.

        Returns
        -------
        np.ndarray of int
            1 = predicted member, 0 = predicted non-member.
        """
        if self._fitted_threshold is None:
            raise RuntimeError("Call fit() before predict().")
        losses = model.per_sample_loss(X, y)
        return (losses <= self._fitted_threshold).astype(int)

    def advantage(
        self,
        model: _LossModel,
        X_members: np.ndarray,
        y_members: np.ndarray,
        X_nonmembers: np.ndarray,
        y_nonmembers: np.ndarray,
    ) -> float:
        """Compute MIA advantage = TPR - FPR.

        Parameters
        ----------
        model:
            The target model.
        X_members / y_members:
            Samples that WERE in the training set (true members).
        X_nonmembers / y_nonmembers:
            Held-out samples NOT in the training set (true non-members).

        Returns
        -------
        float
            MIA advantage in [-1, 1]. Random baseline = 0.0. Higher = worse privacy.
        """
        if self._fitted_threshold is None:
            raise RuntimeError("Call fit() before advantage().")

        member_preds = self.predict(model, X_members, y_members)
        nonmember_preds = self.predict(model, X_nonmembers, y_nonmembers)

        tpr = float(np.mean(member_preds))  # fraction of members correctly identified
        fpr = float(np.mean(nonmember_preds))  # fraction of non-members incorrectly flagged
        return tpr - fpr

    def auc_score(
        self,
        model: _LossModel,
        X_members: np.ndarray,
        y_members: np.ndarray,
        X_nonmembers: np.ndarray,
        y_nonmembers: np.ndarray,
    ) -> float:
        """Compute ROC-AUC of the attack.

        Uses negative loss as the membership score (higher = more likely member).
        """
        member_losses = model.per_sample_loss(X_members, y_members)
        nonmember_losses = model.per_sample_loss(X_nonmembers, y_nonmembers)

        scores = np.concatenate([-member_losses, -nonmember_losses])
        labels = np.concatenate(
            [
                np.ones(len(member_losses)),
                np.zeros(len(nonmember_losses)),
            ]
        )
        return float(roc_auc_score(labels, scores))

    @property
    def fitted_threshold(self) -> float | None:
        """The loss threshold used for membership prediction."""
        return self._fitted_threshold
