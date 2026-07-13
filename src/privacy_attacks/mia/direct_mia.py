"""Direct (confidence-threshold) membership-inference attack.

The direct attack exploits the tendency of a classifier to assign higher
confidence to samples it was trained on than to held-out samples.  A single
threshold on the model's confidence score separates predicted *members* from
predicted *non-members*.

Reference
---------
Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).
Membership inference attacks against machine learning models. IEEE S&P.
https://arxiv.org/abs/1610.05820
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import roc_auc_score

# A target model only needs to expose ``predict_proba``; we type it loosely.
from typing import Any, Protocol


class _ProbaModel(Protocol):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover
        ...


class DirectMIA:
    """Confidence-threshold membership-inference attack.

    Parameters
    ----------
    threshold:
        Confidence threshold above which a sample is predicted to be a member.
        If ``None``, it is selected automatically during :meth:`fit`
        (median member confidence, or the balanced-accuracy-optimal threshold
        when non-member reference data is supplied).
    use_true_label:
        If ``True`` (default) the confidence score is the probability the model
        assigns to the *true* class of each sample.  If ``False`` (or when
        labels are unavailable at predict time) the maximum class probability
        is used instead.
    """

    def __init__(
        self,
        threshold: Optional[float] = None,
        use_true_label: bool = True,
    ) -> None:
        self.threshold = threshold
        self.use_true_label = use_true_label
        self.model_: Optional[_ProbaModel] = None
        self.threshold_: Optional[float] = None

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def _confidence(
        self,
        model: _ProbaModel,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a per-sample confidence score in ``[0, 1]``."""
        proba = np.asarray(model.predict_proba(X))
        if self.use_true_label and y is not None:
            y = np.asarray(y).astype(int)
            return proba[np.arange(proba.shape[0]), y]
        return proba.max(axis=1)

    def score_samples(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Confidence score used as the membership signal for each sample."""
        if self.model_ is None:
            raise RuntimeError("DirectMIA must be fitted before scoring.")
        return self._confidence(self.model_, X, y)

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(
        self,
        target_model: _ProbaModel,
        X_members: np.ndarray,
        y_members: Optional[np.ndarray] = None,
        X_nonmembers: Optional[np.ndarray] = None,
        y_nonmembers: Optional[np.ndarray] = None,
    ) -> "DirectMIA":
        """Register the target model and select a threshold.

        Only ``target_model`` and ``X_members`` are required.  Supplying
        reference non-member data lets the attack pick a threshold that
        maximises balanced accuracy between the two groups.
        """
        self.model_ = target_model

        if self.threshold is not None:
            self.threshold_ = float(self.threshold)
            return self

        member_conf = self._confidence(target_model, X_members, y_members)

        if X_nonmembers is not None:
            non_conf = self._confidence(target_model, X_nonmembers, y_nonmembers)
            self.threshold_ = self._best_threshold(member_conf, non_conf)
        else:
            # No reference set: fall back to the median member confidence.
            self.threshold_ = float(np.median(member_conf))
        return self

    @staticmethod
    def _best_threshold(
        member_conf: np.ndarray, non_conf: np.ndarray
    ) -> float:
        """Threshold maximising balanced accuracy over candidate cut points."""
        candidates = np.unique(np.concatenate([member_conf, non_conf]))
        best_t, best_score = float(candidates[0]), -1.0
        for t in candidates:
            tpr = float(np.mean(member_conf >= t))
            tnr = float(np.mean(non_conf < t))
            bal = 0.5 * (tpr + tnr)
            if bal > best_score:
                best_score, best_t = bal, float(t)
        return best_t

    def predict(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Predict membership (``True`` = predicted member) for each row of ``X``."""
        if self.model_ is None or self.threshold_ is None:
            raise RuntimeError("DirectMIA must be fitted before calling predict.")
        return self._confidence(self.model_, X, y) >= self.threshold_

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    def evaluate(
        self,
        X_members: np.ndarray,
        X_nonmembers: np.ndarray,
        y_members: Optional[np.ndarray] = None,
        y_nonmembers: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Compute attack AUC and accuracy on a labelled member/non-member set.

        The AUC is threshold-independent (computed from the raw confidence
        scores); the accuracy uses the fitted threshold.
        """
        if self.model_ is None:
            raise RuntimeError("DirectMIA must be fitted before evaluation.")
        member_conf = self._confidence(self.model_, X_members, y_members)
        non_conf = self._confidence(self.model_, X_nonmembers, y_nonmembers)

        scores = np.concatenate([member_conf, non_conf])
        labels = np.concatenate(
            [np.ones(len(member_conf)), np.zeros(len(non_conf))]
        )
        auc = float(roc_auc_score(labels, scores))

        preds = np.concatenate(
            [member_conf >= self.threshold_, non_conf >= self.threshold_]
        )
        accuracy = float(np.mean(preds == labels.astype(bool)))
        return {
            "auc": auc,
            "accuracy": accuracy,
            "threshold": float(self.threshold_),
            "n_members": int(len(member_conf)),
            "n_nonmembers": int(len(non_conf)),
        }
