"""The blind baseline -- the null model that makes a leakage claim honest.

The single most important lesson from the 2025-26 membership-inference literature
([Blind Baselines Beat MIAs](https://arxiv.org/html/2406.16201v1)) is that a large
fraction of reported "leakage" is not leakage at all: it is distribution shift between
the set an evaluator calls "members" and the set it calls "non-members". An attacker who
never queries the model, and looks only at the *inputs*, can already separate the two.

A blind baseline quantifies exactly that artifact. But a *weak* baseline is dangerous:
if the baseline can only fit linear structure, a purely input-derived but non-linear
membership artifact (e.g. an XOR of two features) will sail past it and be mis-certified
as model leakage. So we do not trust a single model -- we run a **panel** (linear,
non-linear tree ensemble, and nearest-neighbour) with out-of-fold predictions and take
the *strongest* baseline as the floor. The verdict is only ever as strong as the best
null we could build.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from privacy_attacks.certification.stats import roc_auc


@dataclass
class BlindResult:
    """Out-of-fold scores from the strongest baseline, plus the whole panel's AUCs."""

    scores: np.ndarray
    best_name: str
    best_auc: float
    panel_aucs: dict[str, float]


class BlindBaseline:
    """Predict membership from input features only, via an out-of-fold model panel."""

    def __init__(self, n_splits: int = 5, random_state: int = 0) -> None:
        self.n_splits = int(n_splits)
        self.random_state = int(random_state)

    def _panel(self) -> dict:
        rs = self.random_state
        return {
            "logreg": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
            # Captures non-linear / interaction artifacts a linear model cannot (XOR).
            "random_forest": RandomForestClassifier(
                n_estimators=200, random_state=rs, class_weight="balanced_subsample"
            ),
            "knn": make_pipeline(
                StandardScaler(), KNeighborsClassifier(n_neighbors=15)
            ),
        }

    def evaluate(
        self,
        member_features: np.ndarray,
        nonmember_features: np.ndarray,
    ) -> BlindResult:
        """Run the panel out-of-fold; return the strongest baseline's scores + AUCs.

        Scores are ordered ``[members..., non-members...]`` to match the stats module.
        Out-of-fold predictions mean no sample is scored by a model trained on it.
        """
        member_features = np.asarray(member_features, dtype=float)
        nonmember_features = np.asarray(nonmember_features, dtype=float)
        X = np.vstack([member_features, nonmember_features])
        y = np.concatenate(
            [np.ones(len(member_features)), np.zeros(len(nonmember_features))]
        ).astype(int)

        n_min = int(min((y == 0).sum(), (y == 1).sum()))
        n_splits = max(2, min(self.n_splits, n_min))
        cv = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.random_state
        )

        panel_aucs: dict[str, float] = {}
        best_name = ""
        best_auc = -1.0
        best_scores: np.ndarray | None = None
        for name, model in self._panel().items():
            proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
            auc = roc_auc(proba, y)
            panel_aucs[name] = round(auc, 6)
            # Fold the baseline's own asymmetry away: a baseline that scores members
            # *lower* is just as informative, so compare on |AUC - 0.5|.
            if abs(auc - 0.5) > abs(best_auc - 0.5) or best_scores is None:
                best_auc = auc
                best_name = name
                best_scores = proba if auc >= 0.5 else -proba

        assert best_scores is not None
        # Report AUC on the (possibly sign-flipped) chosen scores so it is >= 0.5.
        reported_auc = roc_auc(best_scores, y)
        return BlindResult(
            scores=best_scores,
            best_name=best_name,
            best_auc=reported_auc,
            panel_aucs=panel_aucs,
        )

    def membership_scores(
        self,
        member_features: np.ndarray,
        nonmember_features: np.ndarray,
    ) -> np.ndarray:
        """Backward-compatible convenience: just the strongest baseline's scores."""
        return self.evaluate(member_features, nonmember_features).scores
