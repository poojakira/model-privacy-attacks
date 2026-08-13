"""Privacy risk assessment for any OpenML dataset.

Provides a one-call API to evaluate membership-inference privacy leakage
on real-world datasets loaded by name from OpenML (via sklearn).

Example
-------
>>> from privacy_attacks.assess import assess_openml_dataset
>>> result = assess_openml_dataset("adult")
>>> print(result.risk_level, result.mia_auc)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from privacy_attacks.mia.direct_mia import DirectMIA


@dataclass
class PrivacyAssessment:
    """Result of a membership-inference privacy risk assessment.

    Attributes
    ----------
    train_acc : float
        Model accuracy on training data (members).
    test_acc : float
        Model accuracy on test data (non-members).
    generalization_gap : float
        train_acc - test_acc. Larger gaps correlate with higher MIA success.
    mia_auc : float
        Area under the ROC curve for the MIA attack. 0.5 = random, 1.0 = perfect attack.
    mia_advantage : float
        max(TPR - FPR) over all thresholds. Key metric from Yeom et al.
    risk_level : str
        Categorical risk: 'low' (AUC < 0.55), 'medium' (0.55-0.6), 'high' (>= 0.6).
    dataset_name : str
        Name of the dataset assessed.
    n_samples : int
        Number of samples in the dataset.
    n_features : int
        Number of features after preprocessing.
    """

    train_acc: float
    test_acc: float
    generalization_gap: float
    mia_auc: float
    mia_advantage: float
    risk_level: str
    dataset_name: str
    n_samples: int
    n_features: int


def _classify_risk(auc: float) -> str:
    """Map MIA AUC to a categorical risk level."""
    if auc < 0.55:
        return "low"
    if auc < 0.6:
        return "medium"
    return "high"


def _prepare_dataset(
    X: np.ndarray | Any,
    y: np.ndarray | Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode categoricals and handle NaN for arbitrary datasets.

    Accepts either numpy arrays or pandas DataFrames/Series.
    Returns clean float64 arrays ready for sklearn models.
    """
    # Convert to numpy if pandas
    try:
        import pandas as pd

        if isinstance(X, pd.DataFrame):
            X_work = X.copy()
            for col in X_work.columns:
                if X_work[col].dtype == "object" or X_work[col].dtype.name == "category":
                    X_work[col] = X_work[col].astype(str).fillna("missing")
                    le = LabelEncoder()
                    X_work[col] = le.fit_transform(X_work[col])
                else:
                    X_work[col] = X_work[col].fillna(X_work[col].median())
            X_out = X_work.values.astype(np.float64)
        else:
            X_out = np.asarray(X, dtype=np.float64)
            # Fill NaN in numeric arrays
            nan_mask = np.isnan(X_out)
            if nan_mask.any():
                col_medians = np.nanmedian(X_out, axis=0)
                inds = np.where(nan_mask)
                X_out[inds] = np.take(col_medians, inds[1])

        if isinstance(y, pd.DataFrame | pd.Series):
            y_raw = y.astype(str).values
        else:
            y_raw = np.asarray(y)
    except ImportError:
        X_out = np.asarray(X, dtype=np.float64)
        nan_mask = np.isnan(X_out)
        if nan_mask.any():
            col_medians = np.nanmedian(X_out, axis=0)
            inds = np.where(nan_mask)
            X_out[inds] = np.take(col_medians, inds[1])
        y_raw = np.asarray(y)

    # Encode target to integer labels
    if y_raw.dtype.kind in ("U", "S", "O"):
        le = LabelEncoder()
        y_out = le.fit_transform(y_raw)
    else:
        y_out = np.asarray(y_raw, dtype=int)

    return X_out, y_out


def assess_openml_dataset(
    name: str = "adult",
    model: Any = None,
    test_size: float = 0.3,
    seed: int = 42,
    *,
    _data_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> PrivacyAssessment:
    """Assess membership-inference privacy risk on a named OpenML dataset.

    Parameters
    ----------
    name : str
        Name of the OpenML dataset to load (e.g., 'adult', 'diabetes', 'credit-g').
    model : sklearn estimator or None
        Model to train and attack. Must support ``fit`` and ``predict_proba``.
        Defaults to GradientBoostingClassifier with moderate capacity.
    test_size : float
        Fraction of data held out as non-members (default 0.3).
    seed : int
        Random state for reproducibility.
    _data_override : tuple[ndarray, ndarray] or None
        If provided, use this (X, y) instead of fetching from OpenML.
        Intended for testing without network access.

    Returns
    -------
    PrivacyAssessment
        Dataclass with accuracy metrics, MIA AUC, advantage, and risk level.
    """
    # --- Load data ---
    if _data_override is not None:
        X_raw, y_raw = _data_override
    else:
        dataset = fetch_openml(name, version=1, as_frame=True, parser="auto")
        X_raw = dataset.data
        y_raw = dataset.target

    # --- Preprocess ---
    X, y = _prepare_dataset(X_raw, y_raw)
    n_samples, n_features = X.shape

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # --- Scale features ---
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Train model ---
    if model is None:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=10,
            random_state=seed,
        )
    model.fit(X_train, y_train)

    # --- Evaluate model accuracy ---
    train_acc = float(model.score(X_train, y_train))
    test_acc = float(model.score(X_test, y_test))
    generalization_gap = train_acc - test_acc

    # --- Run DirectMIA attack ---
    attack = DirectMIA(use_true_label=True)
    attack.fit(
        target_model=model,
        X_members=X_train,
        y_members=y_train,
        X_nonmembers=X_test,
        y_nonmembers=y_test,
    )

    # Compute MIA AUC from raw confidence scores
    member_scores = attack.score_samples(X_train, y_train)
    non_member_scores = attack.score_samples(X_test, y_test)
    all_scores = np.concatenate([member_scores, non_member_scores])
    all_labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])

    mia_auc = float(roc_auc_score(all_labels, all_scores))

    # MIA Advantage = max(TPR - FPR) over all thresholds
    fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_scores)
    mia_advantage = float(np.max(tpr_arr - fpr_arr))

    # --- Classify risk ---
    risk_level = _classify_risk(mia_auc)

    return PrivacyAssessment(
        train_acc=round(train_acc, 4),
        test_acc=round(test_acc, 4),
        generalization_gap=round(generalization_gap, 4),
        mia_auc=round(mia_auc, 4),
        mia_advantage=round(mia_advantage, 4),
        risk_level=risk_level,
        dataset_name=name,
        n_samples=n_samples,
        n_features=n_features,
    )
