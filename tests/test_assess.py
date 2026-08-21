"""Tests for privacy_attacks.assess  --  privacy risk assessment.

Uses _data_override to inject local sklearn datasets (no network required).
Tests verify that an overfit model shows higher MIA AUC than a well-generalized one.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from privacy_attacks.assess import PrivacyAssessment, assess_openml_dataset


def _get_breast_cancer_data() -> tuple[np.ndarray, np.ndarray]:
    """Load breast cancer dataset as (X, y) tuple for injection."""
    data = load_breast_cancer()
    return data.data, data.target


class TestAssessBasic:
    """Basic functionality tests for assess_openml_dataset."""

    def test_returns_privacy_assessment(self):
        """Function returns a PrivacyAssessment dataclass."""
        X, y = _get_breast_cancer_data()
        result = assess_openml_dataset(
            name="breast_cancer_test",
            _data_override=(X, y),
            seed=42,
        )
        assert isinstance(result, PrivacyAssessment)

    def test_fields_populated(self):
        """All fields are populated with reasonable values."""
        X, y = _get_breast_cancer_data()
        result = assess_openml_dataset(
            name="breast_cancer_test",
            _data_override=(X, y),
            seed=42,
        )
        assert 0.0 <= result.train_acc <= 1.0
        assert 0.0 <= result.test_acc <= 1.0
        assert result.generalization_gap >= -0.5  # could be slightly negative
        assert 0.0 <= result.mia_auc <= 1.0
        assert 0.0 <= result.mia_advantage <= 1.0
        assert result.risk_level in ("low", "medium", "high")
        assert result.dataset_name == "breast_cancer_test"
        assert result.n_samples == len(X)
        assert result.n_features == X.shape[1]

    def test_risk_level_thresholds(self):
        """Risk levels follow documented thresholds."""
        X, y = _get_breast_cancer_data()
        # Use a well-generalized model (shallow tree) for low risk
        model = GradientBoostingClassifier(
            n_estimators=10,
            max_depth=1,
            learning_rate=0.1,
            min_samples_leaf=50,
            random_state=42,
        )
        result = assess_openml_dataset(
            name="test",
            model=model,
            _data_override=(X, y),
            seed=42,
        )
        # Verify risk_level is consistent with mia_auc thresholds
        if result.mia_auc < 0.55:
            assert result.risk_level == "low"
        elif result.mia_auc < 0.6:
            assert result.risk_level == "medium"
        else:
            assert result.risk_level == "high"

    def test_custom_model(self):
        """Accepts a custom model parameter."""
        X, y = _get_breast_cancer_data()
        model = DecisionTreeClassifier(max_depth=2, random_state=42)
        result = assess_openml_dataset(
            name="test_custom",
            model=model,
            _data_override=(X, y),
            seed=42,
        )
        assert isinstance(result, PrivacyAssessment)
        assert result.train_acc > 0.5  # model should learn something

    def test_custom_test_size(self):
        """Respects the test_size parameter."""
        X, y = _get_breast_cancer_data()
        result = assess_openml_dataset(
            name="test_split",
            test_size=0.5,
            _data_override=(X, y),
            seed=42,
        )
        assert isinstance(result, PrivacyAssessment)


class TestOverfitVsGeneralized:
    """Overfit model should show higher MIA AUC than a well-generalized model.

    This is the core property of membership inference: memorization leaks
    membership information through the model's confidence scores.
    """

    def test_overfit_higher_auc_than_generalized(self):
        """Heavily overfit model leaks more privacy than a generalized one."""
        X, y = _get_breast_cancer_data()

        # Well-generalized: shallow, regularized
        generalized_model = GradientBoostingClassifier(
            n_estimators=10,
            max_depth=1,
            learning_rate=0.05,
            min_samples_leaf=50,
            random_state=42,
        )
        result_gen = assess_openml_dataset(
            name="test_gen",
            model=generalized_model,
            _data_override=(X, y),
            seed=42,
        )

        # Heavily overfit: deep, many trees, no regularization
        overfit_model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.3,
            min_samples_leaf=1,
            random_state=42,
        )
        result_overfit = assess_openml_dataset(
            name="test_overfit",
            model=overfit_model,
            _data_override=(X, y),
            seed=42,
        )

        # Core assertion: overfit model leaks more
        assert result_overfit.mia_auc > result_gen.mia_auc, (
            f"Expected overfit AUC ({result_overfit.mia_auc}) > "
            f"generalized AUC ({result_gen.mia_auc})"
        )

        # The overfit model should also have a larger generalization gap
        assert result_overfit.generalization_gap > result_gen.generalization_gap

    def test_overfit_higher_advantage(self):
        """Overfit model has higher MIA advantage."""
        X, y = _get_breast_cancer_data()

        generalized_model = GradientBoostingClassifier(
            n_estimators=10,
            max_depth=1,
            learning_rate=0.05,
            min_samples_leaf=50,
            random_state=42,
        )
        result_gen = assess_openml_dataset(
            name="test_gen",
            model=generalized_model,
            _data_override=(X, y),
            seed=42,
        )

        overfit_model = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=10,
            learning_rate=0.3,
            min_samples_leaf=1,
            random_state=42,
        )
        result_overfit = assess_openml_dataset(
            name="test_overfit",
            model=overfit_model,
            _data_override=(X, y),
            seed=42,
        )

        assert result_overfit.mia_advantage > result_gen.mia_advantage, (
            f"Expected overfit advantage ({result_overfit.mia_advantage}) > "
            f"generalized advantage ({result_gen.mia_advantage})"
        )
