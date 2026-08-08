"""UCI Adult Income MIA Benchmark — Real Privacy Dataset.

This script trains a GradientBoostingClassifier on the UCI Adult Income dataset
(real census records) and evaluates membership inference attacks against it.

The UCI Adult dataset is THE standard benchmark in MIA literature (Shokri et al.
2017). It contains actual census records with sensitive attributes (income,
education, race, sex). Membership inference on this data has real privacy
implications: can an attacker determine if a specific individual's record was
used to train a model?

We demonstrate privacy leakage by:
1. Training a realistic production model (GradientBoosting)
2. Running both Direct MIA and Shadow Model MIA
3. Measuring TPR@FPR thresholds (the metrics privacy engineers care about)
4. Comparing a well-generalized model vs an overfitted model

Reference:
    Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).
    Membership inference attacks against machine learning models. IEEE S&P.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Ensure the src/ package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_attacks.mia.direct_mia import DirectMIA  # noqa: E402
from privacy_attacks.mia.shadow_mia import ShadowMIA  # noqa: E402


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Compute TPR at a given FPR threshold from ROC curve.

    This is the key metric in privacy auditing: at a very low false positive
    rate (e.g., 1% or 0.1%), what fraction of actual members can the attacker
    identify? High TPR@low-FPR = serious privacy risk.
    """
    fpr_arr, tpr_arr, _ = roc_curve(y_true, scores)
    # Find the largest TPR where FPR <= target_fpr
    valid = fpr_arr <= target_fpr
    if not np.any(valid):
        return 0.0
    return float(tpr_arr[valid][-1])


def load_adult_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load UCI Adult Income dataset via sklearn's fetch_openml.

    Returns numerical feature matrix and binary labels (income >50K = 1).
    Handles both categorical and numerical features via label encoding.
    """
    print("Fetching UCI Adult Income dataset (may download on first run)...")
    adult = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = adult.frame

    # Target: >50K = 1, <=50K = 0
    y_col = df.columns[-1]  # target column
    y_raw = df[y_col].astype(str).str.strip().str.rstrip(".")
    y = (y_raw.isin([">50K", ">50K."])).astype(int).values

    # Features: encode categoricals, fill NaN
    X_df = df.drop(columns=[y_col])
    X_encoded = X_df.copy()

    for col in X_encoded.columns:
        if X_encoded[col].dtype == "object" or X_encoded[col].dtype.name == "category":
            X_encoded[col] = X_encoded[col].astype(str).fillna("missing")
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col])
        else:
            X_encoded[col] = X_encoded[col].fillna(X_encoded[col].median())

    X = X_encoded.values.astype(np.float64)
    return X, y


def run_mia_evaluation(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_holdout: np.ndarray,
    y_holdout: np.ndarray,
    model_label: str,
) -> dict:
    """Run full MIA evaluation suite on a trained model.

    Uses X_test as non-members for DirectMIA evaluation,
    and X_holdout as a separate shadow-training pool.
    """
    print(f"\n{'=' * 60}")
    print(f"MIA Evaluation: {model_label}")
    print(f"{'=' * 60}")

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    gap = train_acc - test_acc
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Gap:            {gap:.4f}")

    # ------------------------------------------------------------------
    # Direct MIA (confidence thresholding)
    # ------------------------------------------------------------------
    print("\n  Running Direct MIA (confidence-threshold attack)...")
    t0 = time.perf_counter()

    direct_attack = DirectMIA(use_true_label=True)
    direct_attack.fit(
        target_model=model,
        X_members=X_train,
        y_members=y_train,
        X_nonmembers=X_test,
        y_nonmembers=y_test,
    )
    direct_metrics = direct_attack.evaluate(
        X_members=X_train,
        X_nonmembers=X_test,
        y_members=y_train,
        y_nonmembers=y_test,
    )

    # Compute TPR@FPR thresholds using raw confidence scores
    member_conf = direct_attack.score_samples(X_train, y_train)
    non_conf = direct_attack.score_samples(X_test, y_test)
    all_scores = np.concatenate([member_conf, non_conf])
    all_labels = np.concatenate([np.ones(len(member_conf)), np.zeros(len(non_conf))])

    direct_auc = float(roc_auc_score(all_labels, all_scores))
    tpr_at_1pct = tpr_at_fpr(all_labels, all_scores, 0.01)
    tpr_at_01pct = tpr_at_fpr(all_labels, all_scores, 0.001)

    # MIA Advantage = max(TPR - FPR) over all thresholds
    fpr_arr, tpr_arr, _ = roc_curve(all_labels, all_scores)
    advantage = float(np.max(tpr_arr - fpr_arr))

    direct_elapsed = time.perf_counter() - t0

    print(f"  Direct MIA AUC:       {direct_auc:.4f}")
    print(f"  Direct MIA Advantage: {advantage:.4f}")
    print(f"  TPR@1%FPR:            {tpr_at_1pct:.4f}")
    print(f"  TPR@0.1%FPR:          {tpr_at_01pct:.4f}")
    print(f"  Elapsed:              {direct_elapsed:.2f}s")

    # ------------------------------------------------------------------
    # Shadow Model MIA
    # ------------------------------------------------------------------
    print("\n  Running Shadow Model MIA (4 shadow models)...")
    t1 = time.perf_counter()

    shadow_attack = ShadowMIA(
        n_shadow=4,
        shadow_model_cls="RandomForest",
        attack_model_cls="RandomForest",
        random_state=42,
    )

    # Use holdout + test as the "public" data pool for shadow training
    X_public = np.concatenate([X_holdout, X_test], axis=0)
    y_public = np.concatenate([y_holdout, y_test], axis=0)
    shadow_attack.fit(X_public, y_public, model)

    shadow_metrics = shadow_attack.evaluate(X_train, X_test)

    # Shadow model TPR@FPR
    shadow_member_scores = shadow_attack.predict_proba(X_train)
    shadow_non_scores = shadow_attack.predict_proba(X_test)
    shadow_all_scores = np.concatenate([shadow_member_scores, shadow_non_scores])
    shadow_all_labels = np.concatenate(
        [
            np.ones(len(shadow_member_scores)),
            np.zeros(len(shadow_non_scores)),
        ]
    )
    shadow_tpr_1pct = tpr_at_fpr(shadow_all_labels, shadow_all_scores, 0.01)
    shadow_tpr_01pct = tpr_at_fpr(shadow_all_labels, shadow_all_scores, 0.001)
    shadow_fpr_arr, shadow_tpr_arr, _ = roc_curve(shadow_all_labels, shadow_all_scores)
    shadow_advantage = float(np.max(shadow_tpr_arr - shadow_fpr_arr))

    shadow_elapsed = time.perf_counter() - t1

    print(f"  Shadow MIA AUC:       {shadow_metrics['auc']:.4f}")
    print(f"  Shadow MIA Advantage: {shadow_advantage:.4f}")
    print(f"  Shadow TPR@1%FPR:     {shadow_tpr_1pct:.4f}")
    print(f"  Shadow TPR@0.1%FPR:   {shadow_tpr_01pct:.4f}")
    print(f"  Elapsed:              {shadow_elapsed:.2f}s")

    return {
        "model_label": model_label,
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "generalization_gap": round(gap, 4),
        "direct_mia": {
            "auc": round(direct_auc, 4),
            "advantage": round(advantage, 4),
            "tpr_at_1pct_fpr": round(tpr_at_1pct, 4),
            "tpr_at_0_1pct_fpr": round(tpr_at_01pct, 4),
            "accuracy": round(direct_metrics["accuracy"], 4),
            "threshold": round(direct_metrics["threshold"], 4),
            "elapsed_seconds": round(direct_elapsed, 2),
        },
        "shadow_mia": {
            "auc": round(shadow_metrics["auc"], 4),
            "advantage": round(shadow_advantage, 4),
            "tpr_at_1pct_fpr": round(shadow_tpr_1pct, 4),
            "tpr_at_0_1pct_fpr": round(shadow_tpr_01pct, 4),
            "accuracy": round(shadow_metrics["accuracy"], 4),
            "n_shadow_models": shadow_metrics["n_shadow"],
            "elapsed_seconds": round(shadow_elapsed, 2),
        },
    }


def main() -> None:
    """Run the full UCI Adult Income MIA benchmark."""
    print("=" * 60)
    print("UCI Adult Income — Membership Inference Attack Benchmark")
    print("Real dataset, real privacy implications")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    X, y = load_adult_dataset()
    print(f"\nDataset: UCI Adult Income (n={len(X)}, features={X.shape[1]})")
    print(f"Class balance: {np.mean(y == 1):.1%} positive (>50K income)")

    # ------------------------------------------------------------------
    # 2. Split: 60% train, 20% test (non-members), 20% holdout
    #    This mirrors the standard MIA evaluation protocol.
    # ------------------------------------------------------------------
    X_train_full, X_holdout, y_train_full, y_holdout = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
    )
    # Result: 60% train, 20% test, 20% holdout

    print(f"Train (members):      {len(X_train)} samples")
    print(f"Test (non-members):   {len(X_test)} samples")
    print(f"Holdout (shadow):     {len(X_holdout)} samples")

    # Standardize features for consistent model behavior
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_holdout = scaler.transform(X_holdout)

    # ------------------------------------------------------------------
    # 3. Train models with varying overfitting to show privacy leakage
    # ------------------------------------------------------------------
    configs = [
        {
            "label": "well_generalized",
            "params": {
                "n_estimators": 50,
                "max_depth": 3,
                "learning_rate": 0.1,
                "min_samples_leaf": 50,
                "random_state": 42,
            },
        },
        {
            "label": "moderate_overfit",
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "min_samples_leaf": 5,
                "random_state": 42,
            },
        },
        {
            "label": "heavily_overfitted",
            "params": {
                "n_estimators": 500,
                "max_depth": 12,
                "learning_rate": 0.2,
                "min_samples_leaf": 1,
                "random_state": 42,
            },
        },
    ]

    all_results = []

    for cfg in configs:
        print(f"\n\nTraining: {cfg['label']} (params: {cfg['params']})")
        model = GradientBoostingClassifier(**cfg["params"])
        model.fit(X_train, y_train)

        result = run_mia_evaluation(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_holdout=X_holdout,
            y_holdout=y_holdout,
            model_label=cfg["label"],
        )
        result["hyperparameters"] = cfg["params"]
        all_results.append(result)

    # ------------------------------------------------------------------
    # 4. Write results
    # ------------------------------------------------------------------
    output = {
        "benchmark": "adult_income_mia",
        "real_dataset": True,
        "dataset_source": "UCI Adult Income via sklearn",
        "dataset_description": (
            "48,842 records from 1994 US Census. Contains age, education, "
            "occupation, race, sex, income (>50K or <=50K). Standard MIA "
            "benchmark in privacy literature (Shokri et al. 2017)."
        ),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "split_strategy": "60% train / 20% test / 20% holdout, stratified",
        "model_type": "GradientBoostingClassifier",
        "attack_methods": ["DirectMIA (confidence-threshold)", "ShadowMIA (4 shadow models)"],
        "metrics_explanation": {
            "auc": "Area under ROC. 0.5 = no leakage, >0.6 = exploitable privacy risk",
            "advantage": "max(TPR - FPR) over all thresholds. Key MIA metric.",
            "tpr_at_1pct_fpr": "True positive rate at 1% false positive rate",
            "tpr_at_0_1pct_fpr": "True positive rate at 0.1% false positive rate",
        },
        "privacy_interpretation": {
            "well_generalized": "AUC near 0.5 means model leaks minimal membership info",
            "heavily_overfitted": "AUC > 0.6 means attacker can identify training members",
        },
        "results": all_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reference": "Shokri et al. 2017, IEEE S&P. arXiv:1610.05820",
    }

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "adult_income_mia_benchmark.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # 5. Print summary
    # ------------------------------------------------------------------
    print(f"\n\n{'=' * 60}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 60}")
    print(
        f"\n{'Model':<22} {'Train/Test Gap':<16} {'Direct AUC':<13} {'Shadow AUC':<13} {'Advantage'}"
    )
    print("-" * 77)
    for r in all_results:
        print(
            f"  {r['model_label']:<20} {r['generalization_gap']:<16.4f} "
            f"{r['direct_mia']['auc']:<13.4f} {r['shadow_mia']['auc']:<13.4f} "
            f"{r['direct_mia']['advantage']:.4f}"
        )

    print(f"\n\nResults written to: {output_path}")
    print("\nKey takeaway: More overfitting -> larger generalization gap -> higher MIA AUC")
    print("This demonstrates real privacy leakage on real census data.")


if __name__ == "__main__":
    main()
