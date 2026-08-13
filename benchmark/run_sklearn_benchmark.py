"""Real sklearn benchmark: Membership Inference Attack on Iris dataset.

This script trains a RandomForestClassifier on the Iris dataset (built-in,
no downloads required) and evaluates the DirectMIA attack against it.

This is NOT synthetic data — it uses a real model on a real dataset to produce
actual MIA advantage metrics.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ensure the src/ package is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from privacy_attacks.mia.direct_mia import DirectMIA  # noqa: E402


def main() -> None:
    """Train model, run MIA, write results."""
    print("=" * 60)
    print("Iris MIA Benchmark — Real Model, Real Data")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load Iris dataset (built-in, no network required)
    # ------------------------------------------------------------------
    X, y = load_iris(return_X_y=True)
    print(f"\nDataset: Iris (n={len(X)}, features={X.shape[1]}, classes={len(np.unique(y))})")

    # ------------------------------------------------------------------
    # 2. Split: 70% train (members), 30% test (non-members)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    print(f"Train (members): {len(X_train)} samples")
    print(f"Test (non-members): {len(X_test)} samples")

    # ------------------------------------------------------------------
    # 3. Train a real RandomForestClassifier
    # ------------------------------------------------------------------
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=None,  # fully grown trees tend to overfit → higher MIA signal
    )
    model.fit(X_train, y_train)

    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    print(f"\nModel accuracy — Train: {train_acc:.4f}, Test: {test_acc:.4f}")
    print(f"Generalization gap: {train_acc - test_acc:.4f}")

    # ------------------------------------------------------------------
    # 4. Run DirectMIA attack
    # ------------------------------------------------------------------
    print("\nRunning DirectMIA (confidence-threshold attack)...")
    t0 = time.perf_counter()

    attack = DirectMIA(use_true_label=True)
    attack.fit(
        target_model=model,
        X_members=X_train,
        y_members=y_train,
        X_nonmembers=X_test,
        y_nonmembers=y_test,
    )

    # Evaluate on the same member/non-member split
    eval_results = attack.evaluate(
        X_members=X_train,
        X_nonmembers=X_test,
        y_members=y_train,
        y_nonmembers=y_test,
    )

    elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # 5. Compute MIA advantage (TPR - FPR)
    # ------------------------------------------------------------------
    # TPR = fraction of members correctly predicted as members
    member_preds = attack.predict(X_train, y_train)
    non_member_preds = attack.predict(X_test, y_test)

    tpr = float(np.mean(member_preds))  # predicted member among actual members
    fpr = float(np.mean(non_member_preds))  # predicted member among actual non-members
    mia_advantage = tpr - fpr

    # ------------------------------------------------------------------
    # 6. Assemble results
    # ------------------------------------------------------------------
    results = {
        "benchmark": "iris_mia",
        "dataset": "sklearn.datasets.load_iris (built-in, 150 samples, 4 features, 3 classes)",
        "model": "RandomForestClassifier(n_estimators=100, random_state=42)",
        "split": "70/30 stratified, random_state=42",
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "model_train_accuracy": round(train_acc, 4),
        "model_test_accuracy": round(test_acc, 4),
        "generalization_gap": round(train_acc - test_acc, 4),
        "attack_method": "DirectMIA (confidence-threshold, Shokri et al. 2017)",
        "attack_auc": round(eval_results["auc"], 4),
        "attack_accuracy": round(eval_results["accuracy"], 4),
        "attack_threshold": round(eval_results["threshold"], 4),
        "tpr": round(tpr, 4),
        "fpr": round(fpr, 4),
        "mia_advantage_tpr_minus_fpr": round(mia_advantage, 4),
        "elapsed_seconds": round(elapsed, 3),
        "synthetic_data": False,
        "real_benchmark": True,
        "note": (
            "This is a real benchmark on the Iris dataset. "
            "MIA advantage > 0 indicates the attack detects membership signal. "
            "Random forests with full depth tend to memorize training data, "
            "producing measurable MIA advantage."
        ),
    }

    # ------------------------------------------------------------------
    # 7. Write results
    # ------------------------------------------------------------------
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    output_path = results_dir / "iris_mia_benchmark.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print(f"\n{'-' * 60}")
    print("RESULTS")
    print(f"{'-' * 60}")
    print(f"  Attack AUC:           {results['attack_auc']}")
    print(f"  Attack Accuracy:      {results['attack_accuracy']}")
    print(f"  TPR (true members):   {results['tpr']}")
    print(f"  FPR (false positives):{results['fpr']}")
    print(f"  MIA Advantage:        {results['mia_advantage_tpr_minus_fpr']}")
    print(f"  Elapsed:              {results['elapsed_seconds']}s")
    print(f"\nResults written to: {output_path}")
    print(f"{'-' * 60}")

    # Exit with error if attack is worse than random (sanity check)
    if results["attack_auc"] < 0.5:
        print("\nWARNING: AUC < 0.5, attack performs worse than random.")
        sys.exit(1)


if __name__ == "__main__":
    main()
