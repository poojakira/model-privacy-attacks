"""Validate the privacy-certification engine on a REAL model + REAL data.

This is *not* a toy. We use the canonical membership-inference target -- MNIST (70k
real handwritten digit images, 784 pixel features) -- and train two genuine neural
networks on the *same* small member set:

* an OVERFIT MLP (large capacity, negligible weight decay) that memorizes, and
* a REGULARIZED MLP (small capacity, strong weight decay) that generalizes.

The engine then decides, against a *strong* blind-baseline panel (linear + random
forest + kNN over the raw pixels), whether each model leaks membership beyond what the
inputs alone reveal. Because members and non-members are both MNIST, the blind baseline
sits near 0.5, so any certified separation is genuinely attributable to the model.

The punchline: the two models have almost identical *naive* attack AUC, so a credulous
tool cannot tell them apart. Only the null-calibrated certificate does -- it CERTIFIES
the overfit model's leakage (p < 0.01, CI strictly above zero) and refuses to certify
the regularized one.

Requires network access on first run (fetches MNIST via OpenML; cached thereafter).

    python examples/real_model_demo.py
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from privacy_attacks.certification import CertificateConfig, certify
from privacy_attacks.certification.adapters import attack_scores
from privacy_attacks.certification.stats import roc_auc

SEED = 0
N_MEMBERS = 2000  # small enough that a high-capacity net memorizes it


def load_split():
    """Load MNIST and carve a member (train) set + a disjoint non-member set."""
    X, y = fetch_openml(
        "mnist_784", version=1, as_frame=False, return_X_y=True, parser="liac-arff"
    )
    X = X.astype(float) / 255.0
    y = y.astype(int)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]
    member_X, member_y = X[:N_MEMBERS], y[:N_MEMBERS]
    nonmember_X, nonmember_y = X[N_MEMBERS : 2 * N_MEMBERS], y[N_MEMBERS : 2 * N_MEMBERS]
    return member_X, member_y, nonmember_X, nonmember_y


def summarize(name, model, member_X, member_y, nonmember_X, nonmember_y):
    train_acc = accuracy_score(member_y, model.predict(member_X))
    test_acc = accuracy_score(nonmember_y, model.predict(nonmember_X))
    scores = attack_scores(model, member_X, nonmember_X, signal="max_confidence")
    labels = np.concatenate(
        [np.ones(len(member_X)), np.zeros(len(nonmember_X))]
    ).astype(int)
    naive_auc = roc_auc(scores, labels)

    config = CertificateConfig(
        attack_name="max_confidence", fpr_target=0.001, alpha=0.05,
        n_boot=2000, confidence=0.95, seed=SEED, target_id=name,
    )
    cert = certify(scores, member_X, nonmember_X, n_members=len(member_X), config=config)
    d = cert.delta_auc

    print("=" * 78)
    print(f"MODEL: {name}")
    print("-" * 78)
    print(f"  train_acc / test_acc  : {train_acc:.4f} / {test_acc:.4f}")
    print(f"  NAIVE raw attack AUC  : {naive_auc:.4f}   <- what a credulous tool reports")
    print(f"  blind baseline (null) : {cert.blind_auc['point']:.4f} "
          f"(best='{cert.blind_auc['best_baseline']}', panel={cert.blind_auc['panel_aucs']})")
    print(f"  delta AUC             : {d['delta_auc']:.4f} "
          f"(95% CI [{d['delta_ci_low']:.4f}, {d['delta_ci_high']:.4f}], p={d['p_value']:.4f})")
    print(f"  VERDICT               : {cert.verdict}")
    print(f"  rationale             : {cert.verdict_rationale}")
    return {
        "model": name, "train_acc": train_acc, "test_acc": test_acc,
        "naive_auc": naive_auc, "blind_auc": cert.blind_auc["point"],
        "delta": d["delta_auc"], "ci_low": d["delta_ci_low"],
        "ci_high": d["delta_ci_high"], "p_value": d["p_value"], "verdict": cert.verdict,
    }


def main() -> None:
    mX, my, nX, ny = load_split()
    print(f"Dataset: MNIST (real) | members={len(mX)} non-members={len(nX)} "
          f"features={mX.shape[1]}\n")

    overfit = MLPClassifier(hidden_layer_sizes=(256, 256), alpha=1e-6,
                            max_iter=150, random_state=SEED)
    overfit.fit(mX, my)
    regularized = MLPClassifier(hidden_layer_sizes=(64,), alpha=3.0,
                                max_iter=150, random_state=SEED)
    regularized.fit(mX, my)

    rows = [
        summarize("overfit_mlp_256x256_alpha1e-6", overfit, mX, my, nX, ny),
        summarize("regularized_mlp_64_alpha3.0", regularized, mX, my, nX, ny),
    ]

    print("\n" + "=" * 78)
    print("SUMMARY (real numbers from this run)")
    print("=" * 78)
    print("model | train | test | naive_AUC | blind_AUC | delta_AUC (95% CI) | p | verdict")
    for r in rows:
        print(f"{r['model']} | {r['train_acc']:.3f} | {r['test_acc']:.3f} | "
              f"{r['naive_auc']:.3f} | {r['blind_auc']:.3f} | "
              f"{r['delta']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | "
              f"{r['p_value']:.4f} | {r['verdict']}")


if __name__ == "__main__":
    main()
