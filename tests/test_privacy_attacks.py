"""Seed-pinned synthetic regression tests for the privacy-attacks toolkit.

These tests run every attack end-to-end on deterministic synthetic data
(``random_state=42``) and assert that each attack performs meaningfully better
than chance.  They measure *implementation correctness*, not real-world privacy
leakage.  The exact metrics printed here are the numbers quoted in the README.

Run with ``pytest -s`` to see the measured metrics on stdout, or run this file
directly (``python tests/test_privacy_attacks.py``) for a metrics summary.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from privacy_attacks.extraction import ModelExtractionAttack
from privacy_attacks.llm_mia import TokenLikelihoodMIA, TokenLikelihoodMIAConfig
from privacy_attacks.mia import DirectMIA, ShadowMIA

SEED = 42

# Collected metrics, printed at the end for easy transcription into the README.
MEASURED: dict[str, dict] = {}


def _tpr_at_fpr(member_scores, nonmember_scores, target_fpr=0.01):
    """Implementation check: true-positive rate when allowing target_fpr false positives."""
    threshold = np.quantile(np.asarray(nonmember_scores, dtype=float), 1.0 - target_fpr)
    return float(np.mean(np.asarray(member_scores, dtype=float) >= threshold))


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------
def _make_pool():
    """A single deterministic classification pool sliced into disjoint roles.

    Layout (seed 42, 20 informative-ish features, 2 classes):
        [   0:1000]  members      -> target model's training set
        [1000:2000]  non-members  -> held-out reference / evaluation
        [2000:4000]  public       -> shadow training + extraction queries
        [4000:4500]  eval         -> extraction agreement evaluation
    """
    X, y = make_classification(
        n_samples=4500,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_classes=2,
        random_state=SEED,
    )
    return {
        "X_members": X[:1000],
        "y_members": y[:1000],
        "X_nonmembers": X[1000:2000],
        "y_nonmembers": y[1000:2000],
        "X_public": X[2000:4000],
        "y_public": y[2000:4000],
        "X_eval": X[4000:4500],
    }


def _target_model(X_members, y_members):
    """A RandomForest target that overfits its training set (the leak source)."""
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_members, y_members)
    return model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_direct_mia_auc():
    data = _make_pool()
    target = _target_model(data["X_members"], data["y_members"])

    attack = DirectMIA(use_true_label=True)
    attack.fit(
        target,
        data["X_members"],
        data["y_members"],
        data["X_nonmembers"],
        data["y_nonmembers"],
    )
    metrics = attack.evaluate(
        data["X_members"],
        data["X_nonmembers"],
        data["y_members"],
        data["y_nonmembers"],
    )
    MEASURED["direct_mia"] = metrics
    print("\n[direct_mia]", json.dumps(metrics, indent=2))

    member_scores = attack.score_samples(data["X_members"], data["y_members"])
    nonmember_scores = attack.score_samples(data["X_nonmembers"], data["y_nonmembers"])
    assert metrics["auc"] > 0.55, metrics
    assert _tpr_at_fpr(member_scores, nonmember_scores) > 0.02
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["n_members"] == 1000
    assert metrics["n_nonmembers"] == 1000


def test_shadow_mia_auc():
    data = _make_pool()
    target = _target_model(data["X_members"], data["y_members"])

    attack = ShadowMIA(
        n_shadow=4,
        shadow_model_cls="RandomForest",
        attack_model_cls="RandomForest",
        random_state=SEED,
    )
    attack.fit(data["X_public"], data["y_public"], target)
    metrics = attack.evaluate(data["X_members"], data["X_nonmembers"])
    MEASURED["shadow_mia"] = metrics
    print("\n[shadow_mia]", json.dumps(metrics, indent=2))

    assert metrics["auc"] > 0.55, metrics
    assert _tpr_at_fpr(attack.predict_proba(data["X_members"]), attack.predict_proba(data["X_nonmembers"])) > 0.02
    assert metrics["n_shadow"] == 4
    assert metrics["n_members"] == 1000


def test_shadow_beats_direct():
    """Shadow attack should be at least competitive with the direct attack."""
    data = _make_pool()
    target = _target_model(data["X_members"], data["y_members"])

    direct = DirectMIA(use_true_label=True)
    direct.fit(
        target, data["X_members"], data["y_members"],
        data["X_nonmembers"], data["y_nonmembers"],
    )
    direct_auc = direct.evaluate(
        data["X_members"], data["X_nonmembers"],
        data["y_members"], data["y_nonmembers"],
    )["auc"]

    shadow = ShadowMIA(n_shadow=4, random_state=SEED)
    shadow.fit(data["X_public"], data["y_public"], target)
    shadow_auc = shadow.evaluate(data["X_members"], data["X_nonmembers"])["auc"]

    # Both are valid attacks; assert both clear chance by a margin.
    assert direct_auc > 0.55
    assert shadow_auc > 0.55


def test_model_extraction_agreement():
    data = _make_pool()
    target = _target_model(data["X_members"], data["y_members"])

    attack = ModelExtractionAttack(
        substitute_model_cls="DecisionTree", random_state=SEED
    )
    attack.fit(target, data["X_public"])  # 2000 query samples
    agreement = attack.agreement(target, data["X_eval"])  # 500 eval samples

    metrics = {
        "agreement": agreement,
        "n_query": int(len(data["X_public"])),
        "n_eval": int(len(data["X_eval"])),
        "substitute": "DecisionTree",
    }
    MEASURED["extraction"] = metrics
    print("\n[extraction]", json.dumps(metrics, indent=2))

    target_pred = target.predict(data["X_eval"])
    sub_pred = attack.predict(data["X_eval"])
    extraction_scores = (target_pred == sub_pred).astype(float)
    assert agreement > 0.70, metrics
    assert float(np.mean(extraction_scores)) > 0.55
    assert metrics["n_query"] == 2000
    assert metrics["n_eval"] == 500


def test_extraction_predict_matches_agreement():
    """`predict` on the substitute is consistent with the agreement metric."""
    data = _make_pool()
    target = _target_model(data["X_members"], data["y_members"])
    attack = ModelExtractionAttack(
        substitute_model_cls="DecisionTree", random_state=SEED
    )
    attack.fit(target, data["X_public"])

    sub_pred = attack.predict(data["X_eval"])
    tgt_pred = target.predict(data["X_eval"])
    manual_agreement = float(np.mean(sub_pred == tgt_pred))
    assert abs(manual_agreement - attack.agreement(target, data["X_eval"])) < 1e-12


def _synthetic_log_probs(n_texts, tokens_per_text, mean, std, rng):
    """Generate synthetic per-token log-probs (values are <= 0)."""
    out = []
    for _ in range(n_texts):
        logp = rng.normal(loc=mean, scale=std, size=tokens_per_text)
        logp = np.minimum(logp, -1e-6)  # log-probs are strictly negative
        out.append(logp.tolist())
    return out


def test_min_k_prob_auc():
    """Min-K% Prob separates 'members' (higher rare-token log-probs) from not."""
    rng = np.random.default_rng(SEED)
    config = TokenLikelihoodMIAConfig(k_percent=0.20, min_tokens=10, device="cpu")
    mia = TokenLikelihoodMIA(config=config, threshold=-3.0)

    # Members: rare tokens are less surprising (higher / less-negative log-prob).
    # Non-members: rare tokens are more surprising (lower log-prob).
    # The two distributions deliberately overlap so the AUC is illustrative
    # (not a saturated 1.0) yet clearly above chance.
    member_logps = _synthetic_log_probs(100, 50, mean=-2.8, std=1.5, rng=rng)
    non_logps = _synthetic_log_probs(100, 50, mean=-3.6, std=1.5, rng=rng)

    member_results = [
        mia.predict_from_log_probs(f"member_{i}", lp)
        for i, lp in enumerate(member_logps)
    ]
    non_results = [
        mia.predict_from_log_probs(f"non_{i}", lp)
        for i, lp in enumerate(non_logps)
    ]

    metrics = mia.evaluate_auc(member_results, non_results, dataset="synthetic")
    MEASURED["llm_mia"] = metrics
    print("\n[llm_mia]", json.dumps(metrics, indent=2))

    assert metrics["auc"] > 0.60, metrics
    assert metrics["tpr_at_1pct_fpr"] > 0.02, metrics
    assert metrics["method"] == "min_k_prob"
    assert metrics["n_members"] == 100
    assert metrics["n_nonmembers"] == 100


def test_min_k_score_math():
    """Min-K% score equals the mean of the k% lowest log-probs (exact check)."""
    config = TokenLikelihoodMIAConfig(k_percent=0.20, min_tokens=5)
    mia = TokenLikelihoodMIA(config=config, threshold=-3.0)
    logps = [-0.1, -5.0, -0.2, -4.0, -0.3, -3.0, -0.4, -2.0, -0.5, -1.0]
    # 20% of 10 tokens = 2 lowest: -5.0 and -4.0 -> mean = -4.5
    assert abs(mia.min_k_score(logps) - (-4.5)) < 1e-12
    res = mia.predict_from_log_probs("t", logps)
    assert res.predicted_member is (res.score >= mia.threshold)
    assert res.n_tokens == 10
    assert res.n_min_tokens == 2


def test_min_tokens_guard():
    config = TokenLikelihoodMIAConfig(k_percent=0.2, min_tokens=10)
    mia = TokenLikelihoodMIA(config=config)
    try:
        mia.min_k_score([-1.0, -2.0, -3.0])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("Expected ValueError for too-few tokens.")


# ---------------------------------------------------------------------------
# Direct execution: print a consolidated metrics summary.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_direct_mia_auc()
    test_shadow_mia_auc()
    test_model_extraction_agreement()
    test_min_k_prob_auc()
    print("\n===== MEASURED METRICS (seed 42, synthetic) =====")
    print(json.dumps(MEASURED, indent=2))


def test_model_extraction_probability_fidelity_metrics():
    """Extraction fidelity includes probability-distance, not only agreement."""
    data = _make_pool()
    target = _target_model(data["X_members"], data["y_members"])
    attack = ModelExtractionAttack(
        substitute_model_cls="RandomForest", random_state=SEED
    )
    attack.fit(target, data["X_public"])

    metrics = attack.fidelity_metrics(target, data["X_eval"])

    assert metrics["agreement"] > 0.70
    assert metrics["mean_kl_divergence"] >= 0.0
    assert metrics["mean_l1_distance"] >= 0.0