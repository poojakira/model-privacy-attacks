"""Tests for the privacy-certification engine.

These focus on the properties that make the certificate *trustworthy*: the AUC matches a
reference implementation, the null-calibrated test does not cry wolf, the verdict is
conservative, the integrity digest is tamper-evident, and the policy gate fails closed.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from privacy_attacks.certification import (
    CertificateConfig,
    certify,
    evaluate_policy,
    verify_integrity,
)
from privacy_attacks.certification.certificate import (
    CERTIFIED_LEAKAGE,
    NO_MODEL_LEAKAGE,
)
from privacy_attacks.certification.cli import main as cli_main
from privacy_attacks.certification.stats import (
    paired_delta_auc_test,
    roc_auc,
    tpr_at_fpr,
)

FAST = dict(n_boot=400, seed=0)


def test_roc_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    s = rng.normal(y * 0.6, 1.0)
    assert abs(roc_auc(s, y) - roc_auc_score(y, s)) < 1e-9


def test_tpr_at_fpr_bounds_and_null():
    rng = np.random.default_rng(1)
    labels = np.concatenate([np.ones(1000), np.zeros(1000)]).astype(int)
    separable = np.concatenate([rng.normal(4, 1, 1000), rng.normal(0, 1, 1000)])
    assert tpr_at_fpr(separable, labels, 0.01) > 0.5
    # Pure noise: TPR at 1% FPR should be small.
    noise = rng.normal(0, 1, 2000)
    assert tpr_at_fpr(noise, labels, 0.01) < 0.1


def test_paired_test_reports_no_leakage_when_model_equals_blind():
    rng = np.random.default_rng(2)
    labels = np.concatenate([np.ones(600), np.zeros(600)]).astype(int)
    blind = np.concatenate([rng.normal(0.3, 1, 600), rng.normal(0, 1, 600)])
    result = paired_delta_auc_test(blind.copy(), blind.copy(), labels, n_boot=400)
    assert result.p_value > 0.5  # cannot beat itself
    assert result.delta == 0.0


def test_paired_test_detects_real_leakage():
    rng = np.random.default_rng(3)
    n = 800
    labels = np.concatenate([np.ones(n), np.zeros(n)]).astype(int)
    blind = np.concatenate([rng.normal(0.3, 1, n), rng.normal(0, 1, n)])
    model = blind + np.concatenate([rng.normal(0.8, 1, n), rng.normal(0, 1, n)])
    result = paired_delta_auc_test(model, blind, labels, n_boot=400)
    assert result.delta > 0
    assert result.p_value < 0.05


def _same_distribution_leakage():
    """Same input distribution, but model is far more confident on members."""
    rng = np.random.default_rng(10)
    d = 8
    member_X = rng.normal(0, 1, (400, d))
    nonmember_X = rng.normal(0, 1, (400, d))
    member_scores = np.clip(rng.normal(0.9, 0.1, 400), 0, 1)
    nonmember_scores = np.clip(rng.normal(0.6, 0.15, 400), 0, 1)
    scores = np.concatenate([member_scores, nonmember_scores])
    return scores, member_X, nonmember_X


def test_certify_certifies_genuine_leakage():
    scores, mX, nX = _same_distribution_leakage()
    cert = certify(scores, mX, nX, n_members=400, config=CertificateConfig(**FAST))
    assert cert.verdict == CERTIFIED_LEAKAGE
    assert cert.delta_auc["delta_ci_low"] > 0
    assert cert.blind_auc["point"] < 0.65  # same distribution -> blind near chance


def test_certify_rejects_nonlinear_artifact():
    """C-1 regression: an XOR-style input artifact must be caught by the panel.

    The 'attack score' is a pure function of the inputs (x0*x1) -- zero model leakage.
    A linear baseline cannot represent XOR, but the panel's random forest can, so the
    certificate must NOT certify leakage.
    """
    rng = np.random.default_rng(12)
    n = 500
    member_X = rng.normal(0, 1, (n, 4))
    nonmember_X = rng.normal(0, 1, (n, 4))
    # Members constructed so that x0*x1 tends positive; non-members tend negative.
    member_X[:, 0] = np.abs(member_X[:, 0])
    member_X[:, 1] = np.abs(member_X[:, 1])
    nonmember_X[:, 0] = np.abs(nonmember_X[:, 0])
    nonmember_X[:, 1] = -np.abs(nonmember_X[:, 1])
    member_scores = member_X[:, 0] * member_X[:, 1] + rng.normal(0, 0.01, n)
    nonmember_scores = nonmember_X[:, 0] * nonmember_X[:, 1] + rng.normal(0, 0.01, n)
    scores = np.concatenate([member_scores, nonmember_scores])
    cert = certify(scores, member_X, nonmember_X, n_members=n,
                   config=CertificateConfig(**FAST))
    # A weak linear-only baseline would falsely CERTIFY here; the panel must not.
    assert cert.verdict != CERTIFIED_LEAKAGE
    assert cert.blind_auc["best_baseline"] == "random_forest"


def test_certify_rejects_distribution_artifact():
    """Members/non-members differ in input; the 'attack' just echoes that shift."""
    rng = np.random.default_rng(11)
    d = 8
    member_X = rng.normal(0.8, 1, (400, d))
    nonmember_X = rng.normal(0, 1, (400, d))
    member_scores = 0.7 + 0.05 * member_X[:, 0] + rng.normal(0, 0.05, 400)
    nonmember_scores = 0.7 + 0.05 * nonmember_X[:, 0] + rng.normal(0, 0.05, 400)
    scores = np.concatenate([member_scores, nonmember_scores])
    cert = certify(scores, member_X, nonmember_X, n_members=400,
                   config=CertificateConfig(**FAST))
    assert cert.verdict == NO_MODEL_LEAKAGE
    assert cert.blind_auc["point"] > 0.7  # inputs alone separate the sets


def test_hmac_verification_fails_closed_on_downgrade(monkeypatch):
    """C-2 regression: with a key present, an unsigned/stripped cert must not verify."""
    scores, mX, nX = _same_distribution_leakage()
    monkeypatch.setenv("PRIVACY_CERT_HMAC_KEY", "super-secret")
    cert = certify(scores, mX, nX, n_members=400, config=CertificateConfig(**FAST))
    d = cert.to_dict()
    assert d["integrity"]["signed"] is True
    assert verify_integrity(d) is True
    # Downgrade: strip the HMAC and flip the flag, recompute the public checksum.
    from privacy_attacks.certification.certificate import _canonical_payload
    import hashlib
    d["verdict"] = NO_MODEL_LEAKAGE
    d["integrity"] = {
        "algorithm": "sha256",
        "kind": "checksum",
        "signed": False,
        "digest": hashlib.sha256(_canonical_payload(d).encode()).hexdigest(),
    }
    # Key still present at verification time -> must fail closed despite valid checksum.
    assert verify_integrity(d) is False


def test_integrity_is_tamper_evident():
    scores, mX, nX = _same_distribution_leakage()
    cert = certify(scores, mX, nX, n_members=400, config=CertificateConfig(**FAST))
    d = cert.to_dict()
    assert verify_integrity(d) is True
    # Flip a headline number; the digest must no longer verify.
    d["delta_auc"]["delta_auc"] = 0.999
    assert verify_integrity(d) is False


def test_certificate_is_reproducible():
    scores, mX, nX = _same_distribution_leakage()
    c1 = certify(scores, mX, nX, n_members=400, config=CertificateConfig(**FAST))
    c2 = certify(scores, mX, nX, n_members=400, config=CertificateConfig(**FAST))
    assert c1.integrity["digest"] == c2.integrity["digest"]


def test_policy_gate_fails_closed_on_certified_leakage():
    scores, mX, nX = _same_distribution_leakage()
    cert = certify(scores, mX, nX, n_members=400, config=CertificateConfig(**FAST))
    policy = {"require_verdict_not_in": [CERTIFIED_LEAKAGE]}
    result = evaluate_policy(cert.to_dict(), policy)
    assert result.passed is False


def test_cli_returns_policy_violation_exit_code(tmp_path):
    rng = np.random.default_rng(5)
    npz = tmp_path / "run.npz"
    np.savez(
        npz,
        member_scores=np.clip(rng.normal(0.9, 0.1, 300), 0, 1),
        nonmember_scores=np.clip(rng.normal(0.6, 0.15, 300), 0, 1),
        member_features=rng.normal(0, 1, (300, 6)),
        nonmember_features=rng.normal(0, 1, (300, 6)),
    )
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"require_verdict_not_in": [CERTIFIED_LEAKAGE]}))
    cert_out = tmp_path / "cert.json"

    exit_code = cli_main(
        [
            "audit",
            str(npz),
            "--policy",
            str(policy),
            "--out",
            str(cert_out),
            "--n-boot",
            "300",
        ]
    )
    assert exit_code == 2  # policy violation
    saved = json.loads(cert_out.read_text())
    assert saved["verdict"] == CERTIFIED_LEAKAGE
