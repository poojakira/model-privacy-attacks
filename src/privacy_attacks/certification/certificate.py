"""The Privacy Certificate: a null-calibrated, reproducible leakage verdict.

This ties the pieces together. Given per-sample attack scores (from the target model)
and the input features (for the blind-baseline panel), it produces a certificate that
answers three questions a raw AUC cannot:

1. Does the model leak *beyond* the strongest blind baseline we can build? (paired test)
2. How confident are we? (bootstrap CIs + a one-sided bootstrap p-value)
3. Was the substantive content altered? (content checksum, optional HMAC signature)

The verdict is a single, honest one-sided bootstrap test at level ``alpha``: leakage is
"CERTIFIED" only when the model out-separates the blind baseline with p < alpha.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from privacy_attacks.certification.blind_baseline import BlindBaseline
from privacy_attacks.certification.stats import (
    bootstrap_ci,
    bootstrap_tpr_difference,
    paired_delta_auc_test,
    tpr_at_fpr_detailed,
)

SCHEMA_VERSION = "1.1"
TOOL_VERSION = "0.2.0"

# Fields excluded from the integrity digest so a certificate is reproducible bit-for-bit
# across runs/versions/time. These are provenance metadata, not substantive claims.
_UNHASHED_FIELDS = {"created_utc", "integrity", "tool_version", "schema_version"}

# Verdict labels.
CERTIFIED_LEAKAGE = "CERTIFIED_LEAKAGE"
NO_MODEL_LEAKAGE = "NO_MODEL_LEAKAGE"
INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class CertificateConfig:
    attack_name: str = "max_confidence"
    fpr_target: float = 0.001
    alpha: float = 0.05
    n_boot: int = 2000
    confidence: float = 0.95
    seed: int = 0
    target_id: str = "unknown-target"
    # Minimum expected non-members in the tail for the low-FPR metric to be reliable.
    min_tail_count: int = 5


@dataclass
class PrivacyCertificate:
    schema_version: str
    tool_version: str
    created_utc: str
    target_id: str
    attack_name: str
    n_members: int
    n_nonmembers: int
    fpr_target: float
    alpha: float
    decision_rule: str
    # Core statistics.
    model_auc: dict
    blind_auc: dict
    delta_auc: dict
    low_fpr: dict
    verdict: str
    verdict_rationale: str
    integrity: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _canonical_payload(cert_dict: dict) -> str:
    """Deterministic serialization of the substantive fields only."""
    payload = {k: v for k, v in cert_dict.items() if k not in _UNHASHED_FIELDS}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _integrity_block(cert_dict: dict) -> dict:
    """Content checksum (SHA-256), upgraded to an HMAC signature when a key is set.

    Honesty note: a bare SHA-256 over public fields only detects *accidental*
    corruption -- anyone can recompute it. It is called a "checksum", not
    "tamper-evident". Tamper-evidence requires the secret-keyed HMAC, enabled by setting
    ``PRIVACY_CERT_HMAC_KEY``.
    """
    canonical = _canonical_payload(cert_dict).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    block: dict[str, Any] = {
        "algorithm": "sha256",
        "digest": digest,
        "kind": "checksum",
        "signed": False,
    }
    key = os.environ.get("PRIVACY_CERT_HMAC_KEY")
    if key:
        block["hmac_sha256"] = hmac.new(
            key.encode("utf-8"), canonical, hashlib.sha256
        ).hexdigest()
        block["kind"] = "hmac"
        block["signed"] = True
    return block


def certify(
    attack_scores: np.ndarray,
    member_features: np.ndarray,
    nonmember_features: np.ndarray,
    n_members: int,
    config: CertificateConfig | None = None,
) -> PrivacyCertificate:
    """Produce a :class:`PrivacyCertificate`.

    ``attack_scores`` are ordered ``[members..., non-members...]`` (higher = more
    member-like); ``n_members`` is the count of leading member entries.
    """
    config = config or CertificateConfig()
    attack_scores = np.asarray(attack_scores, dtype=float)
    n_total = len(attack_scores)
    n_nonmembers = n_total - n_members
    labels = np.concatenate([np.ones(n_members), np.zeros(n_nonmembers)]).astype(int)

    # Strongest blind baseline over inputs only (panel: linear + forest + kNN).
    blind = BlindBaseline(random_state=config.seed)
    blind_result = blind.evaluate(member_features, nonmember_features)
    blind_scores = blind_result.scores

    # Null-calibrated AUC comparison (one-sided bootstrap, H1: model_auc > blind_auc).
    paired = paired_delta_auc_test(
        attack_scores, blind_scores, labels,
        n_boot=config.n_boot, confidence=config.confidence, seed=config.seed,
    )
    model_auc_ci = bootstrap_ci(
        attack_scores, labels, "auc", n_boot=config.n_boot,
        confidence=config.confidence, seed=config.seed,
    )

    # Low-FPR regime -- reported only when the sample can actually support the target.
    min_neg_needed = int(np.ceil(config.min_tail_count / config.fpr_target))
    low_fpr_reliable = n_nonmembers >= min_neg_needed
    model_tpr, realized_fpr, _ = tpr_at_fpr_detailed(
        attack_scores, labels, config.fpr_target
    )
    blind_tpr, _, _ = tpr_at_fpr_detailed(blind_scores, labels, config.fpr_target)
    if low_fpr_reliable:
        certified_tpr_ci = bootstrap_tpr_difference(
            attack_scores, blind_scores, labels, config.fpr_target,
            n_boot=config.n_boot, confidence=config.confidence, seed=config.seed,
        )
        certified_tpr_block: Any = certified_tpr_ci.as_dict()
    else:
        certified_tpr_block = None
    low_fpr = {
        "requested_fpr": config.fpr_target,
        "realized_fpr": round(realized_fpr, 6),
        "reliable": bool(low_fpr_reliable),
        "min_nonmembers_required": min_neg_needed,
        "model_tpr": round(model_tpr, 6),
        "blind_tpr": round(blind_tpr, 6),
        "certified_tpr_difference": certified_tpr_block,
        "note": (
            "Reliable." if low_fpr_reliable
            else f"Unreliable: need >= {min_neg_needed} non-members for FPR="
            f"{config.fpr_target:g}; the low-FPR metric is omitted from the verdict."
        ),
    }

    # Single, honest decision rule: one-sided bootstrap test at level alpha.
    decision_rule = (
        f"CERTIFIED when the paired one-sided bootstrap p-value for "
        f"(model_auc > best_blind_auc) is < alpha={config.alpha}; NO_MODEL_LEAKAGE "
        f"when the {int(config.confidence * 100)}% delta-AUC CI lies entirely <= 0."
    )
    if paired.p_value < config.alpha and paired.delta > 0:
        verdict = CERTIFIED_LEAKAGE
        rationale = (
            f"Model AUC {paired.model_auc:.3f} exceeds the strongest blind baseline "
            f"('{blind_result.best_name}' AUC {blind_result.best_auc:.3f}) by "
            f"{paired.delta:.3f} (one-sided bootstrap p={paired.p_value:.4f}; "
            f"{int(config.confidence * 100)}% CI [{paired.delta_ci_low:.3f}, "
            f"{paired.delta_ci_high:.3f}]). Leakage is attributable to the model."
        )
    elif paired.delta_ci_high <= 0.0:
        verdict = NO_MODEL_LEAKAGE
        rationale = (
            f"Model AUC {paired.model_auc:.3f} does not exceed the blind baseline "
            f"('{blind_result.best_name}' AUC {blind_result.best_auc:.3f}); the "
            f"{int(config.confidence * 100)}% delta CI upper bound is "
            f"{paired.delta_ci_high:.3f} <= 0. Any separation is a distribution "
            f"artifact, not model leakage."
        )
    else:
        verdict = INCONCLUSIVE
        rationale = (
            f"Delta AUC CI [{paired.delta_ci_low:.3f}, {paired.delta_ci_high:.3f}] "
            f"(p={paired.p_value:.4f}) does not meet the alpha={config.alpha} bar. "
            f"Insufficient evidence to certify model leakage beyond the "
            f"'{blind_result.best_name}' baseline."
        )

    blind_auc_block = {
        "point": round(blind_result.best_auc, 6),
        "best_baseline": blind_result.best_name,
        "panel_aucs": blind_result.panel_aucs,
    }

    cert = PrivacyCertificate(
        schema_version=SCHEMA_VERSION,
        tool_version=TOOL_VERSION,
        created_utc=datetime.now(timezone.utc).isoformat(),
        target_id=config.target_id,
        attack_name=config.attack_name,
        n_members=int(n_members),
        n_nonmembers=int(n_nonmembers),
        fpr_target=config.fpr_target,
        alpha=config.alpha,
        decision_rule=decision_rule,
        model_auc=model_auc_ci.as_dict(),
        blind_auc=blind_auc_block,
        delta_auc=paired.as_dict(),
        low_fpr=low_fpr,
        verdict=verdict,
        verdict_rationale=rationale,
    )
    cert.integrity = _integrity_block(cert.to_dict())
    return cert


def verify_integrity(cert_dict: dict) -> bool:
    """Verify a certificate's integrity, failing closed.

    * The content checksum must match (detects corruption / edits to substantive fields).
    * If ``PRIVACY_CERT_HMAC_KEY`` is set at verification time, a valid HMAC is
      *required* -- the stored ``signed`` flag is never trusted, closing the downgrade
      attack where an editor strips the HMAC and flips the flag.
    """
    stored = cert_dict.get("integrity", {})
    recomputed = _integrity_block(cert_dict)
    if stored.get("digest") != recomputed.get("digest"):
        return False
    if os.environ.get("PRIVACY_CERT_HMAC_KEY"):
        if "hmac_sha256" not in stored or "hmac_sha256" not in recomputed:
            return False  # key present but certificate is unsigned -> fail closed
        return hmac.compare_digest(stored["hmac_sha256"], recomputed["hmac_sha256"])
    return True
