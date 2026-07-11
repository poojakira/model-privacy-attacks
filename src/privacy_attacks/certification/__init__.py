"""Privacy certification: turn a raw attack score into a defensible, gate-able verdict.

The certification layer is the difference between a research demo ("the attack got AUC
0.7") and a product ("the model leaks beyond a blind baseline with p<0.05; here is a
reproducible, tamper-evident certificate, and it fails your CI policy").
"""

from privacy_attacks.certification.adapters import (
    ATTACK_SIGNALS,
    CallableTarget,
    TargetModel,
    attack_scores,
)
from privacy_attacks.certification.blind_baseline import BlindBaseline
from privacy_attacks.certification.certificate import (
    CertificateConfig,
    PrivacyCertificate,
    certify,
    verify_integrity,
)
from privacy_attacks.certification.policy import evaluate_policy, load_policy
from privacy_attacks.certification.stats import (
    bootstrap_ci,
    paired_delta_auc_test,
    roc_auc,
    tpr_at_fpr,
)

__all__ = [
    "certify",
    "CertificateConfig",
    "PrivacyCertificate",
    "verify_integrity",
    "BlindBaseline",
    "CallableTarget",
    "TargetModel",
    "attack_scores",
    "ATTACK_SIGNALS",
    "evaluate_policy",
    "load_policy",
    "roc_auc",
    "tpr_at_fpr",
    "bootstrap_ci",
    "paired_delta_auc_test",
]
