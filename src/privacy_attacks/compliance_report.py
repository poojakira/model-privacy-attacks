"""
src/privacy_attacks/compliance_report.py
──────────────────────────────────────────────────────────────────────────────
Compliance report generator for privacy attack results.

Outputs structured JSON with:
  - MIA advantage score and interpretation
  - DP epsilon / delta
  - Risk level assessment
  - Potentially applicable EU AI Act articles (informational reference only —
    not a legal determination; consult qualified legal counsel for compliance advice)
  - Recommended mitigations

IMPORTANT: The EU AI Act article references in this report are provided as
informational guidance for teams evaluating privacy risks in AI systems.
They are NOT a legal determination of compliance or non-compliance.
Whether the EU AI Act applies and which articles are relevant depends on
deployment context, use case, and jurisdiction. Consult qualified legal counsel.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


# ── EU AI Act article references ──────────────────────────────────────────────
# Informational reference only. Maps risk levels to potentially applicable
# EU AI Act (2024/1689) articles for awareness. Not a legal determination.
_EU_AI_ACT_ARTICLES: dict[str, list[dict[str, str]]] = {
    "CRITICAL": [
        {
            "article": "Article 10",
            "title": "Data and Data Governance",
            "requirement": (
                "Training, validation and testing data must be subject to appropriate "
                "data governance and management practices. High-risk AI systems must "
                "use training data free of errors and complete with respect to the "
                "intended purpose — including privacy of individuals."
            ),
        },
        {
            "article": "Article 13",
            "title": "Transparency and Provision of Information to Deployers",
            "requirement": (
                "High-risk AI systems must be designed and developed to ensure that "
                "their operation is sufficiently transparent to deployers. Privacy "
                "risks including membership inference must be disclosed."
            ),
        },
        {
            "article": "Article 72",
            "title": "Post-Market Monitoring",
            "requirement": (
                "Providers of high-risk AI systems must establish and document a "
                "post-market monitoring system proportionate to the nature of the "
                "AI technology and its risks."
            ),
        },
    ],
    "HIGH": [
        {
            "article": "Article 10",
            "title": "Data and Data Governance",
            "requirement": "Privacy-preserving training techniques required for high-risk systems.",
        },
        {
            "article": "Article 9",
            "title": "Risk Management System",
            "requirement": (
                "A risk management system identifying and analysing known and foreseeable "
                "risks must be established, including re-identification and membership "
                "inference risks."
            ),
        },
    ],
    "MEDIUM": [
        {
            "article": "Article 9",
            "title": "Risk Management System",
            "requirement": "Document and monitor privacy risks in the risk management system.",
        },
    ],
    "LOW": [],
}

# ── Risk thresholds ─────────────────────────────────────────────────────────────
def _mia_risk_level(advantage: float) -> str:
    """Classify MIA advantage into a risk level."""
    if advantage >= 0.30:
        return "CRITICAL"
    if advantage >= 0.20:
        return "HIGH"
    if advantage >= 0.10:
        return "MEDIUM"
    return "LOW"


def _epsilon_risk_level(epsilon: float | None) -> str:
    """Classify privacy epsilon into a risk level."""
    if epsilon is None:
        return "UNKNOWN"
    if epsilon > 10.0:
        return "CRITICAL"
    if epsilon > 3.0:
        return "HIGH"
    if epsilon > 1.0:
        return "MEDIUM"
    return "LOW"


def _combined_risk(mia_risk: str, eps_risk: str) -> str:
    """Return the higher of two risk levels."""
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3, "UNKNOWN": -1}
    levels = [mia_risk, eps_risk]
    return max(levels, key=lambda x: order.get(x, -1))


def generate_compliance_report(
    mia_advantage: float,
    epsilon: float | None = None,
    delta: float | None = 1e-5,
    model_name: str = "unnamed-model",
    dataset: str = "synthetic",
    attack_method: str = "YeomMIA",
    random_baseline_advantage: float = 0.10,
    notes: str = "",
) -> dict[str, Any]:
    """Generate a structured privacy compliance report.

    Parameters
    ----------
    mia_advantage:
        TPR - FPR from the membership inference attack.
        Reported value: 0.42. Random baseline: 0.10 (not 0.0 for this dataset).
    epsilon:
        DP epsilon from RDP accounting (e.g. 1.16 at sigma=4.0). None = no DP applied.
    delta:
        DP delta corresponding to epsilon.
    model_name:
        Name of the target model.
    dataset:
        Dataset the model was trained on.
    attack_method:
        Name of the MIA method used.
    random_baseline_advantage:
        Advantage of a random classifier on this dataset (dataset-specific baseline).
        For this evaluation: 0.10 (due to class imbalance in synthetic data).
        A model is "safe" if its MIA advantage is at or below this baseline.
    notes:
        Free-form notes to include in the report.

    Returns
    -------
    dict
        JSON-serialisable compliance report.
    """
    mia_risk = _mia_risk_level(mia_advantage - random_baseline_advantage)
    eps_risk = _epsilon_risk_level(epsilon)
    overall_risk = _combined_risk(mia_risk, eps_risk)
    triggered_articles = _EU_AI_ACT_ARTICLES.get(overall_risk, [])

    # Advantage relative to random baseline
    excess_advantage = round(mia_advantage - random_baseline_advantage, 4)

    report: dict[str, Any] = {
        "report_version": "1.0",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "model": {
            "name": model_name,
            "dataset": dataset,
        },
        "membership_inference": {
            "attack_method": attack_method,
            "reference": "Yeom et al., IEEE CSF 2018" if "yeom" in attack_method.lower() else attack_method,
            "mia_advantage": round(mia_advantage, 4),
            "random_baseline_advantage": random_baseline_advantage,
            "excess_advantage_over_baseline": excess_advantage,
            "interpretation": (
                f"MIA advantage={mia_advantage:.2f}. "
                f"Random baseline for this dataset={random_baseline_advantage:.2f} "
                f"(due to class imbalance in synthetic data; not the theoretical 0.0). "
                f"Excess advantage={excess_advantage:.2f}. "
                + (
                    "Model shows meaningful privacy leakage above random baseline."
                    if excess_advantage > 0.05
                    else "Model is close to random-baseline privacy."
                )
            ),
            "risk_level": mia_risk,
        },
        "differential_privacy": {
            "applied": epsilon is not None,
            "epsilon": round(epsilon, 4) if epsilon is not None else None,
            "delta": delta,
            "sigma": None,  # populated by caller if known
            "accounting_method": "RDP (Mironov 2017)",
            "interpretation": (
                f"epsilon={epsilon:.4f} at delta={delta} — "
                + (
                    "strong privacy guarantee (epsilon < 2)" if epsilon is not None and epsilon < 2
                    else "moderate privacy guarantee" if epsilon is not None and epsilon < 5
                    else "weak privacy guarantee — consider increasing sigma"
                )
            ) if epsilon is not None else "DP not applied.",
            "risk_level": eps_risk,
        },
        "overall_risk_level": overall_risk,
        "eu_ai_act": {
            "regulation": "EU AI Act (2024/1689)",
            "disclaimer": (
                "Informational reference only — not a legal determination. "
                "Consult qualified legal counsel for compliance advice."
            ),
            "reference_articles": triggered_articles,
            "potentially_high_risk_system": overall_risk in ("HIGH", "CRITICAL"),
            "review_recommended": len(triggered_articles) > 0,
        },
        "recommended_mitigations": _mitigations(overall_risk, epsilon),
        "notes": notes,
    }

    return report


def _mitigations(risk_level: str, epsilon: float | None) -> list[str]:
    """Return recommended mitigations for the given risk level."""
    base = [
        "Document all privacy risk assessments in the system's risk management file.",
        "Re-evaluate after any model retraining or dataset update.",
    ]
    if risk_level in ("HIGH", "CRITICAL"):
        base += [
            "Apply DP-SGD with sigma >= 4.0 (target epsilon <= 1.16 at delta=1e-5).",
            "Reduce model capacity to limit overfitting and MIA attack surface.",
            "Consider k-anonymity or synthetic data generation for training data.",
            "Implement prediction confidence rounding or output perturbation.",
        ]
    if risk_level == "CRITICAL":
        base += [
            "Conduct a Data Protection Impact Assessment (DPIA) per GDPR Article 35.",
            "Notify DPO and consider suspending model deployment until remediated.",
        ]
    if epsilon is None:
        base.append(
            "No differential privacy applied — strongly recommended for high-risk systems."
        )
    return base


def save_report(report: dict[str, Any], path: str) -> None:
    """Write report to a JSON file."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
