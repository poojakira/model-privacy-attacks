"""
Privacy compliance report generator.

Produces a structured JSON report mapping privacy attack findings to
EU AI Act articles and NIST AI RMF functions. Designed to be called
programmatically by security teams or CI pipelines.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Any


def _calculate_risk_level(mia_advantage: float, threshold: float) -> str:
    """
    Calculate risk level based on MIA advantage relative to a random baseline.

    Risk levels:
      LOW      — advantage < 2x threshold (near-random guessing)
      MEDIUM   — advantage between 2x and 3x threshold
      HIGH     — advantage between 3x and 5x threshold
      CRITICAL — advantage >= 5x threshold (strongly memorizing model)
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    ratio = mia_advantage / threshold
    if ratio < 2.0:
        return "LOW"
    elif ratio < 3.0:
        return "MEDIUM"
    elif ratio < 5.0:
        return "HIGH"
    else:
        return "CRITICAL"


def _eu_ai_act_articles(mia_advantage: float, threshold: float, has_inversion: bool) -> list[str]:
    """
    Determine which EU AI Act articles are triggered by the assessment findings.

    Art. 10 — Data Governance: triggered when MIA advantage exceeds threshold
               (evidence of training data memorization).
    Art. 15 — Accuracy and Robustness: triggered when model inversion risk is present
               (evidence that training data can be reconstructed from outputs).
    """
    articles = []
    if mia_advantage > threshold:
        articles.append("Art. 10")  # Data governance / training data exposure
    if has_inversion:
        articles.append("Art. 15")  # Accuracy & robustness / PII reconstruction risk
    return articles


def generate_report(
    model_id: str,
    mia_advantage: float,
    mia_threshold: float = 0.10,
    epsilon: float | None = None,
    sigma: float | None = None,
    dataset_size: int | None = None,
    has_model_inversion: bool = False,
    additional_findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Generate a structured JSON compliance report for a privacy risk assessment.

    This report is the output artifact from running privacy attacks against a model.
    It maps findings to EU AI Act articles and NIST AI RMF functions and is
    designed to be attached to a security design review or compliance audit.

    Args:
        model_id: Identifier for the model being assessed (e.g., "resnet18-prod-v2").
        mia_advantage: Membership inference attack advantage score (0.0–1.0).
                       Represents how much better than random the attacker performs.
                       A random baseline (no leakage) ≈ mia_threshold.
        mia_threshold: Random baseline threshold for MIA. Default 0.10 (10%).
                       Set this to the expected random-guessing rate for your dataset.
        epsilon: Achieved differential privacy epsilon (if DP-SGD was applied).
                 None if no DP defense was applied.
        sigma: DP-SGD noise multiplier used during training (if applicable).
        dataset_size: Number of training samples (used for context in the report).
        has_model_inversion: Whether model inversion attack succeeded (triggers Art. 15).
        additional_findings: Additional finding dicts to include in the report.
                             Each dict should have keys: attack, severity, remediation_hint.

    Returns:
        dict: Structured compliance report. Save as JSON for audit trail.

    Example:
        >>> report = generate_report(
        ...     model_id="resnet18-cifar10",
        ...     mia_advantage=0.42,
        ...     mia_threshold=0.10,
        ...     epsilon=1.16,
        ...     sigma=4.0,
        ...     dataset_size=50000,
        ... )
        >>> print(report["risk_level"])
        HIGH
        >>> print(report["eu_ai_act_articles_triggered"])
        ['Art. 10']
    """
    if not 0.0 <= mia_advantage <= 1.0:
        raise ValueError(f"mia_advantage must be between 0.0 and 1.0, got {mia_advantage}")
    if not 0.0 < mia_threshold <= 1.0:
        raise ValueError(f"mia_threshold must be between 0.0 and 1.0, got {mia_threshold}")

    risk_level = _calculate_risk_level(mia_advantage, mia_threshold)
    eu_articles = _eu_ai_act_articles(mia_advantage, mia_threshold, has_model_inversion)

    # Core MIA finding
    mia_finding: dict[str, Any] = {
        "attack": "membership_inference",
        "mia_advantage": round(mia_advantage, 4),
        "random_baseline": round(mia_threshold, 4),
        "severity": risk_level,
        "eu_ai_act_articles_triggered": eu_articles,
        "nist_ai_rmf_function": "MANAGE 2.2",
        "mitre_technique": "T1005",
        "remediation_hint": _mia_remediation_hint(mia_advantage, mia_threshold, epsilon),
    }

    findings: list[dict[str, Any]] = [mia_finding]

    # DP defense finding (if applicable)
    if epsilon is not None:
        dp_risk = "LOW" if epsilon <= 1.0 else ("MEDIUM" if epsilon <= 3.0 else "HIGH")
        findings.append({
            "attack": "dp_sgd_defense_assessment",
            "achieved_epsilon": round(epsilon, 4),
            "sigma": round(sigma, 4) if sigma is not None else None,
            "risk_level": dp_risk,
            "eu_ai_act_articles_triggered": ["Art. 10"],
            "nist_ai_rmf_function": "GOVERN 1.1",
            "remediation_hint": (
                f"Epsilon = {epsilon:.2f}. Target epsilon <= 1.0 for strong privacy guarantees. "
                f"Increase sigma (current: {sigma}) or reduce training epochs."
                if epsilon > 1.0
                else f"Epsilon = {epsilon:.2f} meets the recommended threshold of <= 1.0."
            ),
        })

    # Model inversion finding (if applicable)
    if has_model_inversion:
        findings.append({
            "attack": "model_inversion",
            "severity": "HIGH",
            "eu_ai_act_articles_triggered": ["Art. 15"],
            "nist_ai_rmf_function": "MANAGE 4.1",
            "mitre_technique": "T1005",
            "remediation_hint": (
                "Model inversion succeeded. Apply prediction confidence thresholding, "
                "output perturbation, or restrict API access to top-k predictions only."
            ),
        })

    # Additional caller-supplied findings
    if additional_findings:
        findings.extend(additional_findings)

    # Severity summary
    severity_summary: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in findings:
        sev = f.get("severity") or f.get("risk_level") or "LOW"
        if sev in severity_summary:
            severity_summary[sev] += 1

    # Recommended defense
    recommended_defense = _recommend_defense(mia_advantage, mia_threshold, epsilon)

    # Remediation hints (deduplicated)
    remediation_hints = list({
        f["remediation_hint"]
        for f in findings
        if "remediation_hint" in f and f["remediation_hint"]
    })

    report: dict[str, Any] = {
        "tool": "model-privacy-attacks",
        "version": "1.0.0",
        "assessment_date": date.today().isoformat(),
        "assessment_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "model_id": model_id,
        "dataset_size": dataset_size,
        "mia_advantage": round(mia_advantage, 4),
        "threshold": round(mia_threshold, 4),
        "risk_level": risk_level,
        "eu_ai_act_articles_triggered": eu_articles,
        "achieved_epsilon": round(epsilon, 4) if epsilon is not None else None,
        "sigma": round(sigma, 4) if sigma is not None else None,
        "recommended_defense": recommended_defense,
        "findings": findings,
        "severity_summary": severity_summary,
        "remediation_hints": remediation_hints,
    }

    return report


def _mia_remediation_hint(
    mia_advantage: float,
    threshold: float,
    epsilon: float | None,
) -> str:
    """Generate a context-aware remediation hint for MIA findings."""
    if mia_advantage <= threshold:
        return "MIA advantage within acceptable range. No immediate action required."

    hints = []
    if epsilon is None or epsilon > 3.0:
        hints.append("Apply DP-SGD with epsilon <= 1.0 and sigma >= 1.5")
    elif epsilon > 1.0:
        hints.append(f"Current epsilon {epsilon:.2f} exceeds target 1.0 — increase sigma")
    hints.append("Reduce training epochs to limit overfitting and memorization")
    hints.append(
        "Add output perturbation: label smoothing (smoothing=0.1) or "
        "prediction confidence thresholding (return top-k only)"
    )
    return "; ".join(hints)


def _recommend_defense(
    mia_advantage: float,
    threshold: float,
    epsilon: float | None,
) -> str:
    """Recommend the primary defense based on the assessment results."""
    if mia_advantage <= threshold:
        return "No defense required — MIA advantage within acceptable range"
    if epsilon is None:
        return "DP-SGD with epsilon <= 1.0 (not currently applied)"
    if epsilon > 3.0:
        return f"Increase DP-SGD sigma to achieve epsilon <= 1.0 (current epsilon: {epsilon:.2f})"
    if epsilon > 1.0:
        return f"Tune DP-SGD: increase sigma to reach epsilon <= 1.0 (current: {epsilon:.2f})"
    return f"DP-SGD applied with epsilon = {epsilon:.2f} — meets target"


def save_report(report: dict[str, Any], output_path: str) -> None:
    """Save a compliance report to a JSON file.

    Args:
        report: Report dict from generate_report().
        output_path: File path to write (e.g., "privacy_audit.json").
    """
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
