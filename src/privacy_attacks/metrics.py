"""Privacy-attack evaluation metrics.

The field has moved away from headline accuracy/AUC toward the low-false-positive
regime, because an attack is only meaningful if it can confidently identify *some*
members without drowning in false alarms (Carlini et al., 2022). We expose both: a
manual AUC via the trapezoid rule and the TPR@low-FPR "advantage".
"""

from __future__ import annotations

from collections.abc import Sequence


def compute_auc(tpr_list: Sequence[float], fpr_list: Sequence[float]) -> float:
    """Area under the ROC curve via the trapezoid rule.

    Points are sorted by increasing FPR, then the area is accumulated as the sum of
    trapezoids ``(x1 - x0) * (y0 + y1) / 2``.
    """
    if len(tpr_list) != len(fpr_list):
        raise ValueError("tpr_list and fpr_list must have equal length.")
    if len(fpr_list) < 2:
        return 0.0

    points = sorted(zip((float(x) for x in fpr_list), (float(y) for y in tpr_list)))
    area = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        area += (x1 - x0) * (y0 + y1) / 2.0
    return area


def attack_advantage(tpr_at_low_fpr: float) -> float:
    """Membership-inference advantage in the low-FPR regime.

    At a fixed, tiny FPR (conventionally 0.1%), the false-positive contribution is
    negligible, so the attacker's advantage over random guessing is dominated by the
    true-positive rate. We report that TPR directly, clamped to ``[0, 1]``.
    """
    return max(0.0, min(1.0, float(tpr_at_low_fpr)))


def membership_inference_report(
    attack_name: str,
    auc: float,
    advantage: float,
) -> str:
    """Render a short, human-readable membership-inference result summary."""
    if auc <= 0.55:
        verdict = "no meaningful leakage (indistinguishable from a coin flip)"
    elif auc < 0.7:
        verdict = "weak but measurable leakage"
    elif auc < 0.85:
        verdict = "clear leakage -- membership is recoverable"
    else:
        verdict = "severe leakage -- membership is highly recoverable"

    return (
        f"Membership Inference Report: {attack_name}\n"
        f"{'-' * (28 + len(attack_name))}\n"
        f"  AUC                 : {auc:.4f}\n"
        f"  Advantage (TPR@low FPR): {advantage:.4f}\n"
        f"  Verdict             : {verdict}"
    )
