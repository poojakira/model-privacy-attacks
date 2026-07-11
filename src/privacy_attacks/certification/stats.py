"""Statistical primitives for privacy certification.

Design stance
-------------
Every number this module emits is meant to survive a hostile statistician. That means:

* AUC is the rank-based (Mann-Whitney U) estimator, tie-aware and vectorized.
* Uncertainty is reported as a bootstrap confidence interval, never a bare point.
* The "is there leakage beyond a blind baseline?" question is answered with a *paired*
  bootstrap test on the AUC difference -- resampling the same sample indices for both
  the model-based attack and the blind baseline so their correlation is preserved. We
  deliberately avoid a hand-rolled DeLong covariance estimator: the paired bootstrap is
  harder to get subtly wrong and makes weaker assumptions.

All randomness is seeded, so a certificate is reproducible bit-for-bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based ROC-AUC (probability a positive outranks a negative; ties = 0.5).

    ``labels`` are 1 for the positive class (member) and 0 for negative (non-member).
    Vectorized via average ranks so it is fast enough for thousands of bootstrap rounds.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    # Average ranks (1-based), handling ties by assigning the mean rank within a tie run.
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    n = len(scores)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # +1 for 1-based ranks
        ranks[i : j + 1] = avg_rank
        i = j + 1
    rank_of = np.empty(n, dtype=float)
    rank_of[order] = ranks

    sum_ranks_pos = rank_of[labels == 1].sum()
    u = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
    """True-positive rate at a fixed, low false-positive rate.

    The threshold is set from the non-member (negative) score distribution so that at
    most ``fpr`` of non-members are flagged, then we report the fraction of members
    above it. This is the modern MI metric (Carlini et al., 2022): a high average AUC is
    worthless if the attacker cannot expose members without a flood of false alarms.
    """
    tpr, _, _ = tpr_at_fpr_detailed(scores, labels, fpr)
    return tpr


def tpr_at_fpr_detailed(
    scores: np.ndarray, labels: np.ndarray, fpr: float
) -> tuple[float, float, float]:
    """Return ``(tpr, realized_fpr, threshold)``.

    The *realized* FPR is reported honestly: at small non-member counts the requested
    ``fpr`` cannot be achieved (its granularity is 1 / n_neg), so the actual FPR at the
    chosen threshold is what matters, not the nominal target.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0, 0.0, float("nan")
    threshold = float(np.quantile(neg, 1.0 - fpr, method="higher"))
    realized_fpr = float(np.mean(neg > threshold))
    tpr = float(np.mean(pos > threshold))
    return tpr, realized_fpr, threshold


def bootstrap_tpr_difference(
    model_scores: np.ndarray,
    blind_scores: np.ndarray,
    labels: np.ndarray,
    fpr: float,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Bootstrap CI for ``model_tpr@fpr - blind_tpr@fpr`` (paired, same resample).

    This replaces subtracting two noisy point estimates: the difference is resampled
    jointly so its uncertainty is quantified rather than hidden.
    """
    model_scores = np.asarray(model_scores, dtype=float)
    blind_scores = np.asarray(blind_scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    point = tpr_at_fpr(model_scores, labels, fpr) - tpr_at_fpr(
        blind_scores, labels, fpr
    )
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _stratified_resample(n_pos, n_neg, rng)
        y = labels[idx]
        diffs[b] = tpr_at_fpr(model_scores[idx], y, fpr) - tpr_at_fpr(
            blind_scores[idx], y, fpr
        )
    alpha = 1.0 - confidence
    lo = float(np.quantile(diffs, alpha / 2.0))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    return Interval(point=point, ci_low=lo, ci_high=hi, confidence=confidence)


@dataclass
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    point: float
    ci_low: float
    ci_high: float
    confidence: float

    def as_dict(self) -> dict:
        return {
            "point": round(self.point, 6),
            "ci_low": round(self.ci_low, 6),
            "ci_high": round(self.ci_high, 6),
            "confidence": self.confidence,
        }


def _stratified_resample(
    n_pos: int, n_neg: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices for a stratified bootstrap that preserves the member/non-member counts.

    Assumes the score array is laid out as ``[positives..., negatives...]``.
    """
    pos_idx = rng.integers(0, n_pos, size=n_pos)
    neg_idx = n_pos + rng.integers(0, n_neg, size=n_neg)
    return np.concatenate([pos_idx, neg_idx])


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    metric: str = "auc",
    fpr: float = 0.001,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Stratified bootstrap CI for ``auc`` or ``tpr_at_fpr``.

    Scores/labels must be ordered positives-first for the stratified resampler.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    def compute(s: np.ndarray, y: np.ndarray) -> float:
        if metric == "auc":
            return roc_auc(s, y)
        if metric == "tpr_at_fpr":
            return tpr_at_fpr(s, y, fpr)
        raise ValueError(f"unknown metric {metric!r}")

    point = compute(scores, labels)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _stratified_resample(n_pos, n_neg, rng)
        boot[b] = compute(scores[idx], labels[idx])

    alpha = 1.0 - confidence
    lo = float(np.quantile(boot, alpha / 2.0))
    hi = float(np.quantile(boot, 1.0 - alpha / 2.0))
    return Interval(point=point, ci_low=lo, ci_high=hi, confidence=confidence)


@dataclass
class PairedAUCTest:
    """Result of a paired bootstrap test for ``model_auc - blind_auc``."""

    model_auc: float
    blind_auc: float
    delta: float
    delta_ci_low: float
    delta_ci_high: float
    p_value: float
    confidence: float

    def as_dict(self) -> dict:
        return {
            "model_auc": round(self.model_auc, 6),
            "blind_auc": round(self.blind_auc, 6),
            "delta_auc": round(self.delta, 6),
            "delta_ci_low": round(self.delta_ci_low, 6),
            "delta_ci_high": round(self.delta_ci_high, 6),
            "p_value": round(self.p_value, 6),
            "confidence": self.confidence,
        }


def paired_delta_auc_test(
    model_scores: np.ndarray,
    blind_scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedAUCTest:
    """Test whether the model-based attack out-separates a blind baseline.

    Both score vectors are aligned to the same samples and the same ``labels``. We
    resample sample indices *once per bootstrap round* and recompute both AUCs on that
    identical resample, so the (positive) correlation between the two estimators is
    preserved -- the whole point of a paired test.

    Returns the observed delta, its bootstrap CI, and a one-sided p-value for
    H1: model_auc > blind_auc. The p-value is the bootstrap mass at delta <= 0, with
    add-one smoothing so it is never exactly zero.
    """
    model_scores = np.asarray(model_scores, dtype=float)
    blind_scores = np.asarray(blind_scores, dtype=float)
    labels = np.asarray(labels).astype(int)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())

    model_auc = roc_auc(model_scores, labels)
    blind_auc = roc_auc(blind_scores, labels)
    delta = model_auc - blind_auc

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = _stratified_resample(n_pos, n_neg, rng)
        y = labels[idx]
        deltas[b] = roc_auc(model_scores[idx], y) - roc_auc(blind_scores[idx], y)

    alpha = 1.0 - confidence
    ci_low = float(np.quantile(deltas, alpha / 2.0))
    ci_high = float(np.quantile(deltas, 1.0 - alpha / 2.0))
    # One-sided p-value for H1: delta > 0.
    p_value = float((np.sum(deltas <= 0.0) + 1) / (n_boot + 1))
    return PairedAUCTest(
        model_auc=model_auc,
        blind_auc=blind_auc,
        delta=delta,
        delta_ci_low=ci_low,
        delta_ci_high=ci_high,
        p_value=p_value,
        confidence=confidence,
    )
