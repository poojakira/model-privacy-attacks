"""
Differential privacy budget calculator.

Estimates the privacy loss (epsilon, delta) for a training run using the
Gaussian mechanism, given dataset size, epochs, batch size, and noise multiplier.

This is a developer-facing self-service tool: calculate your privacy exposure
BEFORE training, so you can tune sigma to meet your target epsilon.

Usage:
    from privacy_attacks.budget_calculator import privacy_budget_calculator

    result = privacy_budget_calculator(
        dataset_size=50000,
        epochs=10,
        batch_size=256,
        sigma=1.0,
    )
    print(result["epsilon"])      # e.g., 3.2
    print(result["risk_level"])   # MEDIUM
    print(result["recommendation"])

Or from the command line:
    python -m privacy_attacks.budget_calculator \\
        --dataset-size 50000 --epochs 10 --batch-size 256 --sigma 1.0
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any


def _gaussian_epsilon_approximation(
    n: int,
    batch_size: int,
    epochs: int,
    sigma: float,
    delta: float,
) -> float:
    """
    Approximate epsilon for DP-SGD using the moments accountant / RDP framework
    (simplified analytical approximation).

    This is a conservative approximation suitable for pre-training estimates.
    For production privacy budgets, use the `autodp` or `prv_accountant` library
    for tighter bounds.

    Based on: Mironov (2017) Rényi Differential Privacy of the Gaussian Mechanism.
    Simplified from: Abadi et al. (2016) "Deep Learning with Differential Privacy."

    Args:
        n: Dataset size (number of training examples).
        batch_size: Batch size used during training.
        epochs: Number of training epochs.
        sigma: Noise multiplier (standard deviation / sensitivity).
               Higher sigma = more noise = smaller epsilon (more privacy).
        delta: Target delta (probability of privacy failure). Typical: 1e-5.

    Returns:
        float: Estimated epsilon (privacy loss). Lower is better.
               epsilon <= 1.0: Strong privacy. epsilon 1–3: Moderate. > 3: Weak.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if n <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("n, batch_size, and epochs must be positive integers")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    # Sampling ratio (probability of each example being in a batch)
    q = batch_size / n

    # Number of gradient steps
    steps = int(math.ceil(epochs * n / batch_size))

    # Gaussian mechanism sensitivity-to-noise ratio
    z = sigma

    # Conservative epsilon using the advanced composition theorem
    # (tighter than naive composition; looser than moments accountant)
    # epsilon ≈ q * sqrt(2 * steps * log(1/delta)) / z
    # This is a simplified bound; use autodp for tighter analysis.
    eps_numerator = q * math.sqrt(2.0 * steps * math.log(1.0 / delta))
    epsilon = eps_numerator / z

    return max(0.0, epsilon)


def _risk_level_from_epsilon(epsilon: float) -> str:
    """
    Classify privacy risk based on achieved epsilon.

    Risk thresholds (based on practical DP-ML literature):
      LOW      — epsilon <= 1.0  (strong privacy; recommended for sensitive data)
      MEDIUM   — epsilon <= 3.0  (moderate; acceptable for non-sensitive data)
      HIGH     — epsilon <= 10.0 (weak; privacy guarantees are limited)
      CRITICAL — epsilon > 10.0  (near-useless privacy guarantee)
    """
    if epsilon <= 1.0:
        return "LOW"
    elif epsilon <= 3.0:
        return "MEDIUM"
    elif epsilon <= 10.0:
        return "HIGH"
    else:
        return "CRITICAL"


def _generate_recommendation(
    epsilon: float,
    sigma: float,
    dataset_size: int,
    epochs: int,
    batch_size: int,
    delta: float,
    risk_level: str,
) -> str:
    """
    Generate a specific, actionable recommendation for the given privacy budget.
    """
    if risk_level == "LOW":
        return (
            f"Epsilon = {epsilon:.2f} meets the recommended threshold of <= 1.0. "
            f"Current sigma = {sigma:.1f} provides strong differential privacy guarantees."
        )

    # Calculate what sigma would be needed to reach epsilon = 1.0
    target_epsilon = 1.0
    q = batch_size / dataset_size
    steps = int(math.ceil(epochs * dataset_size / batch_size))
    required_sigma = q * math.sqrt(2.0 * steps * math.log(1.0 / delta)) / target_epsilon

    if risk_level == "MEDIUM":
        return (
            f"Epsilon = {epsilon:.2f} is acceptable for non-sensitive data. "
            f"To reach epsilon <= 1.0 (LOW risk), increase sigma to >= {required_sigma:.1f}. "
            f"Note: higher sigma reduces model utility."
        )
    else:
        return (
            f"Epsilon = {epsilon:.2f} provides weak privacy guarantees. "
            f"To reach epsilon <= 1.0, increase sigma to >= {required_sigma:.1f} "
            f"or reduce epochs (current: {epochs}). "
            f"Consider reducing epochs to {max(1, epochs // 2)} and re-evaluating."
        )


def privacy_budget_calculator(
    dataset_size: int,
    epochs: int,
    batch_size: int,
    sigma: float,
    delta: float = 1e-5,
) -> dict[str, Any]:
    """
    Calculate the differential privacy budget for a DP-SGD training run.

    Use this BEFORE training to understand your privacy exposure and tune sigma
    to meet your target epsilon. This is the paved path: design privacy in,
    don't measure it after the fact.

    Args:
        dataset_size: Number of training examples.
        epochs: Number of training epochs planned.
        batch_size: Mini-batch size used in training.
        sigma: Noise multiplier for the Gaussian mechanism.
               Typical range: 0.5 (weak privacy) to 4.0 (strong privacy).
               Higher sigma = more noise = smaller epsilon = stronger privacy
               but lower model accuracy.
        delta: Target delta (default: 1e-5). Represents the probability that
               the epsilon bound is violated. Typically set to 1/dataset_size
               or 1e-5 for large datasets.

    Returns:
        dict with keys:
            epsilon (float): Estimated privacy loss. Lower is better.
            delta (float): The delta value used.
            risk_level (str): LOW / MEDIUM / HIGH / CRITICAL.
            recommendation (str): Actionable next steps.
            noise_multiplier (float): The sigma value used.
            effective_batch_size (int): batch_size as provided.
            sampling_rate (float): batch_size / dataset_size.
            num_steps (int): Total gradient steps (epochs * n / batch_size).

    Example:
        >>> result = privacy_budget_calculator(50000, 10, 256, 1.0)
        >>> result["risk_level"]
        'MEDIUM'
        >>> result = privacy_budget_calculator(50000, 10, 256, 2.0)
        >>> result["risk_level"]
        'LOW'

    Raises:
        ValueError: If any parameter is invalid (negative, zero, out of range).
    """
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if batch_size > dataset_size:
        raise ValueError(
            f"batch_size ({batch_size}) cannot exceed dataset_size ({dataset_size})"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    if not 0 < delta < 1:
        raise ValueError(f"delta must be in (0, 1), got {delta}")

    epsilon = _gaussian_epsilon_approximation(
        n=dataset_size,
        batch_size=batch_size,
        epochs=epochs,
        sigma=sigma,
        delta=delta,
    )

    risk_level = _risk_level_from_epsilon(epsilon)
    num_steps = int(math.ceil(epochs * dataset_size / batch_size))
    sampling_rate = batch_size / dataset_size

    recommendation = _generate_recommendation(
        epsilon=epsilon,
        sigma=sigma,
        dataset_size=dataset_size,
        epochs=epochs,
        batch_size=batch_size,
        delta=delta,
        risk_level=risk_level,
    )

    return {
        "epsilon": round(epsilon, 4),
        "delta": delta,
        "risk_level": risk_level,
        "recommendation": recommendation,
        "noise_multiplier": sigma,
        "effective_batch_size": batch_size,
        "sampling_rate": round(sampling_rate, 6),
        "num_steps": num_steps,
        "note": (
            "Epsilon estimated using simplified Gaussian mechanism approximation. "
            "For production privacy budgets, use autodp or prv_accountant for tighter bounds."
        ),
    }


def _main() -> None:
    """CLI entry point for privacy budget calculator."""
    parser = argparse.ArgumentParser(
        description="Calculate differential privacy budget for a DP-SGD training run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if sigma=1.0 gives acceptable privacy
  python -m privacy_attacks.budget_calculator \\
    --dataset-size 50000 --epochs 10 --batch-size 256 --sigma 1.0

  # Find sigma needed for LOW risk
  python -m privacy_attacks.budget_calculator \\
    --dataset-size 50000 --epochs 10 --batch-size 256 --sigma 2.0
        """,
    )
    parser.add_argument("--dataset-size", type=int, required=True, help="Number of training examples")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, required=True, help="Mini-batch size")
    parser.add_argument("--sigma", type=float, required=True, help="DP-SGD noise multiplier")
    parser.add_argument("--delta", type=float, default=1e-5, help="Target delta (default: 1e-5)")
    parser.add_argument("--output", type=str, default=None, help="Write result to JSON file")

    args = parser.parse_args()

    result = privacy_budget_calculator(
        dataset_size=args.dataset_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sigma=args.sigma,
        delta=args.delta,
    )

    print(json.dumps(result, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\nResult written to {args.output}")


if __name__ == "__main__":
    _main()
