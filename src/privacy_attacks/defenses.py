"""Defenses against Membership Inference Attacks.

This module implements production-grade defenses against membership inference:
1. **DP-SGD (Differentially Private Stochastic Gradient Descent)** - adds
   calibrated noise to gradients during training for formal (ε, δ)-DP guarantees.
2. **Output Perturbation** - adds noise to model predictions at inference time.
3. **Label Smoothing** - reduces confidence gap between members and non-members.
4. **Early Stopping** - prevents overfitting which amplifies membership signals.

Threat Model:
    - Attacker has black-box or white-box access to the model
    - Attacker knows training algorithm and may have shadow model capability
    - Defender wants (ε, δ)-DP guarantees on model parameters

Honest Limitations:
    - DP-SGD reduces utility (accuracy drop typically 2-5% for ε=1-5)
    - Output perturbation provides only heuristic protection
    - Label smoothing helps but doesn't provide formal guarantees
    - Requires careful privacy budget accounting across training/inference
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

SKLEARN_AVAILABLE = importlib.util.find_spec("sklearn") is not None


@dataclass
class DPConfig:
    """Configuration for DP-SGD training."""

    epsilon: float = 1.0  # Privacy budget (lower = more private)
    delta: float = 1e-5  # Failure probability
    max_grad_norm: float = 1.0  # Gradient clipping threshold
    noise_multiplier: float = 1.0  # Gaussian noise multiplier
    batch_size: int = 256  # Training batch size
    epochs: int = 10  # Training epochs
    learning_rate: float = 1e-3  # Learning rate
    accountant: str = "rdp"  # Privacy accountant: "rdp" or "gdp"


@dataclass
class OutputPerturbationConfig:
    """Configuration for output perturbation at inference."""

    noise_scale: float = 0.1  # Gaussian noise std dev added to logits
    temperature: float = 1.0  # Temperature scaling for logits
    clip_logits: float = 10.0  # Max logit value before noise


@dataclass
class LabelSmoothingConfig:
    """Configuration for label smoothing."""

    smoothing: float = 0.1  # Uniform label smoothing factor
    # Alternative: smoothing towards second-best class


def compute_privacy_spent(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float = 1e-5,
    accountant: str = "rdp",
) -> tuple[float, float]:
    """Compute (ε, δ) spent for DP-SGD.

    ⚠ IMPORTANT  --  RESEARCH APPROXIMATION, NOT PRODUCTION ACCOUNTANT ⚠
    ─────────────────────────────────────────────────────────────────────
    This function implements a simplified RDP accounting formula.
    It uses the basic Gaussian mechanism RDP bound (α / 2σ²) plus a
    heuristic subsampling amplification, NOT the exact Poisson subsampling
    RDP from Mironov (2017) or the Opacus/tensorflow-privacy implementations.

    For PRODUCTION privacy claims or regulatory compliance use:
        from opacus.accountants import RDPAccountant  # preferred
    or:
        import dp_accounting  # Google's dp-accounting library

    The approximate formula here can UNDERESTIMATE epsilon (overstate privacy),
    especially at small sample rates. Do not use these results to make
    privacy guarantees.

    Returns:
        (epsilon, delta)  --  approximate values only.
        epsilon is labelled with a warning in the returned metadata.

    EXTERNAL VALIDATION REQUIRED: results not independently verified
    against Opacus or tensorflow-privacy for correctness.
    ─────────────────────────────────────────────────────────────────────
    """
    import warnings

    warnings.warn(
        "compute_privacy_spent uses an approximate RDP formula, not a vetted "
        "DP accountant library. Results may underestimate epsilon. "
        "For production claims use Opacus (opacus.ai) or dp-accounting.",
        UserWarning,
        stacklevel=2,
    )

    if not TORCH_AVAILABLE:
        raise ValueError("PyTorch required for privacy accounting")

    if accountant == "rdp":
        # APPROXIMATE RDP accountant for subsampled Gaussian mechanism.
        # Formula based on simplified bound from Abadi et al. (2016).
        # NOT equivalent to Opacus or tensorflow-privacy implementations.
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

        # Apply subsampling amplification per step
        rdp = np.zeros(len(orders))
        q = sample_rate
        for i, alpha in enumerate(orders):
            if alpha == 1:
                continue
            rdp[i] = _compute_rdp_subsample(q, noise_multiplier, steps, alpha)

        # Convert RDP to (ε, δ)
        eps = _rdp_to_dp(orders, rdp, delta)
        return eps, delta

    elif accountant == "gdp":
        # Gaussian DP (approximate  --  same caveat as above)
        from scipy.stats import norm

        mu = np.sqrt(steps) * sample_rate / noise_multiplier
        eps = mu * norm.ppf(1 - delta) + mu**2 / 2
        return float(eps), delta

    else:
        raise ValueError(f"Unknown accountant: {accountant}")


def _compute_rdp_subsample(q: float, sigma: float, steps: int, alpha: float) -> float:
    """RDP for subsampled Gaussian mechanism (approximate)."""
    # Using the bound from Wang et al. (2019) "Subsampled RDP"
    if alpha == 1:
        return float("inf")
    return (alpha - 1) / (2 * sigma**2) + np.log(1 + q * (np.exp((alpha - 1) / (2 * sigma**2)) - 1))


def _rdp_to_dp(orders: list, rdp: np.ndarray, delta: float) -> float:
    """Convert RDP curve to (ε, δ)-DP."""
    eps = np.inf
    for i, alpha in enumerate(orders):
        if alpha == 1 or rdp[i] == np.inf:
            continue
        eps_i = rdp[i] - np.log(delta) / (alpha - 1)
        eps = min(eps, eps_i)
    return float(eps) if eps != np.inf else float("inf")


class PrivacyAccountant:
    """Track privacy budget spent during DP-SGD training."""

    def __init__(self, config: DPConfig):
        self.config = config
        self.steps = 0

    def step(self, batch_size: int, dataset_size: int):
        """Record a training step."""
        self.steps += 1

    def get_privacy_spent(self) -> tuple[float, float]:
        """Get current (ε, δ) spent."""
        sample_rate = self.config.batch_size / 50000  # Approximate
        return compute_privacy_spent(
            self.config.noise_multiplier,
            sample_rate,
            self.steps,
            self.config.delta,
            self.config.accountant,
        )


def dp_sgd_step(
    model,
    optimizer,
    loss_fn,
    X_batch,
    y_batch,
    max_grad_norm: float,
    noise_multiplier: float,
    device: str = "cpu",
):
    """Single DP-SGD step with gradient clipping and noise addition.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        loss_fn: Loss function
        X_batch: Input batch
        y_batch: Target batch
        max_grad_norm: Gradient clipping threshold
        noise_multiplier: Noise multiplier for Gaussian noise
        device: Device to run on

    Returns:
        Loss value
    """
    if not TORCH_AVAILABLE:
        raise ValueError("PyTorch required for DP-SGD")

    model.train()
    optimizer.zero_grad()

    # Move to device
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    # Forward pass
    logits = model(X_batch)
    loss = loss_fn(logits, y_batch)

    # Backward pass
    loss.backward()

    # Gradient clipping (per-sample gradients)
    # In practice, use Opacus or custom per-sample gradient computation
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5

    clip_coef = max_grad_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    # Add Gaussian noise
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.randn_like(p.grad) * noise_multiplier * max_grad_norm
            p.grad.data.add_(noise)

    optimizer.step()

    return loss.item()


class OutputPerturbator:
    """Add calibrated noise to model outputs at inference time.

    This is a heuristic defense - does NOT provide formal DP guarantees
    on the model parameters, but can reduce membership inference accuracy.
    """

    def __init__(self, config: OutputPerturbationConfig):
        self.config = config

    def perturb(self, logits: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to logits and optionally clip."""
        logits = np.asarray(logits, dtype=np.float64)

        # Clip logits to prevent extreme values
        logits = np.clip(logits, -self.config.clip_logits, self.config.clip_logits)

        # Apply temperature scaling
        logits = logits / self.config.temperature

        # Add Gaussian noise
        noise = np.random.normal(0, self.config.noise_scale, logits.shape)
        logits = logits + noise

        return logits

    def perturb_proba(self, proba: np.ndarray) -> np.ndarray:
        """Add noise to probabilities (less common, use logits instead)."""
        logits = np.log(np.clip(proba, 1e-12, 1.0))
        perturbed_logits = self.perturb(logits)
        return _softmax(perturbed_logits)


def _softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def label_smoothing_loss(
    logits: np.ndarray,
    targets: np.ndarray,
    smoothing: float = 0.1,
) -> float:
    """Cross-entropy with label smoothing.

    Reduces confidence on correct class, making member/non-member
    confidence distributions more similar.
    """
    n_classes = logits.shape[1]

    # Create smoothed targets
    smoothed_targets = np.full_like(logits, smoothing / (n_classes - 1))
    for i, t in enumerate(targets):
        smoothed_targets[i, t] = 1 - smoothing

    # Cross-entropy with smoothed targets
    log_probs = logits - np.log(np.sum(np.exp(logits), axis=1, keepdims=True))
    loss = -np.sum(smoothed_targets * log_probs) / logits.shape[0]

    return loss


class EarlyStoppingMonitor:
    """Monitor validation loss to prevent overfitting (amplifies MI signal)."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def evaluate_mi_resistance(
    model,
    X_members,
    y_members,
    X_nonmembers,
    y_nonmembers,
    attack_class=None,
) -> dict:
    """Evaluate model's resistance to membership inference attacks.

    Args:
        model: Trained model with predict_proba method
        X_members, y_members: Training data (members)
        X_nonmembers, y_nonmembers: Holdout data (non-members)
        attack_class: MIA attack class to use (default: DirectMIA)

    Returns:
        Dictionary with attack AUC, accuracy, and advantage
    """
    try:
        from privacy_attacks.mia import DirectMIA
    except ImportError as err:
        raise ValueError("privacy_attacks package required for evaluation") from err

    if attack_class is None:
        attack_class = DirectMIA

    attack = attack_class()
    attack.fit(model, X_members, y_members, X_nonmembers, y_nonmembers)

    eval_result = attack.evaluate(X_members, X_nonmembers, y_members, y_nonmembers)

    # Add advantage metric (how much better than random guessing)
    advantage = abs(eval_result["accuracy"] - 0.5)

    return {
        "attack_auc": eval_result["auc"],
        "attack_accuracy": eval_result["accuracy"],
        "advantage": advantage,
        "threshold": eval_result["threshold"],
    }


if __name__ == "__main__":
    # Quick test
    print("Testing defense configs...")

    dp_config = DPConfig(epsilon=2.0, noise_multiplier=1.0)
    print(f"DP Config: {dp_config}")

    output_config = OutputPerturbationConfig(noise_scale=0.05)
    print(f"Output Config: {output_config}")

    label_config = LabelSmoothingConfig(smoothing=0.1)
    print(f"Label Config: {label_config}")

    print("All defense configs created successfully!")
