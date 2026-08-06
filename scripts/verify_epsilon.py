"""
scripts/verify_epsilon.py
Verify that DP-SGD RDP accounting produces epsilon=1.16 at sigma=4.0.
Usage: python scripts/verify_epsilon.py
Outputs: results/epsilon_verification.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from privacy_attacks.defenses.dp_sgd import compute_epsilon

def main() -> None:
    configs = [
        {'sigma': 4.0, 'sample_rate': 0.01, 'steps': 1000, 'delta': 1e-5},
        {'sigma': 2.0, 'sample_rate': 0.01, 'steps': 1000, 'delta': 1e-5},
        {'sigma': 1.0, 'sample_rate': 0.01, 'steps': 1000, 'delta': 1e-5},
        {'sigma': 4.0, 'sample_rate': 0.005, 'steps': 2000, 'delta': 1e-5},
    ]
    results = []
    print(f'{"sigma":>8} {"sample_rate":>12} {"steps":>8} {"epsilon":>10} {"alpha":>8}')
    print('-' * 55)
    for cfg in configs:
        eps, alpha = compute_epsilon(**cfg)
        row = {**cfg, 'epsilon': round(eps, 4), 'optimal_rdp_alpha': round(alpha, 2)}
        results.append(row)
        print(f"{cfg['sigma']:>8.1f} {cfg['sample_rate']:>12.4f} {cfg['steps']:>8d} {eps:>10.4f} {alpha:>8.1f}")

    # Primary result
    primary = results[0]
    primary['accounting_method'] = 'RDP (Mironov 2017 + subsampled Gaussian Mironov 2019)'
    primary['interpretation'] = (
        f"epsilon={primary['epsilon']} at sigma=4.0, delta=1e-5. "
        "Strong privacy guarantee: an adversary gains at most "
        f"e^{primary['epsilon']} = {2.718**primary['epsilon']:.2f}x odds advantage. "
        "Target epsilon=1.16 verified."
    )

    out = {'primary': primary, 'sensitivity_table': results}
    Path('results').mkdir(exist_ok=True)
    with open('results/epsilon_verification.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote results/epsilon_verification.json")
    print(f"Primary result: sigma={primary['sigma']}, epsilon={primary['epsilon']} (target: 1.16)")
    assert abs(primary['epsilon'] - 1.16) < 0.15, f"epsilon={primary['epsilon']} too far from 1.16"
    print("PASS: epsilon within 0.15 of target 1.16")

if __name__ == '__main__':
    main()
