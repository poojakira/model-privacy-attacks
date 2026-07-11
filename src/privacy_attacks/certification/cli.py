"""Command-line privacy-certification gate.

Usage
-----
Precompute your attack scores and inputs into an ``.npz`` with arrays:
``member_scores``, ``nonmember_scores``, ``member_features``, ``nonmember_features``.
Then::

    python -m privacy_attacks.certify audit run.npz --policy policy.yaml --out cert.json

Exit codes: 0 = policy passed (or no policy), 2 = policy violated, 1 = usage/data error.
The non-zero exit on violation is what makes this a CI gate rather than a report.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from privacy_attacks.certification.certificate import CertificateConfig, certify
from privacy_attacks.certification.policy import evaluate_policy, load_policy

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_POLICY_VIOLATION = 2


def _load_npz(path: str):
    data = np.load(path, allow_pickle=False)
    required = [
        "member_scores",
        "nonmember_scores",
        "member_features",
        "nonmember_features",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path} missing arrays: {missing}. Required: {required}")
    return data


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="privacy-certify",
        description="Null-calibrated privacy certificate + CI policy gate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    audit = sub.add_parser("audit", help="Certify a precomputed attack run.")
    audit.add_argument("run", help="Path to .npz with scores + features.")
    audit.add_argument("--policy", help="Path to a JSON/YAML policy file.")
    audit.add_argument("--out", help="Write the certificate JSON to this path.")
    audit.add_argument("--attack-name", default="max_confidence")
    audit.add_argument("--target-id", default="unknown-target")
    audit.add_argument("--fpr", type=float, default=0.001)
    audit.add_argument("--alpha", type=float, default=0.05)
    audit.add_argument("--n-boot", type=int, default=2000)
    audit.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        data = _load_npz(args.run)
        member_scores = np.asarray(data["member_scores"], dtype=float)
        nonmember_scores = np.asarray(data["nonmember_scores"], dtype=float)
        attack_scores = np.concatenate([member_scores, nonmember_scores])
        config = CertificateConfig(
            attack_name=args.attack_name,
            fpr_target=args.fpr,
            alpha=args.alpha,
            n_boot=args.n_boot,
            seed=args.seed,
            target_id=args.target_id,
        )
        cert = certify(
            attack_scores=attack_scores,
            member_features=data["member_features"],
            nonmember_features=data["nonmember_features"],
            n_members=len(member_scores),
            config=config,
        )
    except (KeyError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(_render(cert))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(cert.to_json())
        print(f"\nCertificate written to {args.out}")

    if args.policy:
        result = evaluate_policy(cert.to_dict(), load_policy(args.policy))
        print("\nPolicy gate:")
        print(result.summary())
        if not result.passed:
            return EXIT_POLICY_VIOLATION

    return EXIT_OK


def _render(cert) -> str:
    d = cert.to_dict()
    lf = d["low_fpr"]
    if lf["reliable"] and lf["certified_tpr_difference"]:
        tpr_line = (
            f"  certified TPR diff: {lf['certified_tpr_difference']['point']:.4f} "
            f"[{lf['certified_tpr_difference']['ci_low']:.4f}, "
            f"{lf['certified_tpr_difference']['ci_high']:.4f}] "
            f"(realized FPR {lf['realized_fpr']:g})\n"
        )
    else:
        tpr_line = f"  low-FPR metric    : unavailable ({lf['note']})\n"
    return (
        "Privacy Certificate\n"
        "===================\n"
        f"  target            : {d['target_id']}\n"
        f"  attack            : {d['attack_name']}\n"
        f"  samples           : {d['n_members']} members / {d['n_nonmembers']} non-members\n"
        f"  model AUC         : {d['model_auc']['point']:.4f} "
        f"[{d['model_auc']['ci_low']:.4f}, {d['model_auc']['ci_high']:.4f}]\n"
        f"  blind AUC (null)  : {d['blind_auc']['point']:.4f} "
        f"(best='{d['blind_auc']['best_baseline']}', panel={d['blind_auc']['panel_aucs']})\n"
        f"  delta AUC         : {d['delta_auc']['delta_auc']:.4f} "
        f"[{d['delta_auc']['delta_ci_low']:.4f}, {d['delta_auc']['delta_ci_high']:.4f}]"
        f"  p={d['delta_auc']['p_value']:.4f}\n"
        f"{tpr_line}"
        f"  VERDICT           : {d['verdict']}\n"
        f"  rationale         : {d['verdict_rationale']}\n"
        f"  integrity         : {d['integrity']['algorithm']}:"
        f"{d['integrity']['digest'][:16]}... kind={d['integrity']['kind']} "
        f"signed={d['integrity']['signed']}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
