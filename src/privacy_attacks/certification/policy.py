"""Policy evaluation -- turns a certificate into a pass/fail CI gate.

A policy is a small declarative document. Each rule names a numeric field of the
certificate and a threshold expression; ``verdict_not_in`` optionally forbids verdicts.
Any violated rule fails the gate (non-zero CLI exit), which is what lets a team wire
this into CI as a privacy-regression gate.

Example (YAML)::

    fail_if:
      certified_tpr_at_fpr: "> 0.05"
      delta_auc.delta_ci_low: "> 0.10"
    require_verdict_not_in: ["CERTIFIED_LEAKAGE"]
"""

from __future__ import annotations

import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}
_EXPR = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(-?\d+(?:\.\d+)?)\s*$")


@dataclass
class RuleResult:
    field: str
    expression: str
    actual: Any
    passed: bool


@dataclass
class PolicyResult:
    passed: bool
    rules: list[RuleResult]

    def summary(self) -> str:
        lines = [("PASS" if self.passed else "FAIL") + " overall"]
        for r in self.rules:
            status = "ok" if r.passed else "VIOLATED"
            lines.append(f"  [{status}] {r.field} {r.expression} (actual={r.actual})")
        return "\n".join(lines)


def load_policy(path: str) -> dict:
    """Load a policy from JSON or YAML (YAML requires PyYAML)."""
    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "PyYAML is required for YAML policies; use a .json policy or "
                "`pip install pyyaml`."
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _get_field(cert_dict: dict, dotted: str) -> Any:
    """Resolve a possibly-dotted field path, unwrapping ``{"point": ...}`` blocks."""
    node: Any = cert_dict
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(f"policy references unknown certificate field: {dotted!r}")
        node = node[part]
    if isinstance(node, dict) and "point" in node:
        node = node["point"]
    if node is None:
        raise ValueError(
            f"policy references {dotted!r} but that metric is unavailable for this "
            f"certificate (e.g. low-FPR metric at small sample size). Remove the rule "
            f"or increase the sample."
        )
    return node


def evaluate_policy(cert_dict: dict, policy: dict) -> PolicyResult:
    """Evaluate a policy against a certificate dict; return per-rule results."""
    rules: list[RuleResult] = []

    for field_name, expr in (policy.get("fail_if") or {}).items():
        match = _EXPR.match(str(expr))
        if not match:
            raise ValueError(f"invalid threshold expression {expr!r} for {field_name}")
        op_sym, num = match.group(1), float(match.group(2))
        actual = _get_field(cert_dict, field_name)
        # A "fail_if" rule violates the gate when the condition is TRUE.
        violated = _OPS[op_sym](float(actual), num)
        rules.append(
            RuleResult(field_name, str(expr), actual, passed=not violated)
        )

    forbidden = policy.get("require_verdict_not_in")
    if forbidden:
        verdict = cert_dict.get("verdict")
        rules.append(
            RuleResult(
                "verdict",
                f"not in {forbidden}",
                verdict,
                passed=verdict not in forbidden,
            )
        )

    return PolicyResult(passed=all(r.passed for r in rules), rules=rules)
