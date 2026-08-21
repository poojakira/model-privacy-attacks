# Security Policy

## Overview

`model-privacy-attacks` is a **privacy risk assessment framework**  --  it implements published academic attacks as compliance measurement tools. This policy covers responsible disclosure for vulnerabilities in the framework itself (not in third-party models assessed with it).

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| `main` branch | ✅ Active |
| Tagged releases | ✅ Within 90 days of release |
| Older releases | ❌ No security patches |

---

## Scope

### In Scope

Vulnerabilities in this repository's own code and tooling, including:

- **Incorrect epsilon calculations** in `budget_calculator.py` that underestimate privacy risk (false LOW classification when true risk is HIGH)
- **Report generation logic errors** in `report.py` that suppress EU AI Act article triggers or produce incorrect risk levels
- **Dependency vulnerabilities** in pinned packages (`numpy`, `scikit-learn`, `torch`)
- **CI/CD pipeline weaknesses** (e.g., SARIF upload bypass, coverage gate bypass)
- **Path traversal or arbitrary file write** in report output paths
- **Injection vulnerabilities** in model ID or report field inputs that reach shell commands or file paths

### Out of Scope

- Privacy vulnerabilities in *models you assess using this tool*  --  that is the tool's intended purpose, not a security bug
- Social engineering, phishing, or physical access
- Issues in third-party dependencies without a disclosed CVE
- Theoretical attacks that require pre-existing compromise of the assessment environment

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities via one of the following channels:

1. **GitHub Private Security Advisory** (preferred):
   Navigate to `Security` → `Advisories` → `New draft security advisory` in this repository.

2. **Email**: Send a description to the repository maintainer. Include `[SECURITY] model-privacy-attacks` in the subject line.

### What to include

- Description of the vulnerability and its impact
- Affected component (file, function, version)
- Steps to reproduce (minimal reproduction case preferred)
- Suggested severity (CVSS score if available)
- Whether you believe it is exploitable in a typical assessment workflow

---

## Response Timeline

| Stage | Target SLA |
|-------|-----------|
| Acknowledgment | 72 hours |
| Initial triage and severity assessment | 5 business days |
| Patch or mitigation | 14 business days for HIGH/CRITICAL |
| Public disclosure | Coordinated  --  typically 90 days after report |

For CRITICAL vulnerabilities (e.g., silent epsilon underestimation leading to incorrect LOW classification), we will issue a patch and changelog entry within 7 business days.

---

## Severity Guidance

We use CVSS 3.1 as a reference. For this tool's threat model, severity is elevated for:

- **Silent underestimation of privacy risk**: A bug that causes `risk_level` to return `LOW` when the correct answer is `HIGH` or `CRITICAL` is a compliance integrity failure. We treat these as HIGH severity regardless of CVSS base score.
- **Incorrect EU AI Act article suppression**: A bug that prevents triggered articles from appearing in the report when they should could expose users to regulatory non-compliance.

---

## Acknowledgments

We will credit reporters in the relevant changelog entry and release notes, unless the reporter requests anonymity.

---

## Related: Tool Threat Model

See [THREAT_MODEL.md](THREAT_MODEL.md) for a structured analysis of threats to the assessment framework itself, including adversarial model inputs, report manipulation, and supply chain risks.
