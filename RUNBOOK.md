# Runbook — model-privacy-attacks

Operational guide for developing, testing, and running the model-privacy-attacks framework.

---

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.10+ | Required. On Windows use `py`; on Linux/macOS use `python3`. |
| pip | Bundled with Python. |
| GNU Make | Optional. All `make` targets have manual equivalents below. |
| Git | For version control and pre-commit hooks. |

## Installation

### Clone and install (editable, with dev dependencies)

**Linux / macOS:**
```bash
cd model-privacy-attacks
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

**Windows (PowerShell):**
```powershell
cd model-privacy-attacks
py -m pip install --upgrade pip
py -m pip install -e ".[dev]"
```

This installs `numpy`, `scikit-learn`, plus dev tools (`pytest`, `pytest-cov`, `pytest-asyncio`).

### Additional tooling (lint, build, security)

**Linux / macOS:**
```bash
python3 -m pip install build ruff bandit pip-audit
```

**Windows:**
```powershell
py -m pip install build ruff bandit pip-audit
```

---

## Running Tests

**Linux / macOS:**
```bash
PYTHONPATH=src pytest tests/ -v --cov
```

**Windows (PowerShell):**
```powershell
$env:PYTHONPATH="src"; pytest tests/ -v --cov
```

Run a single test file:
```bash
# Linux / macOS
pytest tests/test_privacy_attacks.py -v

# Windows
pytest tests\test_privacy_attacks.py -v
```

The CI enforces a minimum coverage gate of 30% (`--cov-fail-under=30`).

---

## Running Individual Attack Demos

All attacks are black-box and work on synthetic data out of the box (no model downloads required).

### Membership Inference Attacks (MIA)

**Linux / macOS:**
```bash
python3 -m pytest tests/test_privacy_attacks.py -v -k "mia"
```

**Windows:**
```powershell
py -m pytest tests/test_privacy_attacks.py -v -k "mia"
```

### LLM Membership Inference (Min-K% Prob)

```bash
# Synthetic (no model download needed)
python3 examples/llm_mia_demo.py

# With real GPT-2 log-probs (needs transformers + torch)
python3 examples/llm_mia_demo.py --real
```

**Windows:**
```powershell
py examples\llm_mia_demo.py
py examples\llm_mia_demo.py --real
```

### Model Extraction Attack

```bash
python3 -m pytest tests/test_privacy_attacks.py -v -k "extraction"
```

### Sklearn Benchmark (Iris dataset — real model)

**Linux / macOS:**
```bash
python3 benchmark/run_sklearn_benchmark.py
```

**Windows:**
```powershell
py benchmark\run_sklearn_benchmark.py
```

Output is written to `results/iris_mia_benchmark.json`.

### Verify DP-SGD Epsilon Accounting

**Linux / macOS:**
```bash
python3 scripts/verify_epsilon.py
```

**Windows:**
```powershell
py scripts\verify_epsilon.py
```

Output: `results/epsilon_verification.json`.

---

## Sibling Dependency: attack-v19-core

The `attack-v19-core` repo provides ATT&CK v19 mappings and is expected at `../attack-v19-core` relative to this repository.

**This dependency is optional.** Tests that require it use `pytest.importorskip` and will skip gracefully if `attack_v19_core` is not installed.

To install it (when the sibling repo is cloned):

**Linux / macOS:**
```bash
python3 -m pip install -e "../attack-v19-core"
```

**Windows:**
```powershell
py -m pip install -e "..\attack-v19-core"
```

Or via Make:
```bash
make install-core
```

---

## Make Targets

All targets use the `PYTHON` variable (defaults to `python`). Override on Windows: `make PYTHON=py <target>`.

| Target | Description |
|--------|-------------|
| `make install` | Upgrade pip, install requirements.txt, install package editable, install build/ruff/bandit/pip-audit. |
| `make install-core` | Install the sibling `attack-v19-core` package from `../attack-v19-core`. |
| `make data` | Download ATT&CK data via `attack-v19-core/scripts/download_attack_data.py`. |
| `make lint` | Run `ruff check` on `src/privacy_attacks`, `attack_mapping`, and `tests`. |
| `make format` | Run `ruff format` on the same source directories. |
| `make test` | Install core + download data, then run `pytest tests -q`. |
| `make build` | Build the Python distribution package (`python -m build`). |
| `make security` | Run `bandit` (SAST) and `pip-audit` (dependency vulnerabilities). |
| `make dashboard` | Serve the 3D dashboard at `http://localhost:8080` from `dashboard/`. |
| `make verify` | Full local gate: lint → test → build → security. |

---

## Troubleshooting

### `python` / `python3` not found on Windows

Windows Python Launcher uses `py`. Replace all `python3` commands with `py`:
```powershell
py -m pip install -e ".[dev]"
py -m pytest tests/ -v
```

### `ModuleNotFoundError: No module named 'privacy_attacks'`

The package must be installed in editable mode or `PYTHONPATH` must include `src/`:
```bash
# Fix: install editable
pip install -e ".[dev]"

# Or set PYTHONPATH
export PYTHONPATH=src   # Linux/macOS
$env:PYTHONPATH="src"   # Windows PowerShell
```

### Tests skip with "attack_v19_core is optional and not published on PyPI"

This is expected. The `test_attack_mapping.py` tests require the sibling `attack-v19-core` package. Install it with `make install-core` or `pip install -e "../attack-v19-core"` if the repo is cloned alongside this one.

### `make` not available on Windows

Install Make via [chocolatey](https://chocolatey.org/) (`choco install make`) or run commands manually as shown in the sections above. All Make targets have direct equivalents.

### Permission errors during `pip install -e .`

Use a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\Activate.ps1       # Windows PowerShell
pip install -e ".[dev]"
```

### Dashboard won't load

Ensure port 8080 is free, then:
```bash
python3 -m http.server 8080 --directory dashboard   # Linux/macOS
py -m http.server 8080 --directory dashboard         # Windows
```

Open `http://localhost:8080` in a browser.

---

## Notes

- On Windows, always use `py` instead of `python` or `python3`.
- The test suite (18 tests) runs on synthetic data — no GPU, model downloads, or external datasets required.
- Local dashboard scores are evidence indicators, not production certifications.
- Re-check CI (GitHub Actions) after pushing to main for authoritative pass/fail.
