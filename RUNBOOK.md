# Runbook — Model Privacy Attacks

Step-by-step guide to run membership inference and model extraction attacks locally.

---

## Step 1: Prerequisites

- Python 3.10+ (`py --version` on Windows, `python3 --version` on Linux)
- pip (bundled with Python)
- Git
- Sibling repo `attack-v19-core` cloned alongside this repo (required for ATT&CK mapping tests)

Directory layout:
```
repos/
├── model-privacy-attacks/   ← you are here
└── attack-v19-core/         ← must exist for tests
```

---

## Step 2: Clone

**Windows (PowerShell):**
```powershell
cd C:\Users\pooja\repos
git clone https://github.com/poojakira/model-privacy-attacks.git
cd model-privacy-attacks
```

**Linux/macOS:**
```bash
cd ~/repos
git clone https://github.com/poojakira/model-privacy-attacks.git
cd model-privacy-attacks
```

---

## Step 3: Install

**Windows (PowerShell):**
```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"

# Install attack-v19-core (sibling dependency)
.\.venv\Scripts\python.exe -m pip install -e ..\attack-v19-core
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

# Install attack-v19-core (sibling dependency)
pip install -e ../attack-v19-core
```

**Or use Makefile (if `make` available):**
```powershell
make install
make install-core
```

---

## Step 4: Run

**Windows (PowerShell):**
```powershell
# Run membership inference attack (MIA)
.\.venv\Scripts\python.exe -m model_privacy_attacks.mia --dataset synthetic

# Run model extraction attack
.\.venv\Scripts\python.exe -m model_privacy_attacks.extraction --target-model mlp

# Run full evaluation suite
.\.venv\Scripts\python.exe -m model_privacy_attacks.evaluate
```

**Linux/macOS:**
```bash
python -m model_privacy_attacks.mia --dataset synthetic
python -m model_privacy_attacks.extraction --target-model mlp
python -m model_privacy_attacks.evaluate
```

**Or use Makefile:**
```powershell
make run       # Run default attack pipeline
make dashboard # Serve dashboard at localhost:8080
```

---

## Step 5: Expected Output

Membership inference attack:
```
[MIA] Dataset: synthetic (1000 members, 1000 non-members)
[MIA] Attack model: shadow model + threshold
[MIA] Accuracy: 0.62
[MIA] Precision: 0.59
[MIA] Recall: 0.65
[MIA] AUC: 0.64
```

Model extraction:
```
[Extraction] Target: MLP (3 layers)
[Extraction] Queries used: 5000
[Extraction] Clone fidelity: 0.89
[Extraction] Clone test accuracy: 0.85
```

> **Note:** All attacks run on **synthetic data**. No real user data is involved.

---

## Step 6: Run Tests

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

**Linux/macOS:**
```bash
pytest tests/ -v
```

**With coverage:**
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ --cov=src --cov-report=term-missing
```

**Full verification (lint + test + build + security):**
```powershell
make verify
```

---

## Available Makefile Targets

| Command | What it does |
|---------|-------------|
| `make install` | Install dependencies into venv |
| `make install-core` | Install attack-v19-core from sibling dir |
| `make test` | Run pytest |
| `make lint` | Run ruff linter |
| `make format` | Auto-format with ruff |
| `make build` | Build wheel package |
| `make security` | Run bandit + pip-audit |
| `make verify` | All of the above in sequence |
| `make dashboard` | Serve dashboard at localhost:8080 |

---

## View Dashboard

```powershell
py -m http.server 8080 --directory dashboard
# Open http://localhost:8080
```

Or view hosted: https://poojakira.github.io/mlsec-dashboards/model-privacy-attacks/

> **Note:** Dashboard is informational — not connected to live attack results.

---

## Troubleshooting

### `make test` Fails with "No module named attack_v19_core"

**Fix:** Install the sibling dependency:
```powershell
.\.venv\Scripts\python.exe -m pip install -e ..\attack-v19-core
```

---

### ImportError: No module named 'sklearn'

scikit-learn is required:
```powershell
.\.venv\Scripts\python.exe -m pip install scikit-learn
```

---

### Tests Pass Locally but Fail in CI

- CI runs on Linux — check for Windows-specific path issues
- Run `make lint` before pushing
- Ensure all dependencies are in `pyproject.toml`

---

### Slow Execution

Shadow model training can be slow. To speed up:
```powershell
# Reduce dataset size
.\.venv\Scripts\python.exe -m model_privacy_attacks.mia --dataset synthetic --n-samples 200
```

---

## Known Limitations

- Educational tool — use Opacus for production privacy defenses
- Attacks run on synthetic data only
- MIA accuracy is modest (~0.62) which is expected for simple shadow models
- Not production-ready; CI must pass on Linux before claiming results
- `make test` depends on `../attack-v19-core` being cloned alongside
