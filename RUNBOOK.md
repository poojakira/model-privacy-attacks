# Runbook

How to build, test, and run this project locally.

## Commands

| Task | Command |
|------|---------|
| Install everything | `make install` |
| Install attack-v19-core (local sibling) | `make install-core` |
| Run linter | `make lint` |
| Auto-format code | `make format` |
| Run tests | `make test` |
| Build package | `make build` |
| Run security scan (bandit + pip-audit) | `make security` |
| Run all checks | `make verify` |
| Serve dashboard locally | `make dashboard` (opens on port 8080) |

## Notes

- `make test` depends on `attack-v19-core` being installed. It expects the sibling directory `../attack-v19-core` by default (override with `ATTACK_CORE_PATH`).
- The dashboard at `dashboard/index.html` is a static 3D visualization. It's informational, not a test artifact.
- After pushing to main, check that GitHub Actions CI passes on Linux. Local Windows/Mac results don't guarantee CI will pass.
- This repo is not production-ready. Don't cite dashboard scores or local results as certifications.
