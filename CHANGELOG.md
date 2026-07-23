# Changelog - model-privacy-attacks

## [1.0.0] - 2026-07-22

### Changed - ATT&CK v19 Migration

#### Technique Remappings (Revoked -> New)
| Old ID | New ID | Rule Table Keys Affected |
|--------|--------|-------------------------|
| T1562 | T1685 | differential_privacy_bypass |

#### Rule Table Updates
```python
# BEFORE
"differential_privacy_bypass": ["T1562", "T1565"],

# AFTER
"differential_privacy_bypass": ["T1685", "T1565"],
```

### Added
- T1685 (Disable or Modify Tools) replacing revoked T1562 for privacy bypass detection

### Migration
See [attack-v19-core MIGRATION_GUIDE.md](../attack-v19-core/MIGRATION_GUIDE.md) for full migration steps.