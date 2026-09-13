# AFOS v30.1 Verification & Compliance Manifest (operator reviewed 202609131158 v0.1 test eval schema)

## 1. Audit Attestation History
- **Prior Score:** 42/100 (Assessed under fragmented locks, instruction drift, and undefined arbitration math)
- **Target Remediated Score:** ≥ 95/100
- **Auditor Type:** Internal Engineering Rigor Benchmark (Synthetic - Non-Audited)
- **Regulatory Framework Alignment:** Basel III/IV IRB Framework Standards (Design-Level Emulation Only)

## 2. Systemic Fixes Implemented
| Subsystem | Root Cause | Remediated State |
| :--- | :--- | :--- |
| **Packaging** | Poetry/Maturin mismatch | Standardized on PEP 621 via `pyproject.toml` and Hatchling |
| **Dependencies** | Multi-lockfile divergence | Pinned `uv.lock`, removed unmanaged base lockfiles |
| **AI Governance** | Conflicting 0.50/0.85 thresholds | Defined tiered governance: Autonomy (≥0.85), HITL (0.50–0.84), Abort (<0.50) |
| **Credit Risk** | Divergence math ambiguity | Separated $\Delta_{\text{bi}}$ (arbitration flag) from $\Delta_{\text{downside}}$ (capital buffer) |
| **CI Security** | Unpinned actions & mutable envs | Pinned 40-character SHAs, switched to `uv sync --frozen` |

## 3. Reproduction Harness
Run the canonical verification sequence:
```bash
uv sync --frozen --all-extras
uv run python scripts/build_instructions.py --verify
uv run pytest -v --cov=afos --cov-fail-under=85
```
