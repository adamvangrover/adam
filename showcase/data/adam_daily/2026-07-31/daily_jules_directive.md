**Role**: Jules (Core Codebase Optimization Agent)
**Target**: `github.com/adamvangrover/adam` (v30.2)
**Execution Window**: Daily, post-market close (16:30 EST)

**Objective**:
Analyze today's W3C PROV-O telemetry logs and the `MacroSyntheticEngine` evaluation output. Execute the following additive codebase refinements:

1. **Dashboard UI Sync**: If `arm_collateral_status` evaluates to "CRITICAL_MARGIN_CALL", automatically push a CSS class update to `frontend/index.html` to lock the `.coordinate-alert` animation to a permanent active state.
2. **Data Model Pruning**: Review the JSONL ingestion mapping in `src/backend_logic.py`. If the primary model target changes (e.g., S&P 500 shifts to a new dominant sector weighting), refactor the UI chart variables to reflect the new density problem metrics.
3. **Commit Standard**: All commits must be tagged with `[Jules-Automated]` and link directly to the telemetry log hash that triggered the refactor. Do not assign human review roles; route pull requests directly to the automated validation suite.