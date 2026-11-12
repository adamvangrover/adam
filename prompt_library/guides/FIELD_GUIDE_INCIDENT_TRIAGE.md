# Field Guide: Operational Incident Triage, Amnesia Recovery & Circuit Breakers

## 1. Scope & Objective
This Field Guide is the operational handbook for Incident Commanders, on-call SREs, and autonomous monitoring agents responding to production anomalies, amnesia events, deadlock states, or catastrophic circuit breaker trips in the AFOS ecosystem.

## 2. Emergency Escalation Matrix (P0 - P3)
- **P0 (Catastrophic Breach)**: Layer 0 safety bypass, ungrounded capital deployment, cryptographic signature failure, or data amnesia.
  - *Response Time:* Immediate (< 5 minutes). Action: Execute global kill-switch (`LOCKDOWN`), transition capital to 75% Fortress Ballast / 0% Convex Hunt.
- **P1 (System Degradation)**: p99 DAG latency > 45ms, consensus lockout (> 30% agents quarantined), or database migration deadlock.
  - *Response Time:* < 15 minutes. Action: Shed speculative worker jobs, activate fallback deterministic heuristics.
- **P2 (Localized Model Drift)**: Microstructure noise floor violation, TimesFM quantile exceedance ratio > 2.52 AE.
  - *Response Time:* < 1 hour. Action: Downgrade model tier to Challenger, notify model governance committee.
- **P3 (Telemetry / Non-Critical Log Drift)**: Formatting discrepancies, non-blocking warning logs.
  - *Response Time:* Next daily maintenance sprint.

## 3. Incident Triage Procedures

### Procedure 1: Triangulated Consensus Lockout
- **Symptoms**: Multi-agent consensus engine (`afos_v3/prov_o_consensus.py`) fails to converge; all trades or credit reports blocked.
- **Diagnosis**: Check `MICROSTRUCTURE_REL_NOISE_FLOOR` parameters. If high volatility asset (e.g. VIX or short-dated tech CDS) is subjected to index-level 10 bps threshold, false-positive outlier rejections occur.
- **Remediation**: Adjust noise floor to 100 bps for volatile instruments; verify SEC CIK cross-references.

### Procedure 2: Agent Amnesia & History Loss
- **Symptoms**: Agent fails to recall preceding multi-turn context or historical incident resolutions.
- **Diagnosis**: Check `.jules/history.jsonl` or `SwarmLogger` file open modes. Ensure no process opened logs with `"w"` instead of `"a"`.
- **Remediation**: Hydrate memory from W3C PROV-O JSONL ring-buffer backups; restart agent with state reconstruction enabled.

### Procedure 3: Port Collisions & Zombie Processes
- **Symptoms**: `EADDRINUSE` on internal ports (e.g., 3000, 8000, 8080).
- **Diagnosis & Remediation Commands**:
  ```bash
  # Identify PID occupying port
  lsof -i :8000
  # Terminate gracefully or force kill
  kill $(lsof -t -i :8000) 2>/dev/null || true
  ```

### Procedure 4: Zero-Trust Token Verification Failure
- **Symptoms**: Agent RPC calls rejected with HTTP 401/403 or `InvalidTokenSignature`.
- **Diagnosis**: Verify asymmetric ECDSA public key rotation in `adam-secrets.yaml` and ConfigMap `adam-public-keys`. Ensure verifier does not expect symmetric signing secret.
- **Remediation**: Resync public keys via STS endpoint `src/zero_trust/sts.py`.

## 4. Post-Incident Review Protocol
Every P0/P1 incident must log an entry in `EXPERIENCE.md` with:
1. Incident ID & Timestamp.
2. Root Cause Analysis (RCA) with epistemic tags.
3. Cryptographic W3C PROV-O audit trail reference.
4. Regression guardrail test added to `tests/`.
