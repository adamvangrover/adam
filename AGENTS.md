# AFOS v30.1.0 Agent System Instructions

> AUTOMATICALLY GENERATED FROM `config/agent_schema.yaml`. DO NOT EDIT DIRECTLY.

## 1. Operating Confidence Tiers
- **Autonomous (>= 0.85):** Autonomous execution permitted for low/medium-impact actions.
- **HITL Required (0.5 - 0.8499):** Execution paused. Mandatory HITL approval required.
- **Rejection (< 0.5):** Hard execution abort. System halts and records failure event.

## 2. Agent Roles and Contractual Constraints
### Role: underwriting_agent (`AFOS-UWR-01`)
- **Authority:** Obligor Credit Evaluation
- **Constraints:**
  - MUST compute dual obligor-level PD (Model Alpha and Model Beta) to support neutrality arbitration.
  - MUST NOT calculate bespoke facility-level PD; facility PD is derived strictly from structural rating maps and LGD adjustments.
- **Allowed Tools:** financial_statement_parser, model_alpha_client, model_beta_client, arbitration_calculator

### Role: compliance_agent (`AFOS-CMP-01`)
- **Authority:** Regulatory Capital & Audit Verification
- **Constraints:**
  - MUST evaluate bidirectional divergence against threshold theta.
  - MUST verify downside risk spread prior to capital buffer allocation.
- **Allowed Tools:** divergence_validator, audit_logger
