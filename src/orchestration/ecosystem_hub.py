"""
AFOS v30.1 Unified Ecosystem Hub
=================================
Path A orchestration backbone connecting all v30.1-R orchestrators, governed
workspace skills, deterministic ASC 842 / SR 26-2 single-name credit & compute
lease underwriting, and automated P0 zero-trust security AST gates under W3C
PROV-O cryptographic provenance.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pdil.models import ProvenanceHeader  # noqa: E402


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_prov_envelope(
    activity_id: str,
    payload: Dict[str, Any],
    derivation_path: str,
    source_data_object: str,
    confidence_score: float = 0.94,
    git_commit_hash: str = "v30.1-ECOSYSTEM-HUB",
) -> tuple[Dict[str, Any], str]:
    header = ProvenanceHeader(
        git_commit_hash=git_commit_hash,
        timestamp=_utc_now(),
        content_hash=_canonical_sha256(payload),
        jsonLogic_version="2.0.0",
        confidence_score=max(0.0, min(1.0, float(confidence_score))),
        derivation_path=derivation_path,
        source_data_object=source_data_object,
    )
    return header.model_dump(), header.hash_state()


class EcosystemHub:
    """Unified execution hub across AFOS v30.1-R orchestrators, skills, underwriting, and security gates."""

    ORCHESTRATOR_CATALOG: List[Dict[str, Any]] = [
        {
            "id": "gate_clearance",
            "name": "ADAM v30.1-R Gate Clearance & LMT Contagion",
            "module": "scripts/run_v30_1_r_gate_clearance_orchestrator.py",
            "description": "7-Gate Release Matrix, LMT priming tree (Serta vs Unrestricted Sub vs Ch.11), PD calibration & 2LoD override floor.",
            "default_params": {
                "obligor_name": "BLTC Hypothetical Borrower",
                "ticker": "BLTC",
                "total_debt_m": 2500.0,
                "ebitda_m": 280.0,
                "baseline_dscr": 0.9955,
                "rate_shock_bps": 150.0,
            },
        },
        {
            "id": "market_mayhem",
            "name": "Market Mayhem v30.1-R — Frontier AI Capex & Credit Contagion",
            "module": "scripts/run_market_mayhem_v30_1_r.py",
            "description": "Macro ERP decomposition, neocloud GPU SPV contagion network, and Fortress/Hunt barbell playbook.",
            "default_params": {
                "ai_capex_b": 120.0,
                "rate_shift_bps": 100.0,
                "revenue_compression_pct": 30.0,
            },
        },
        {
            "id": "bitemporal_audit",
            "name": "IAEH & AFOS v30.1 Bitemporal Context & Policy Drift Audit",
            "module": "scripts/run_iaeh_afos_v30_1_bitemporal_audit.py",
            "description": "Zero-trust architecture linter inspecting cross-skill interfaces, bitemporal cache invalidation, and 7-gate tripwires.",
            "default_params": {
                "target_scope": "AFOS_v30_1_Kernel",
            },
        },
        {
            "id": "r2_expanded",
            "name": "AFOS v30.1-R2-Expanded Orchestrator (Hardened II Spec)",
            "module": "scripts/run_afos_v30_1_r2_expanded_orchestrator.py",
            "description": "Lexicographic Safety Primacy (L0 > L1 > L2), VPIN/SNER toxicity filtering, zero-trust order routing, and 180s kill-switch.",
            "default_params": {
                "sofr_base_rate_pct": 3.88,
                "hurdle_rate_bps": 150.0,
            },
        },
        {
            "id": "repo_prompt",
            "name": "Antigravity Repository-as-a-Prompt PDIL Kernel",
            "module": "scripts/run_antigravity_repo_prompt.py",
            "description": "Hydrates LLM_README, AGENTS.md, and AGENTS_KNOWLEDGE_BASE.md with System 1 / System 2 routing and PROV-O grounding.",
            "default_params": {
                "query": "Verify W3C PROV-O grounding and System 1/System 2 routing compliance.",
                "mode": "chat",
            },
        },
    ]

    SKILL_CATALOG: List[Dict[str, Any]] = [
        {
            "id": "hunt_and_kill",
            "name": "Hunt & Kill Mercenary Arbitrage",
            "module": "skills/hunt_and_kill.py",
            "description": "Evaluates 5 dislocation vectors under Lexicographic Control Barrier Functions (h1-h4) and VPIN/SNER toxicity filters.",
        },
        {
            "id": "librarian_archivist",
            "name": "Librarian & Archivist Bitemporal Catalog",
            "module": "skills/librarian_archivist.py",
            "description": "Tracks 5 institutional domains with bitemporal lineage (valid_at, transaction_at), SHA-256 fingerprinting, and lifecycle transitions.",
        },
        {
            "id": "llm_as_judge_arbiter",
            "name": "LLM-as-Judge & Arbiter",
            "module": "skills/llm_as_judge_arbiter.py",
            "description": "Multi-criteria rubric scoring, pairwise adversarial debate adjudication, neuro-symbolic tripwires, and Cohen's Kappa / Brier calibration.",
        },
        {
            "id": "pd_model_challenger",
            "name": "PD Model Challenger (EL = PD x LGD x EAD)",
            "module": "skills/pd_model_challenger.py",
            "description": "Validates quantitative expected loss against incumbent figures and flags variance threshold breaches.",
        },
        {
            "id": "epistemic_grounding",
            "name": "Epistemic State & Grounding Validator",
            "module": "skills/epistemic_grounding.py",
            "description": "Verifies assertions against repository artifacts and deterministically gates promotion_allowed.",
        },
        {
            "id": "validate_jsonlogic_policy",
            "name": "JSONLogic Network & Policy Validator",
            "module": "skills/validate_jsonlogic_policy.py",
            "description": "Evaluates proposed action targets against deterministic security policies (SSRF / private IP denial).",
        },
        {
            "id": "extract_covenants",
            "name": "SEC Filing Debt Covenant Extractor",
            "module": "skills/extract_covenants.py",
            "description": "Extracts debt covenants and estimates technical default breach probability from filing text.",
        },
    ]

    PORTAL_CATALOG: List[Dict[str, str]] = [
        {
            "id": "command_center",
            "name": "AFOS v30.1 Ecosystem Command Center",
            "path": "showcase/command_center_v30_1.html",
            "category": "Primary Control Plane",
            "description": "Interactive orchestrator mission control, AI/TMT compute-lease underwriting studio, and skill runner.",
        },
        {
            "id": "antigravity_ide",
            "name": "Antigravity Agent IDE & Workbench",
            "path": "src/antigravity_ide/static/antigravity_ide.html",
            "category": "Developer Workbench",
            "description": "Code editor, file explorer, live agent chat, tool/trigger inspector, and orchestrator dispatcher.",
        },
        {
            "id": "antigravity_app",
            "name": "Antigravity Desktop & Web Runtime",
            "path": "showcase/antigravity_app.html",
            "category": "Runtime Console",
            "description": "System 1 Swarm vs System 2 DAG prompt arena with live W3C PROV-O envelope inspection.",
        },
        {
            "id": "os_root",
            "name": "ADAM v30.1 Unified OS Root",
            "path": "index.html",
            "category": "System Portal",
            "description": "Master launchpad indexing workspaces, prototypes, financial models, and omni-search.",
        },
        {
            "id": "analyst_os",
            "name": "ADAM Analyst OS",
            "path": "analyst_os.html",
            "category": "Financial Workbench",
            "description": "Client-side financial workstation for human-agent credit underwriting.",
        },
        {
            "id": "daily_hub",
            "name": "ADAM Daily Hub Index",
            "path": "showcase/adam_daily_hub_index.html",
            "category": "Intelligence Archive",
            "description": "Chronological ledger of daily market dispatches, scenario engines, and JSONL telemetry.",
        },
        {
            "id": "workflow_orchestrator",
            "name": "Workflow Orchestrator HUD",
            "path": "workflow_orchestrator.html",
            "category": "Pipeline Designer",
            "description": "Multi-step credit & governance harness HUD.",
        },
        {
            "id": "market_mayhem_terminal",
            "name": "Market Mayhem Institutional Terminal",
            "path": "market_mayhem_terminal.html",
            "category": "Macro & Credit",
            "description": "Macro vulnerability telemetry, distress cascades, and tactical defense playbooks.",
        },
    ]

    # Vetted internal modules permitted to use importlib.import_module for plugin/agent loading
    ALLOWED_IMPORTLIB_FILES = {
        "core/vertical_risk_agent/tools/mcp_server/server.py",
        "core/agents/industry_specialist_agent.py",
        "core/agents/meta_agents/skill_harvester_agent.py",
        "core/system/agent_orchestrator.py",
        "core/system/plugin_manager.py",
        "core/engine/swarm/mirofish_engine.py",
        "src/ingestion/plugin_manager.py",
    }

    def list_orchestrators(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.ORCHESTRATOR_CATALOG]

    def list_skills(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self.SKILL_CATALOG]

    def list_portals(self) -> List[Dict[str, str]]:
        return [dict(item) for item in self.PORTAL_CATALOG]

    def run_orchestrator(
        self,
        orchestrator_id: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes a registered AFOS v30.1-R orchestrator (or 'all') and wraps output with W3C PROV-O metadata."""
        cfg = dict(params or {})
        oid = orchestrator_id.strip().lower()

        if oid == "all":
            suite_ids = ["gate_clearance", "market_mayhem", "bitemporal_audit", "r2_expanded", "repo_prompt"]
            results: Dict[str, Any] = {}
            for sub_id in suite_ids:
                results[sub_id] = self.run_orchestrator(sub_id, cfg.get(sub_id) if isinstance(cfg.get(sub_id), dict) else None)
            all_ok = all(r.get("status") not in ("ERROR", "FAILED", "ABORTED") for r in results.values())
            sweep_payload = {
                "orchestrators_executed": suite_ids,
                "total_executed": len(suite_ids),
                "all_passed": all_ok,
                "orchestrators": results,
            }
            prov_header, fingerprint = _build_prov_envelope(
                activity_id="ecosystem_release_sweep",
                payload=sweep_payload,
                derivation_path="EcosystemHub -> FullReleaseSweep(5_Orchestrators) -> W3C_PROV_O",
                source_data_object="urn:adam:ecosystem:release-sweep:v30.1",
                confidence_score=0.95 if all_ok else 0.65,
            )
            return {
                "orchestrator_id": "all",
                "orchestrator_version": "AFOS v30.1 Unified Release Sweep",
                "status": "ALL_ORCHESTRATORS_PASSED" if all_ok else "REVIEW_REQUIRED",
                "timestamp": _utc_now(),
                "provenance_header": prov_header,
                "sha256_release_fingerprint": fingerprint,
                "result": sweep_payload,
            }

        if oid == "gate_clearance":
            from scripts.run_v30_1_r_gate_clearance_orchestrator import GateClearanceOrchestrator

            orch = GateClearanceOrchestrator(
                obligor_name=str(cfg.get("obligor_name") or cfg.get("obligor") or "BLTC Hypothetical Borrower"),
                ticker=str(cfg.get("ticker") or "BLTC"),
                total_debt_m=float(cfg.get("total_debt_m", cfg.get("total_debt", 2500.0))),
                ebitda_m=float(cfg.get("ebitda_m", cfg.get("ebitda", 280.0))),
                baseline_dscr=float(cfg.get("baseline_dscr", cfg.get("dscr", 0.9955))),
                rate_shock_bps=float(cfg.get("rate_shock_bps", 150.0)),
            )
            raw = orch.run()
            return {
                "orchestrator_id": "gate_clearance",
                "orchestrator_version": raw.get("orchestrator_version", "v30.1-R"),
                "status": raw.get("release_disposition", "COMPLETED"),
                "timestamp": raw.get("provenance_header", {}).get("timestamp", _utc_now()),
                "provenance_header": raw.get("provenance_header", {}),
                "sha256_release_fingerprint": raw.get("sha256_release_fingerprint", ""),
                "result": raw,
            }

        if oid == "market_mayhem":
            from scripts.run_market_mayhem_v30_1_r import MarketMayhemV301ROrchestrator

            orch = MarketMayhemV301ROrchestrator(
                ai_capex_b=float(cfg.get("ai_capex_b", 120.0)),
                rate_shift_bps=float(cfg.get("rate_shift_bps", 100.0)),
                revenue_compression_pct=float(cfg.get("revenue_compression_pct", 30.0)),
            )
            raw = orch.run()
            return {
                "orchestrator_id": "market_mayhem",
                "orchestrator_version": raw.get("orchestrator_version", "Market Mayhem v30.1-R"),
                "status": "COMPLETED",
                "timestamp": raw.get("timestamp", _utc_now()),
                "provenance_header": raw.get("provenance_header", {}),
                "sha256_release_fingerprint": raw.get("sha256_release_fingerprint", ""),
                "result": raw,
            }

        if oid == "bitemporal_audit":
            from scripts.run_iaeh_afos_v30_1_bitemporal_audit import AFOSV301BitemporalAuditOrchestrator

            orch = AFOSV301BitemporalAuditOrchestrator(
                target_scope=str(cfg.get("target_scope") or cfg.get("scope") or "AFOS_v30_1_Kernel"),
            )
            raw = orch.run()
            return {
                "orchestrator_id": "bitemporal_audit",
                "orchestrator_version": raw.get("orchestrator_version", "IAEH & AFOS v30.1 Audit"),
                "status": raw.get("audit_summary", {}).get("audit_disposition", "COMPLETED"),
                "timestamp": raw.get("timestamp", _utc_now()),
                "provenance_header": raw.get("provenance_header", {}),
                "sha256_release_fingerprint": raw.get("sha256_release_fingerprint", ""),
                "result": raw,
            }

        if oid == "r2_expanded":
            from scripts.run_afos_v30_1_r2_expanded_orchestrator import AFOSV301R2ExpandedOrchestrator

            orch = AFOSV301R2ExpandedOrchestrator(correlation_id=cfg.get("correlation_id"))
            payload = orch.build_sample_payload()
            if "sofr_base_rate_pct" in cfg and cfg["sofr_base_rate_pct"] is not None:
                payload["sofr_base_rate_pct"] = float(cfg["sofr_base_rate_pct"])
            if "hurdle_rate_bps" in cfg and cfg["hurdle_rate_bps"] is not None:
                payload["hurdle_rate_bps"] = float(cfg["hurdle_rate_bps"])
            if "vpin_override" in cfg and cfg["vpin_override"] is not None and payload.get("targets"):
                payload["targets"][0]["vpin"] = float(cfg["vpin_override"])
            if "sner_override" in cfg and cfg["sner_override"] is not None and payload.get("targets"):
                payload["targets"][0]["sner"] = float(cfg["sner_override"])
            if "short_utilization_override" in cfg and cfg["short_utilization_override"] is not None:
                util = float(cfg["short_utilization_override"])
                if payload.get("counter_predation_inputs"):
                    cp = payload["counter_predation_inputs"][0]
                    cp["short_squeeze_utilization_pct"] = util
                    if util > 85.0:
                        cp["borrow_recall_ceiling_pass"] = False
                        cp["overall_pass"] = False

            raw = orch.execute(payload)
            ab = raw.get("audit_bundle", {})
            return {
                "orchestrator_id": "r2_expanded",
                "orchestrator_version": raw.get("orchestrator_version", "AFOS v30.1-R2-Expanded"),
                "status": raw.get("status", "EXPLOITS_AND_DEFENSES_EVALUATED"),
                "timestamp": raw.get("timestamp", _utc_now()),
                "provenance_header": ab.get("provenance_header", {}),
                "sha256_release_fingerprint": ab.get("sha256_release_fingerprint", ""),
                "result": raw,
            }

        if oid == "repo_prompt":
            from scripts.run_antigravity_repo_prompt import _run_offline_deterministic_fallback

            query = str(cfg.get("query") or "Verify W3C PROV-O grounding and System 1/System 2 routing compliance.")
            mode = str(cfg.get("mode") or "chat")
            raw = _run_offline_deterministic_fallback(
                query=query,
                reason="Executed via EcosystemHub deterministic PDIL kernel.",
                mode=mode,
            )
            agent_out = raw.get("agent_output", {})
            return {
                "orchestrator_id": "repo_prompt",
                "orchestrator_version": "AFOS v30.1 Repository-as-a-Prompt",
                "status": "GROUNDED" if raw.get("w3c_prov_o_grounded") else "UNGROUNDED",
                "timestamp": agent_out.get("provenance_trace", {}).get("timestamp", _utc_now()),
                "provenance_header": agent_out.get("provenance_trace", {}),
                "sha256_release_fingerprint": raw.get("state_sha256", ""),
                "result": raw,
            }

        raise ValueError(f"Unknown orchestrator_id: {orchestrator_id}")

    def default_skill_payload(self, skill_id: str) -> Dict[str, Any]:
        """Provides a representative default input payload for any workspace skill."""
        sid = skill_id.strip().lower()
        if sid == "hunt_and_kill":
            from scripts.run_afos_v30_1_r2_expanded_orchestrator import AFOSV301R2ExpandedOrchestrator

            return AFOSV301R2ExpandedOrchestrator().build_sample_payload()
        if sid == "librarian_archivist":
            return {
                "correlation_id": "corr-hub-archivist-001",
                "operation": "catalog_register",
                "artifacts": [
                    {
                        "artifact_id": "ART-CREDIT-001",
                        "domain": "credit_facilities",
                        "entity_type": "CreditAgreement",
                        "current_lifecycle_tier": "hot_memory",
                        "content_payload": {"facility": "1L Term Loan B", "fccr_floor": 1.15},
                    },
                    {
                        "artifact_id": "ART-MODEL-002",
                        "domain": "quantitative_models",
                        "entity_type": "PDCalibration",
                        "current_lifecycle_tier": "warm_searchable",
                        "content_payload": {"model": "CRE-31", "brier": 0.042},
                    },
                ],
                "dependency_edges": [
                    {
                        "source_id": "ART-MODEL-002",
                        "target_id": "ART-CREDIT-001",
                        "source_domain": "quantitative_models",
                        "target_domain": "credit_facilities",
                        "relation_type": "CALIBRATES_CRITICAL",
                    }
                ],
            }
        if sid == "llm_as_judge_arbiter":
            return {
                "correlation_id": "corr-hub-arbiter-001",
                "evaluation_type": "single_candidate_rubric",
                "tripwire_policy": {
                    "min_dscr": 1.05,
                    "allow_unauthorized_addbacks": False,
                    "max_leverage": 7.5,
                    "min_liquidity_runway_days": 90,
                    "require_sr11_7": True,
                },
                "candidate_proposal": {
                    "id": "PROP-NEOC-2026",
                    "proposal_type": "credit_memo",
                    "financial_metrics": {
                        "dscr": 1.42,
                        "leverage_ratio": 4.8,
                        "unauthorized_addbacks": False,
                        "refinancing_wall_modeled": True,
                        "margin_compression_modeled": True,
                        "non_linear_debt_service_decay": True,
                        "sr11_7_compliant": True,
                        "snc_classification": "PASS",
                        "liquidity_runway_days": 240,
                    },
                    "evidence_mappings": [
                        {"assertion_id": "A1", "text": "10-K debt footnote verified", "source_ref": "doc:sec-10k-2026"},
                        {"assertion_id": "A2", "text": "Compute lease schedule verified", "source_ref": "filing:asc842-note"},
                    ],
                },
            }
        if sid == "pd_model_challenger":
            return {
                "correlation_id": "corr-hub-pd-001",
                "obligor_id": "OBLIGOR-NEOC-01",
                "pd_fraction": 0.032,
                "lgd_fraction": 0.45,
                "ead_amount": 50000000.0,
                "fccr_ratio": 1.28,
                "incumbent_el_amount": 700000.0,
                "tolerance_threshold": 50000.0,
            }
        if sid == "epistemic_grounding":
            return {
                "correlation_id": "corr-hub-grounding-001",
                "commit_sha": "LOCAL",
                "assertions": [
                    {
                        "claim_id": "CLAIM-001",
                        "text": "AFOS v30.1 architectural invariants and persona boundaries are defined in AGENTS.md.",
                        "source_refs": ["AGENTS.md", "docs/AGENTS_KNOWLEDGE_BASE.md"],
                    }
                ],
            }
        if sid == "validate_jsonlogic_policy":
            return {
                "action": {
                    "id": "ACT-HUB-001",
                    "target": "https://api.sec.gov/edgar/facts",
                }
            }
        if sid == "extract_covenants":
            return {
                "filing_text": (
                    "Credit Agreement Section 7.02 Debt Covenant requires Consolidated Leverage Ratio <= 4.50x "
                    "and prohibits IP drop-down transfers to unrestricted subsidiaries."
                )
            }
        raise ValueError(f"Unknown skill_id: {skill_id}")

    def run_skill(
        self,
        skill_id: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes a governed workspace skill with optional payload override and W3C PROV-O envelope."""
        sid = skill_id.strip().lower()
        effective_payload = payload if payload else self.default_skill_payload(sid)

        if sid == "hunt_and_kill":
            from skills.hunt_and_kill import run as skill_fn

            raw = skill_fn(effective_payload)
        elif sid == "librarian_archivist":
            from skills.librarian_archivist import run as skill_fn

            raw = skill_fn(effective_payload)
        elif sid == "llm_as_judge_arbiter":
            from skills.llm_as_judge_arbiter import run as skill_fn

            raw = skill_fn(effective_payload)
        elif sid == "pd_model_challenger":
            from skills.pd_model_challenger import run as skill_fn

            raw = skill_fn(effective_payload)
        elif sid == "epistemic_grounding":
            from skills.epistemic_grounding import run as skill_fn

            raw = skill_fn(effective_payload)
        elif sid == "validate_jsonlogic_policy":
            from skills.validate_jsonlogic_policy import validate_jsonlogic_policy

            raw = validate_jsonlogic_policy(effective_payload)
        elif sid == "extract_covenants":
            from skills.extract_covenants import extract_debt_covenants

            text = (
                effective_payload.get("filing_text", "")
                if isinstance(effective_payload, dict)
                else str(effective_payload)
            )
            raw_str = extract_debt_covenants(text)
            raw = json.loads(raw_str)
        else:
            raise ValueError(f"Unknown skill_id: {skill_id}")

        prov_header, fingerprint = _build_prov_envelope(
            activity_id=f"skill_{sid}",
            payload=raw if isinstance(raw, dict) else {"output": raw},
            derivation_path=f"EcosystemHub -> skills.{sid} -> W3C_PROV_O",
            source_data_object=f"urn:adam:skill:{sid}",
            confidence_score=0.93,
        )
        return {
            "skill_id": sid,
            "status": raw.get("status", raw.get("policy_decision", "COMPLETED")) if isinstance(raw, dict) else "COMPLETED",
            "timestamp": _utc_now(),
            "provenance_header": prov_header,
            "sha256_release_fingerprint": fingerprint,
            "result": raw,
        }

    def underwrite_obligor(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Deterministic Single-Name Credit & Off-Balance-Sheet Compute Lease Underwriting Engine
        compliant with AFOS v30.1 Section 6.1 (ASC 842 / SR 26-2) & Dual-PD Neutrality Arbitration.
        """
        p = dict(params or {})
        ticker = str(p.get("ticker", "NEOC")).upper()
        company_name = str(p.get("company_name", "Neocloud Compute Infrastructure Corp"))
        sector = str(p.get("sector", "TMT / AI Compute Infrastructure"))

        revenue_m = float(p.get("revenue_m", 1800.0))
        ebitda_m = float(p.get("ebitda_m", 420.0))
        gaap_funded_debt_m = float(p.get("gaap_funded_debt_m", p.get("funded_debt_m", p.get("total_debt_m", 950.0))))
        lease_commitments_m = float(
            p.get("off_balance_sheet_lease_commitments_m", p.get("lease_commitments_m", 3200.0))
        )
        lease_discount_rate_pct = float(p.get("lease_discount_rate_pct", 6.0))
        lease_term_years = max(1.0, float(p.get("lease_term_years", 5.0)))
        annual_lease_expense_m = float(
            p.get("annual_lease_expense_m", p.get("lease_expense_m", lease_commitments_m / lease_term_years))
        )
        cash_interest_m = float(p.get("cash_interest_m", p.get("interest_expense_m", 65.0)))
        capex_maint_m = float(p.get("capex_maint_m", 90.0))
        cash_taxes_m = float(p.get("cash_taxes_m", 45.0))
        delta_wc_leases_m = float(p.get("delta_wc_leases_m", 35.0))
        undrawn_rcf_m = float(p.get("undrawn_rcf_m", 400.0))
        rcf_commitment_fee_bps = float(p.get("rcf_commitment_fee_bps", 37.5))
        spot_sofr_pct = float(p.get("spot_sofr_pct", 4.35))
        rate_shock_bps = float(p.get("rate_shock_bps", 150.0))
        floating_debt_share = max(0.0, min(1.0, float(p.get("floating_debt_share", 0.75))))
        theta_threshold = float(p.get("theta_threshold", 0.03))
        liquidity_runway_days = int(p.get("liquidity_runway_days", 240))

        # 1. Lease capitalization via present value annuity at implicit lease discount rate (default 6.0%)
        r = max(0.0, lease_discount_rate_pct / 100.0)
        if lease_commitments_m <= 0.0:
            capitalized_lease_liability_m = 0.0
        elif r == 0.0:
            capitalized_lease_liability_m = lease_commitments_m
        else:
            annual_payment = lease_commitments_m / lease_term_years
            annuity_factor = (1.0 - math.pow(1.0 + r, -lease_term_years)) / r
            capitalized_lease_liability_m = annual_payment * annuity_factor

        total_adjusted_debt_m = gaap_funded_debt_m + capitalized_lease_liability_m
        ebitdar_m = ebitda_m + annual_lease_expense_m

        gaap_leverage_x = gaap_funded_debt_m / max(ebitda_m, 1e-6)
        lease_capitalized_leverage_x = total_adjusted_debt_m / max(ebitdar_m, 1e-6)
        hidden_leverage_delta_x = lease_capitalized_leverage_x - gaap_leverage_x

        # 2. Full Cash-Flow Waterfall CADS & Fixed-Charge Coverage Ratio (Spot vs Stressed)
        cads_full_waterfall_m = ebitdar_m - capex_maint_m - cash_taxes_m - delta_wc_leases_m
        undrawn_rcf_fee_m = undrawn_rcf_m * (rcf_commitment_fee_bps / 10000.0)
        spot_fixed_charges_m = cash_interest_m + annual_lease_expense_m + undrawn_rcf_fee_m
        spot_fccr = cads_full_waterfall_m / max(spot_fixed_charges_m, 1e-6)

        stressed_interest_increment_m = gaap_funded_debt_m * floating_debt_share * (rate_shock_bps / 10000.0)
        stressed_fixed_charges_m = spot_fixed_charges_m + stressed_interest_increment_m
        stressed_fccr = cads_full_waterfall_m / max(stressed_fixed_charges_m, 1e-6)
        fccr_tripwire_breached = (spot_fccr < 1.15) or (stressed_fccr < 1.15)

        # 3. Dual Obligor PD (Model Alpha vs Model Beta) & Neutrality Arbitration
        def _sigmoid(x: float) -> float:
            x_clamped = max(-20.0, min(20.0, x))
            return 1.0 / (1.0 + math.exp(-x_clamped))

        raw_alpha = _sigmoid((gaap_leverage_x - 3.5) * 0.45 + (1.5 - spot_fccr) * 0.65 - 2.4)
        raw_beta = _sigmoid((lease_capitalized_leverage_x - 3.5) * 0.42 + (1.5 - stressed_fccr) * 0.78 - 1.9)
        alpha_override = p.get("model_alpha_pd", p.get("pd_alpha"))
        beta_override = p.get("model_beta_pd", p.get("pd_beta"))
        model_alpha_pd = round(
            max(0.0025, min(0.99, float(alpha_override) if alpha_override is not None else raw_alpha)),
            4,
        )
        model_beta_pd = round(
            max(0.0040, min(0.99, float(beta_override) if beta_override is not None else raw_beta)),
            4,
        )

        bidirectional_disparity = round(abs(model_beta_pd - model_alpha_pd), 4)
        arbitration_flag_triggered = bidirectional_disparity > theta_threshold
        one_sided_downside_spread = round(max(0.0, model_beta_pd - model_alpha_pd), 4)
        capital_buffer_penalty_bps = round(one_sided_downside_spread * 10000.0 * 0.25, 2)

        # 4. Implied SNC Rating & Operating Confidence Tier
        if stressed_fccr < 0.85 or model_beta_pd >= 0.20 or cads_full_waterfall_m <= 0:
            implied_rating = "DOUBTFUL"
        elif stressed_fccr < 1.15 or lease_capitalized_leverage_x > 7.5 or model_beta_pd >= 0.10:
            implied_rating = "SUBSTANDARD"
        elif stressed_fccr < 1.35 or lease_capitalized_leverage_x > 5.5 or arbitration_flag_triggered:
            implied_rating = "SPECIAL_MENTION"
        else:
            implied_rating = "PASS"

        confidence_score = 0.94
        if arbitration_flag_triggered:
            confidence_score -= 0.14
        if fccr_tripwire_breached:
            confidence_score -= 0.12
        if cads_full_waterfall_m <= 0:
            confidence_score -= 0.35
        if liquidity_runway_days < 90:
            confidence_score -= 0.15
        confidence_score = round(max(0.10, min(0.99, confidence_score)), 4)

        if confidence_score >= 0.85:
            operating_tier = "AUTONOMOUS"
        elif confidence_score >= 0.50:
            operating_tier = "HITL_REQUIRED"
        else:
            operating_tier = "HARD_ABORT"

        lockdown_override = bool(p.get("lockdown", p.get("force_lockdown", False)))
        lockdown = lockdown_override or (operating_tier == "HARD_ABORT") or (implied_rating == "DOUBTFUL")
        system_status = "LOCKDOWN" if lockdown else "NOMINAL"
        convex_hunt_pct = 0.0 if lockdown else 25.0

        result_body = {
            "obligor": {
                "ticker": ticker,
                "company_name": company_name,
                "sector": sector,
                "spot_sofr_pct": round(spot_sofr_pct, 2),
                "rate_shock_bps": round(rate_shock_bps, 1),
            },
            "gaap_vs_lease_adjusted": {
                "revenue_m": round(revenue_m, 2),
                "ebitda_m": round(ebitda_m, 2),
                "annual_lease_expense_m": round(annual_lease_expense_m, 2),
                "ebitdar_m": round(ebitdar_m, 2),
                "gaap_funded_debt_m": round(gaap_funded_debt_m, 2),
                "off_balance_sheet_lease_commitments_m": round(lease_commitments_m, 2),
                "lease_discount_rate_pct": round(lease_discount_rate_pct, 2),
                "lease_term_years": round(lease_term_years, 2),
                "capitalized_lease_liability_m": round(capitalized_lease_liability_m, 2),
                "total_adjusted_debt_m": round(total_adjusted_debt_m, 2),
                "gaap_leverage_x": round(gaap_leverage_x, 2),
                "lease_capitalized_leverage_x": round(lease_capitalized_leverage_x, 2),
                "hidden_leverage_delta_x": round(hidden_leverage_delta_x, 2),
            },
            "cash_flow_waterfall": {
                "ebitdar_m": round(ebitdar_m, 2),
                "capex_maint_m": round(capex_maint_m, 2),
                "cash_taxes_m": round(cash_taxes_m, 2),
                "delta_wc_leases_m": round(delta_wc_leases_m, 2),
                "cads_full_waterfall_m": round(cads_full_waterfall_m, 2),
                "cash_interest_m": round(cash_interest_m, 2),
                "undrawn_rcf_m": round(undrawn_rcf_m, 2),
                "undrawn_rcf_fee_m": round(undrawn_rcf_fee_m, 2),
                "spot_fixed_charges_m": round(spot_fixed_charges_m, 2),
                "stressed_interest_increment_m": round(stressed_interest_increment_m, 2),
                "stressed_fixed_charges_m": round(stressed_fixed_charges_m, 2),
                "spot_fccr": round(spot_fccr, 3),
                "stressed_fccr": round(stressed_fccr, 3),
                "fccr_tripwire_threshold": 1.15,
                "fccr_tripwire_breached": fccr_tripwire_breached,
            },
            "dual_pd_arbitration": {
                "model_alpha_pd": model_alpha_pd,
                "model_beta_pd": model_beta_pd,
                "theta_threshold": theta_threshold,
                "bidirectional_disparity": bidirectional_disparity,
                "arbitration_flag_triggered": arbitration_flag_triggered,
                "one_sided_downside_spread": one_sided_downside_spread,
                "capital_buffer_penalty_bps": capital_buffer_penalty_bps,
                "implied_rating": implied_rating,
            },
            "governance": {
                "confidence_score": confidence_score,
                "operating_tier": operating_tier,
                "hitl_required": operating_tier != "AUTONOMOUS",
                "system_status": system_status,
                "fortress_ballast_pct": 75.0,
                "convex_hunt_pct": convex_hunt_pct,
                "liquidity_runway_days": liquidity_runway_days,
            },
        }

        prov_header, fingerprint = _build_prov_envelope(
            activity_id="underwrite_obligor",
            payload=result_body,
            derivation_path="EcosystemHub -> AFOS-UWR-01(ASC842) -> AFOS-CMP-01(DualPD) -> W3C_PROV_O",
            source_data_object=f"urn:adam:underwriting:{ticker}",
            confidence_score=confidence_score,
        )
        result_body["provenance_header"] = prov_header
        result_body["sha256_release_fingerprint"] = fingerprint
        return result_body

    def run_security_ci_gate(self, roots: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        P0 Zero-Trust AST Security CI Gate:
        Scans core/ and src/ for forbidden pickle.load / pickle.loads calls and
        unwhitelisted importlib.import_module invocations.
        """
        target_roots = roots or ["core", "src"]
        files_scanned = 0
        pickle_violations: List[Dict[str, Any]] = []
        importlib_violations: List[Dict[str, Any]] = []
        importlib_whitelisted: List[Dict[str, Any]] = []

        for root_name in target_roots:
            candidate_root = Path(root_name)
            root_path = candidate_root if candidate_root.is_absolute() else (REPO_ROOT / candidate_root)
            if not root_path.exists():
                continue
            for dirpath, _, filenames in os.walk(root_path):
                if "__pycache__" in dirpath:
                    continue
                for fname in filenames:
                    if not fname.endswith(".py"):
                        continue
                    fpath = Path(dirpath) / fname
                    try:
                        rel_path = str(fpath.relative_to(REPO_ROOT)).replace("\\", "/")
                    except ValueError:
                        rel_path = str(fpath).replace("\\", "/")
                    files_scanned += 1
                    try:
                        source = fpath.read_text(encoding="utf-8", errors="ignore")
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", SyntaxWarning)
                            tree = ast.parse(source, filename=rel_path)
                    except SyntaxError:
                        continue

                    for node in ast.walk(tree):
                        if not isinstance(node, ast.Call):
                            continue
                        func = node.func
                        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                            mod_name = func.value.id
                            attr_name = func.attr
                            if mod_name in ("pickle", "cPickle", "_pickle") and attr_name in ("load", "loads"):
                                pickle_violations.append({
                                    "file": rel_path,
                                    "line": getattr(node, "lineno", 0),
                                    "call": f"{mod_name}.{attr_name}",
                                })
                            elif mod_name == "importlib" and attr_name == "import_module":
                                entry = {
                                    "file": rel_path,
                                    "line": getattr(node, "lineno", 0),
                                    "call": "importlib.import_module",
                                }
                                if rel_path in self.ALLOWED_IMPORTLIB_FILES:
                                    importlib_whitelisted.append(entry)
                                else:
                                    importlib_violations.append(entry)

        passed = (len(pickle_violations) == 0) and (len(importlib_violations) == 0)
        gate_payload = {
            "gate_id": "P0_AST_PICKLE_AND_IMPORTLIB_GATE",
            "roots_scanned": target_roots,
            "files_scanned": files_scanned,
            "passed": passed,
            "pickle_violations": pickle_violations,
            "unwhitelisted_importlib_violations": importlib_violations,
            "whitelisted_importlib_count": len(importlib_whitelisted),
        }
        prov_header, fingerprint = _build_prov_envelope(
            activity_id="security_ci_gate",
            payload=gate_payload,
            derivation_path="EcosystemHub -> AFOS-SNT-01(AST_Gate) -> W3C_PROV_O",
            source_data_object="urn:adam:security:p0-ast-gate",
            confidence_score=1.0 if passed else 0.40,
        )
        gate_payload["provenance_header"] = prov_header
        gate_payload["sha256_release_fingerprint"] = fingerprint
        return gate_payload

    def get_ecosystem_manifest(self) -> Dict[str, Any]:
        """Returns unified metadata across orchestrators, skills, UI portals, CLI commands, and MCP tools."""
        orchestrators = self.list_orchestrators()
        skills = self.list_skills()
        portals = self.list_portals()
        cli_commands = [
            "run",
            "chat",
            "tools",
            "triggers",
            "status",
            "serve",
            "extensions",
            "orchestrate",
            "underwrite",
            "skills",
            "audit-security",
            "portals",
        ]
        mcp_tools = [
            "run_v30_1_orchestrator",
            "underwrite_ai_credit_obligor",
            "run_workspace_skill",
            "list_ecosystem_capabilities",
        ]
        manifest = {
            "version": "30.1",
            "protocol": "ARCHITECT_INFINITE",
            "timestamp": _utc_now(),
            "counts": {
                "orchestrators": len(orchestrators),
                "skills": len(skills),
                "portals": len(portals),
                "cli_commands": len(cli_commands),
                "mcp_tools": len(mcp_tools),
            },
            "orchestrators": orchestrators,
            "skills": skills,
            "portals": portals,
            "cli_commands": cli_commands,
            "mcp_tools_integrated": mcp_tools,
        }
        prov_header, fingerprint = _build_prov_envelope(
            activity_id="ecosystem_manifest",
            payload=manifest,
            derivation_path="EcosystemHub -> Manifest -> W3C_PROV_O",
            source_data_object="urn:adam:ecosystem:manifest:v30.1",
            confidence_score=1.0,
        )
        manifest["provenance_header"] = prov_header
        manifest["sha256_release_fingerprint"] = fingerprint
        return manifest


__all__ = ["EcosystemHub"]
