# Probabilistic-to-Deterministic Integration Layer (PDIL) Mapping

This document maps out the dependencies across the Adam repository for various modules, emphasizing the transition from probabilistic models to deterministic execution via the PDIL layer.

## 1. Machine Learning & Model Development
*   **Core Probabilistic Engine (System 1):** Focuses on semantic understanding, inference, and generating W3C PROV-O compliant provenance.
    *   Dependencies: `src/llm/schemas.py`, `core/analysis/gemini_analyzer.py`
*   **Data Sources & Fetching:**
    *   Dependencies: `core/data_sources/data_fetcher.py`, `core/data_sources/financial_news_api.py`

## 2. Orchestration & Agent Skills
*   **Meta-Orchestrator:** Handles the strategic triad and overarching workflows.
    *   Dependencies: `core/agents/meta_agents/omega_meta_orchestrator.py`, `core/engine/cognitive_harness.py`
*   **Agent Implementations (System 2):** Deterministic execution engines guided by probabilistic insights.
    *   Dependencies: `core/agents/agent_base.py`, `core/agents/fundamental_analyst_agent.py`, `core/agents/market_sentiment_agent.py`
*   **Agent State & Interfaces:**
    *   Dependencies: `core/schemas/agent_schema.py`

## 3. Human Gates & Feedback Loops
*   **Governance & Approvals:** Ensures HITL (Human-in-the-Loop) integration when conviction thresholds are not met.
    *   Dependencies: `src/pdil/middleware.py` (`SecurityGovernanceGatekeeper`, `JsonLogicGovernanceGatekeeper`)
*   **Self-Healing & Drift Intelligence:** Detects and corrects behavioral shifts.
    *   Dependencies: `src/pdil/middleware.py` (`DriftIntelligenceLayer`)

## 4. Sandboxes, MCP Servers, and Logs
*   **MCP Server (Model Context Protocol):** Universal tool-calling interface.
    *   Dependencies: `server/mcp_server.py`, `src/mcp_server.py`, `core/mcp/schemas.py`
*   **Logging & Telemetry (PROV-O Compliant):**
    *   Dependencies: `core/utils/logging_utils.py` (`SwarmLogger`), `src/pdil/middleware.py` (`ProofOfThoughtLogger`, `MilestoneLogger`)

## 5. Core Utility Files & Schemas
*   **PDIL Models & Metadata:**
    *   Dependencies: `src/pdil/models.py` (`ProvenanceHeader`)
*   **Core Execution Interfaces:** Standardized agent I/O for horizontal scaling.
    *   Dependencies: `src/schemas/core_types.py` (`AgentInput`, `AgentOutput`)
*   **AFOS Types:**
    *   Dependencies: `afos/afos/core_types.py` (`Obligor`, `Facility`, `RiskAssessmentOutput`, `ModelArbitrationOutput`)

## 6. Time Management, Resource Constraints, and Record Keeping
*   **Time & Token Usage:** Temporal workflow components and resource limit tracking.
    *   Dependencies: Temporal execution layers, `core/schemas/observability.py` (`AgentTelemetry`, `Trace`)
*   **Audit & Record Keeping:** Bitemporal lineage, historical execution trace storage.
    *   Dependencies: `core/schemas/hnasp.py` (`ExecutionTrace`)

## 7. Documentation
*   **System Guidelines & Context:**
    *   Dependencies: `README.md`, `AGENTS.md`, `AGENT_CATALOG.md`, `llms.txt`, `llms-full.txt`
