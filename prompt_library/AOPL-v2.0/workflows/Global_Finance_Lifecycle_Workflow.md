# Global Finance Lifecycle Workflow
**Objective**: Build a sequential, data-driven workflow that chains the Universal Finance Ecosystem prompts together, utilizing intermediate data artifacts to generate a comprehensive, actionable institutional risk and credit framework.

## Workflow Pipeline & Interim Artifacts

This workflow dictates how autonomous agents or LLMs should execute the prompts sequentially, passing structured JSON artifacts between each stage to build cumulative knowledge.

### Stage 1: The Macro Foundations
**Prompt Executed**: `encyclopedic_knowledge/Universal_Finance_Ecosystem_Deep_Dive.md`
**Process**: The LLM establishes the foundational reality of the market, defining the core components (dynamic hedging, GSIBs, central ledger, etc.).
**Interim Artifact Generated**: `Macro_House_View_Base.json`
*Structure*:
```json
{
  "macro_regime": "Descriptive string of current macroeconomic reality.",
  "core_components": ["list", "of", "defined", "mechanisms"],
  "market_participants": {
    "retail": "definition and exposure",
    "institutional": "definition and exposure"
  },
  "systemic_misconceptions_cleared": ["list", "of", "clarified", "truths"]
}
```

### Stage 2: Methodological Synthesis
**Prompt Executed**: `fields_of_study/Global_Institutional_Finance_Synthesis.md`
**Process**: The LLM ingests `Macro_House_View_Base.json` alongside the prompt to determine *how* these entities are practically managed using modern theories (MPT, APT, algorithmic execution).
**Interim Artifact Generated**: `Operational_Risk_Methodology.json`
*Structure*:
```json
{
  "active_theories": ["list", "of", "applied", "theories"],
  "counterparty_tracking_framework": "Detailed methodology string.",
  "pricing_mechanisms": {
    "debt": "Methodology",
    "equity": "Methodology",
    "derivatives": "Methodology"
  },
  "frontier_debates_flagged": ["list", "of", "active", "systemic", "debates"]
}
```

### Stage 3: Autonomous Knowledge Expansion & Implementation
**Prompt Executed**: `independent_research/Financial_Lifecycle_Knowledge_Expansion.md`
**Process**: The LLM ingests both `Macro_House_View_Base.json` and `Operational_Risk_Methodology.json`. It identifies operational gaps in the current pipeline (e.g., how the family office risk scoring interfaces with the central GSIB ledger) and synthesizes missing actionable protocols.
**Final Artifact Generated**: `Unified_Ledger_Protocol.json`
*Structure*:
```json
{
  "identified_gaps": ["list", "of", "operational", "blindspots"],
  "synthesized_solutions": [
    {
      "gap": "Description",
      "actionable_protocol": "Step-by-step risk management or pricing protocol."
    }
  ],
  "integration_map": "Blueprint for implementing these protocols into the firm's central macro house view."
}
```

## Execution Instructions for Agents
1. Execute Stage 1 and explicitly validate the JSON schema of `Macro_House_View_Base.json`.
2. Pass the validated `Macro_House_View_Base.json` as context to the LLM executing Stage 2.
3. Validate `Operational_Risk_Methodology.json`.
4. Pass both preceding JSON artifacts as context for Stage 3 to generate the final, actionable `Unified_Ledger_Protocol.json`.