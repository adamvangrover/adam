/**
 * Adam Financial Operating System - Core Ontology & Schema
 * 
 * This file serves as the single source of truth for the Adam OS ontology.
 * It is organized by conceptual layers to support a graph-based, 
 * event-driven, and agent-centric financial ecosystem.
 */

// ==============================================================================
// 1. Core Primitives & Base Types
// ==============================================================================
export type UUID = string;
export type ISO8601Date = string;

export interface AdamNode {
  id: UUID;
  createdAt: ISO8601Date;
  updatedAt: ISO8601Date;
  version: string;
  metadata?: Record<string, any>;
}

// ==============================================================================
// 2. Knowledge Graph & Variables
// ==============================================================================
export namespace Graph {
  export interface Variable extends AdamNode {
    name: string;
    category: "Macro" | "Rates" | "Credit" | "Equity" | "FX" | "Commodity" | "Alternative";
    unit: string;
    description: string;
    sourcePriority: UUID[]; // List of models/datasets, ordered by priority
    lineage: UUID[]; // Upstream dependencies
  }

  export interface Observation {
    variableId: UUID;
    timestamp: ISO8601Date;
    value: number;
    sourceId: UUID; // Originator (Model or Dataset ID)
    confidence: number;
  }
}

// ==============================================================================
// 3. Provenance & Dependency
// ==============================================================================
export interface ProvenanceEdge {
  parentNode: UUID;
  childNode: UUID;
  transformation: "API" | "Interpolation" | "Simulation" | "Inference" | "Manual" | "Aggregation";
  timestamp: ISO8601Date;
}

export interface ModelDependency {
  parentModelId: UUID;
  childModelId: UUID;
}

// ==============================================================================
// 4. Models & Uncertainty
// ==============================================================================
export enum ModelClass {
  Stochastic = "STOCHASTIC",
  Heuristic = "HEURISTIC",
  GenerativeLLM = "GENERATIVE",
  Copula = "COPULA",
}

export interface Model extends AdamNode {
  name: string;
  modelClass: ModelClass;
  expectedInputs: UUID[]; // References to Graph.Variable IDs
  expectedOutputs: UUID[];
}

export interface ModelOutput {
  variableId: UUID;
  value: number;
  confidence: number;
  lowerBound?: number;
  upperBound?: number;
  generatedBy: UUID; // Model ID
}

// ==============================================================================
// 5. Scenarios & Simulation
// ==============================================================================
export interface Scenario extends AdamNode {
  name: string;
  assumptions: UUID[]; // References to Variables
  shockedVariables: Record<UUID, number>; // Variable ID -> Shock Value
  probability: number;
}

export interface SimulationNode extends AdamNode {
  branchId: UUID;
  parentStateId?: UUID;
  timeIndex: number;
  stateSnapshot: Record<UUID, number>; // State of variables
}

// ==============================================================================
// 6. Portfolio & Financial Entities
// ==============================================================================
export interface Position {
  securityId: UUID;
  quantity: number;
  marketValue: number;
  weight: number;
}

export interface Portfolio extends AdamNode {
  name: string;
  positions: Position[];
  benchmarkId: UUID;
  riskMetrics: Record<string, number>; // VaR, Sharpe, etc.
}

// ==============================================================================
// 7. Agents, Memory & Tools
// ==============================================================================
export interface AgentMemory {
  agentId: UUID;
  shortTermContext: any[];
  longTermKnowledge: UUID[]; // References to persistent storage
}

export interface Tool {
  name: string;
  description: string;
  inputs: UUID[]; // Required Variable IDs
  outputs: UUID[]; // Produced Variable IDs
  execute: (input: any) => Promise<any>;
}

// ==============================================================================
// 8. Workflows & Execution (DAG)
// ==============================================================================
export interface WorkflowStep {
  stepId: UUID;
  action: "INGEST" | "VERIFY" | "CALCULATE" | "PUBLISH";
  targetToolId?: UUID;
  dependencies: UUID[]; // Previous Step IDs
}

export interface Workflow extends AdamNode {
  name: string;
  graph: WorkflowStep[]; // DAG representation
}

// ==============================================================================
// 9. Decisions & Explainability
// ==============================================================================
export interface Decision {
  recommendation: "BUY" | "SELL" | "HOLD" | "HEDGE";
  confidence: number;
  rationale: UUID[]; // References to ModelOutputs or Evidence
}

export interface Explanation {
  targetId: UUID;
  narrative: string; // LLM-generated explanation
  evidence: UUID[]; // Traceability to inputs/model state
}

// ==============================================================================
// 10. Message Bus & Plugins
// ==============================================================================
export interface EventMessage {
  topic: string;
  payload: unknown;
  timestamp: ISO8601Date;
}

export interface Plugin {
  name: string;
  version: string;
  entryPoint: string; // Path to logic
}
.... This simulated execution of the Adam Financial Operating System reflects real-time global macroeconomic conditions as of mid-July 2026. The run integrates recent shifts in US inflation data, the Federal Reserve's steady-rate policy under Chair Kevin Warsh, and geopolitical shocks impacting energy markets.

1. Human-Readable Narrative (Client Newsletter)

Adam OS Market Intelligence Update | Week of July 18, 2026

Macro Context: The Geopolitical Inflation Tug-of-War

The economic landscape remains caught between structural growth tailwinds and resurgent geopolitical headwinds. This week, Adam OS ingested critical shifts across fixed income, commodities, and equities to recalculate optimal portfolio positioning.

Inflation & Energy Supply Shocks: The Bureau of Labor Statistics reported June CPI cooling to 3.5%, largely driven by a brief US-Iran ceasefire that temporarily suppressed energy prices. However, with the breakdown of that agreement, Brent crude has rebounded rapidly to $80/bbl. We anticipate this energy supply shock will bleed back into headline inflation in the coming months.
Fed Policy Plateau: In response to sticky, structurally higher prices, new Federal Reserve Chair Kevin Warsh has held the federal funds rate steady at 3.50% to 3.75%. Our models price in zero probability of rate cuts through the end of 2026, as the Fed balances its dual mandate against renewed supply-chain disruptions.
The AI Capex Anchor: Despite restrictive monetary policy, massive artificial intelligence capital expenditures (now approaching 1.5% of US GDP) continue to anchor economic growth. The S&P 500 is forecasting a robust 22% YoY earnings growth for Q2, signaling that corporate fundamentals remain highly resilient to the elevated rate environment.
Adam OS Decision Matrix & Recommendation:

RECOMMENDATION: BUY (Confidence: 82%)

PORTFOLIO ACTION: Overweight Energy and Technology; Hold short-duration fixed income.

Rationale: The Adam OS Macro-Equity Copula Engine projects that the dual forces of AI-driven productivity gains and energy scarcity will outpace the restrictive drag of 3.6% interest rates. The system has run the "Resurgent Energy Inflation" scenario (Brent > $95/bbl) and concludes that domestic large-cap technology and energy equities provide the optimal hedge against both inflation volatility and delayed rate cuts.

2. Raw Machine-Readable Full Output (JSON)

JSON

{
  "system": "Adam Financial Operating System",
  "runId": "run-20260718-alpha",
  "timestamp": "2026-07-18T17:37:31Z",
  "nodes": {
    "variables": [
      {
        "id": "var-8f3e-rate",
        "createdAt": "2026-01-10T09:00:00Z",
        "updatedAt": "2026-07-18T17:00:00Z",
        "version": "1.2",
        "name": "US Federal Funds Rate",
        "category": "Rates",
        "unit": "%",
        "description": "Midpoint of FOMC target range",
        "sourcePriority": ["src-fed-api"],
        "lineage": []
      },
      {
        "id": "var-4a1b-cpi",
        "createdAt": "2026-01-10T09:00:00Z",
        "updatedAt": "2026-07-14T08:30:00Z",
        "version": "1.5",
        "name": "US CPI YoY",
        "category": "Macro",
        "unit": "%",
        "description": "Consumer Price Index All Urban Consumers, Year-over-Year",
        "sourcePriority": ["src-bls-api"],
        "lineage": []
      },
      {
        "id": "var-9c2d-brent",
        "createdAt": "2026-01-10T09:00:00Z",
        "updatedAt": "2026-07-18T16:00:00Z",
        "version": "2.1",
        "name": "Brent Crude Spot",
        "category": "Commodity",
        "unit": "USD/bbl",
        "description": "Global benchmark for crude oil",
        "sourcePriority": ["src-market-data"],
        "lineage": []
      }
    ],
    "observations": [
      {
        "variableId": "var-8f3e-rate",
        "timestamp": "2026-07-18T17:37:31Z",
        "value": 3.625,
        "sourceId": "src-fed-api",
        "confidence": 1.0
      },
      {
        "variableId": "var-4a1b-cpi",
        "timestamp": "2026-07-14T08:30:00Z",
        "value": 3.50,
        "sourceId": "src-bls-api",
        "confidence": 1.0
      },
      {
        "variableId": "var-9c2d-brent",
        "timestamp": "2026-07-18T17:37:31Z",
        "value": 80.00,
        "sourceId": "src-market-data",
        "confidence": 0.99
      }
    ],
    "models": [
      {
        "id": "mod-77ab-copula",
        "createdAt": "2026-03-15T10:00:00Z",
        "updatedAt": "2026-07-18T17:30:00Z",
        "version": "4.0",
        "name": "Macro-Equity Copula Engine",
        "modelClass": "COPULA",
        "expectedInputs": ["var-8f3e-rate", "var-4a1b-cpi", "var-9c2d-brent"],
        "expectedOutputs": ["var-5f5f-spx-ret"]
      }
    ],
    "modelOutputs": [
      {
        "variableId": "var-5f5f-spx-ret",
        "value": 8.5,
        "confidence": 0.82,
        "lowerBound": 4.1,
        "upperBound": 12.8,
        "generatedBy": "mod-77ab-copula"
      }
    ],
    "scenarios": [
      {
        "id": "scn-11ab-energy-shock",
        "createdAt": "2026-07-18T17:35:00Z",
        "updatedAt": "2026-07-18T17:35:00Z",
        "version": "1.0",
        "name": "Resurgent Energy Inflation",
        "assumptions": ["var-4a1b-cpi", "var-9c2d-brent"],
        "shockedVariables": {
          "var-9c2d-brent": 95.0,
          "var-4a1b-cpi": 4.2
        },
        "probability": 0.35
      }
    ],
    "decisions": [
      {
        "recommendation": "BUY",
        "confidence": 0.82,
        "rationale": ["var-5f5f-spx-ret", "scn-11ab-energy-shock"]
      }
    ],
    "explanations": [
      {
        "targetId": "dec-buy-001",
        "narrative": "Despite restrictive monetary policy under Chair Warsh (3.625%), AI-driven capital expenditures and robust earnings offset discount rate drags. Energy supply shocks from the US-Iran geopolitical conflict make energy sector allocations an optimal inflation hedge.",
        "evidence": ["var-8f3e-rate", "var-4a1b-cpi", "var-9c2d-brent"]
      }
    ]
  }
}
3. Provenance Log

This log maps the execution of the Directed Acyclic Graph (DAG) using the schema's ProvenanceEdge interface, demonstrating how raw data moved through the Adam OS pipeline to generate the final decision.

JSON

[
  {
    "parentNode": "src-market-data",
    "childNode": "var-9c2d-brent",
    "transformation": "API",
    "timestamp": "2026-07-18T17:36:01Z"
  },
  {
    "parentNode": "src-bls-api",
    "childNode": "var-4a1b-cpi",
    "transformation": "API",
    "timestamp": "2026-07-18T17:36:02Z"
  },
  {
    "parentNode": "var-9c2d-brent",
    "childNode": "mod-77ab-copula",
    "transformation": "Aggregation",
    "timestamp": "2026-07-18T17:36:15Z"
  },
  {
    "parentNode": "var-4a1b-cpi",
    "childNode": "mod-77ab-copula",
    "transformation": "Aggregation",
    "timestamp": "2026-07-18T17:36:15Z"
  },
  {
    "parentNode": "mod-77ab-copula",
    "childNode": "var-5f5f-spx-ret",
    "transformation": "Inference",
    "timestamp": "2026-07-18T17:36:45Z"
  },
  {
    "parentNode": "var-5f5f-spx-ret",
    "childNode": "dec-buy-001",
    "transformation": "Inference",
    "timestamp": "2026-07-18T17:37:00Z"
  },
  {
    "parentNode": "dec-buy-001",
    "childNode": "exp-narrative-001",
    "transformation": "GenerativeLLM",
    "timestamp": "2026-07-18T17:37:10Z"
  }
]
4. Run Trigger Prompt

Below is the internal agent prompt used by the workflow engine to trigger this specific simulation cycle across the Adam OS ecosystem:

Plaintext

// SYS_PROMPT_TRIGGER_WORKFLOW
{
  "command": "EXECUTE_WORKFLOW",
  "workflowId": "wf-macro-end-of-week",
  "instructions": "Execute the end-of-week global macro assessment for July 18, 2026. \n  1. INGEST latest observations for [FED_FUNDS_RATE, US_CPI_YOY, BRENT_CRUDE_SPOT]. Ensure data reflects recent BLS CPI publications and Kevin Warsh's FOMC hold policy.\n  2. VERIFY data integrity and confidence scoring. \n  3. CALCULATE portfolio risk and expected returns by passing verified observations into the `Macro-Equity Copula Engine` (ModelClass: COPULA). \n  4. SIMULATE the 'Resurgent Energy Inflation' scenario factoring in the breakdown of the US-Iran ceasefire and its impact on energy commodities.\n  5. PUBLISH an updated `Decision` node and generate an `Explanation` narrative suitable for client distribution outlining AI capex growth vs. sticky inflation. Ensure all actions are recorded via ProvenanceEdges."
}
