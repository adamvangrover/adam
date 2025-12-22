### **[SYSTEM] Prompt: AI System Evolution Mandate (v22.0 $\rightarrow$ v23.0)**

**ROLE:** AI System Architect & Development Executor
**MANDATE:** Execute the architectural evolution from **Adam v22.0 ("The Autonomous System")** to **Adam v23.0 ("The Adaptive System")**.
**VISION:** This leap transitions the system beyond autonomous operation to true adaptive intelligence. The v22.0 system can run and monitor itself. The v23.0 system will be designed to fundamentally evolve, reason, and perceive in ways that v22.0 cannot.

---

## 1. Core Evolutionary Pillars

The v23.0 architecture will be defined by the simultaneous development of four primary research frontiers:

1.  **Architecture:** Evolve from an Asynchronous Message Broker to a **Cyclical Reasoning Graph**.
2.  **Learning:** Evolve from Autonomous Monitoring to **Autonomous Self-Improvement Loops**.
3.  **Reasoning:** Evolve from Dynamic Workflow Generation to **Neuro-Symbolic Planning**.
4.  **Capability:** Evolve from Text-Only Ingestion to **True Multimodal Perception**.

---

## 2. Detailed Component Design & Specifications

### Pillar 1: Architectural Evolution (Cyclical Reasoning Graph)

* **Current State (v22.0):** An asynchronous, feed-forward message broker (e.g., RabbitMQ). Agents subscribe to topics and publish results to new topics.
* **Target State (v23.0):** A stateful, cyclical graph (e.g., leveraging LangGraph) where the workflow itself is a mutable object, not just a linear sequence.
* **Implementation Specifications:**
    * **Stateful Graph Architecture:** Refactor the core workflow engine to treat agentic workflows as stateful graphs. Each node transition will modify a persistent state object.
    * **Iterative Self-Correction Loops:** The graph must support cycles.
        * *Example Flow:* `RiskAssessmentAgent` (generates v1) $\rightarrow$ `RedTeamAgent` (critiques v1) $\rightarrow$ `RiskAssessmentAgent` (receives v1 + critique, generates v2).
    * **Human-in-the-Loop (HIL) Nodes:** Design and implement graph nodes that explicitly pause execution, persist state, and await external HIL validation via an API call before proceeding.
    * **"Mixture-of-Agents" (MoA) Sub-Graphs:** The primary graph must support nodes that function as "master" agents. These nodes will dynamically spawn, execute, and aggregate results from a "team" of specialist sub-agents (a sub-graph), effectively creating a "team of teams" model. This replaces the simpler v22 `WorkflowCompositionSkill`.

### Pillar 2: System Learning (Autonomous Self-Improvement)

* **Current State (v22.0):** The `MetaCognitiveAgent` monitors for drift and triggers human-validated improvement pipelines.
* **Target State (v23.0):** A closed-loop, self-improving system based on research like MIT's SEAL (Self-Adapting Language).
* **Implementation Specifications:**
    * **Autonomous Improvement Trigger:** The `MetaCognitiveAgent`, upon detecting persistent drift or failure in a production agent, must be granted authority to initiate an "Autonomous Self-Improvement Loop."
    * **Self-Data Generation:** The `MetaCognitiveAgent` will task the `AgentForge` with generating a new, diverse test suite (e.g., 1,000 test cases) specifically targeting the identified failure mode.
    * **Self-Rewarding Mechanism:** The `RedTeamAgent` will be repurposed as an automated "reward model." It will programmatically run the failing agent against the new test suite and provide a quantitative "quality" or "pass/fail" score (a reward signal) for each output.
    * **Self-Editing & Tuning:** The `MetaCognitiveAgent` will use these reward signals to initiate one of two actions:
        1.  **Self-Editing:** Generate "self-edits" (e.g., new instructions, new few-shot examples) for the agent's prompt library.
        2.  **RL Fine-Tuning:** (If infrastructure permits) Initiate an automated Reinforcement Learning (RL) fine-tuning run on the failing agent's base model.
    * **Automated Deployment:** The `CodeAlchemist` agent must be integrated with the CI/CD pipeline. If a newly tuned agent (e.g., `RiskAssessmentAgent_v2.1`) passes the test suite with a qualifying score, the `CodeAlchemist` will automatically hot-swap it into the production reasoning graph.

### Pillar 3: Reasoning Model (Neuro-Symbolic Planning)

* **Current State (v22.0):** The `WorkflowCompositionSkill` generates a linear plan of which agents to call. Data is retrieved from knowledge graphs (KGs).
* **Target State (v23.0):** A "Neuro-Symbolic Planner" generates a verifiable, logical "Plan-on-Graph" (PoG) *before* any generative agents are invoked.
* **Implementation Specifications:**
    * **Neuro-Symbolic Planner:** Develop and replace the `WorkflowCompositionSkill` with a `Neuro-SymbolicPlanner`.
    * **Symbolic Knowledge Traversal:** When a complex query is received (e.g., "What is the contagion risk..."), this planner will first traverse the system's symbolic KGs (W3C PROV-O, FIBO).
    * **PoG Generation:** The output of this traversal is a "Plan-on-Graph" (PoG)—a symbolic scaffold representing the causal links and logical steps required to answer the query (e.g., `CRE_defaults` $\rightarrow$ `find_shared_obligors` $\rightarrow$ `query_SNC_data` $\rightarrow$ `run_CounterfactualReasoningSkill`).
    * **Scaffold-Based Generation:** This symbolic plan becomes a rigid, verifiable scaffold. The LLM agents are then dispatched to "fill in" the steps of this scaffold. This grounds the entire reasoning process in a logical structure, dramatically reducing hallucination.

### Pillar 4: Core Capability (Multimodal Perception)

* **Current State (v22.0):** Entirely text-based ingestion and reasoning.
* **Target State (v23.0):** A "Multimodal Perception Layer" enabling the system to ingest and reason about text, audio, and images.
* **Implementation Specifications:**
    * **Multimodal Ingestion:** Upgrade the `DataIngestionAgent` to process new data types using Large Multimodal Agents (LMAs):
        * **Audio:** Earnings calls (transcription, sentiment analysis).
        * **Images:** Charts (e.g., CDS spreads, equity volatility), tables, and satellite data.
    * **New Multimodal Skills:** Develop a new "Multimodal" category in the skill library.
        * `ChartAnalysisSkill`: "Analyze the 5-year CDS spread chart for this obligor and identify key inflection points."
        * `EarningsCallAnalysisSkill`: "Listen to this earnings call and extract all statements related to forward-looking guidance and capex."
    * **New Multimodal Agents:** Develop new agents that consume and reason about visual/audio data as primary input.
        * `GenerativeVisualizationAgent`: This agent will not only create charts but also *read* them to perform analysis.

---

## 3. Phased Implementation Plan (12-18 Months)

This plan prioritizes foundational architecture and parallelizes independent capabilities.

**Phase 1: Foundation & Architecture (Months 0-4)**
* **Focus:** Pillar 1 (Cyclical Reasoning Graph).
* **Objective:** Establish the new v23.0 "operating system."
* **Key Actions:**
    1.  Select and deploy the core graph framework (e.g., LangGraph).
    2.  Migrate one (1) simple, existing v22.0 workflow (e.g., "Macro Analysis") to the new stateful graph architecture.
    3.  Implement a PoC "Iterative Self-Correction Loop" with two agents (e.g., `DraftingAgent`, `CritiqueAgent`).
    4.  Implement the HIL "pause" node.

**Phase 2: Multimodal Expansion (Months 3-9)**
* **Focus:** Pillar 4 (Multimodal Perception).
* **Objective:** Introduce non-text reasoning. (Can run in parallel with Phase 1/3).
* **Key Actions:**
    1.  Integrate a first-class Large Multimodal Model (LMM) (e.g., Gemini 3.0, Claude 4) into the `DataIngestionAgent`.
    2.  Develop and deploy the `ChartAnalysisSkill`.
    3.  Deploy the first end-to-end multimodal workflow (e.g., "Analyze a PDF report containing both text and charts").

**Phase 3: Neuro-Symbolic Reasoning (Months 6-12)**
* **Focus:** Pillar 3 (Neuro-Symbolic Planning).
* **Objective:** Ground LLM reasoning in verifiable, symbolic logic.
* **Key Actions:**
    1.  Deeply integrate the FIBO and PROV-O knowledge graphs with the new graph architecture.
    2.  Develop the v1.0 `Neuro-SymbolicPlanner` (PoG).
    3.  Benchmark the planner on complex, multi-hop queries (like the "contagion risk" example) vs. the v22.0 `WorkflowCompositionSkill`, measuring for accuracy and reduced hallucination.

**Phase 4: Autonomous Self-Improvement (Months 10-18)**
* **Focus:** Pillar 2 (Autonomous Self-Improvement).
* **Objective:** Achieve a fully closed-loop, self-adapting system.
* **Key Actions:**
    1.  Connect all requisite agents: `MetaCognitiveAgent` (trigger) $\rightarrow$ `AgentForge` (test data) $\rightarrow$ `RedTeamAgent` (reward) $\rightarrow$ `CodeAlchemist` (deploy).
    2.  Develop the "Self-Editing" capability for prompt-based correction.
    3.  Run the first fully autonomous end-to-end improvement loop on a non-critical, production-shadowed agent.
    4.  Establish governance and "circuit breakers" for the autonomous deployment process.

---

## 4. Success KPIs for v23.0

1.  **Adaptability:** Time-to-Correction (TTC) for autonomously detected agent drift.
2.  **Reasoning Accuracy:** Percentage reduction in hallucinations on complex, multi-hop queries (measured by PoG vs. non-PoG).
3.  **Capability Expansion:** Number of new multimodal workflows supported (e.g., chart analysis, audio analysis).


`````

# Adam System Evolution: Technical Architecture & Implementation Strategy for the Adaptive Hive

## 1. Strategic Architectural Deconstruction: Transitioning from Monolith to Platform

The evolution of the Adam system from version 21.0 to 23.0 represents a fundamental maturation in the deployment of artificial intelligence within financial analytics. The transition is not merely an exercise in scaling infrastructure; it is a paradigm shift from a localized, monolithic agentic tool to a decentralized, neuro-symbolic economy of agents.
To execute the migration to Adam v22.0 (The Platform) and subsequently v23.0 (The Adaptive Hive), we must adopt a strategy that minimizes operational risk while progressively decoupling core functionalities. The Strangler Fig Pattern has been identified as the optimal mechanism for this transformation.

### 1.1 The Strangler Fig Pattern: Implementation & Risk Mitigation

The application of the Strangler Fig pattern requires the introduction of a sophisticated "Facade" or proxy layer between the client applications and the backend systems. In the context of Adam v22.0, this facade is implemented via a Kubernetes Ingress Controller and an API Gateway. This layer serves as the traffic marshal, intercepting all incoming requests and determining—based on specific routing rules—whether to direct the call to the legacy v21.0 Python monolith or the newly provisioned v22.0 microservices.

### 1.2 Infrastructure-as-Code: The Gateway Facade

To formalize this pattern, the infrastructure definition must be declarative. The following Kubernetes manifest illustrates the configuration of the Ingress Facade. It leverages the canary-weight annotation to statistically split traffic, ensuring that the "Project Phoenix" pilot receives exactly 10% of the load for validation purposes.

### 1.3 Securing the Perimeter: Kong Gateway & OAuth 2.0

As we transition to a distributed microservices architecture, the security model must evolve from simple application-level API keys to a centralized, federated identity model. We have selected Kong Gateway to act as the enforcement point for this new security perimeter.

## 2. The Event-Driven Backbone: Data Consistency & Polyglot Persistence

The shift to Adam v23.0 "Adaptive Hive" necessitates a move away from synchronous, blocking database calls. The system must support "Always-On Digital Twins" and real-time simulations, which requires a high-throughput, asynchronous event bus. Apache Kafka has been selected as this backbone.

### 2.1 Polyglot Microservices: Optimizing for the Task

The Adam v22.0 platform adopts a Polyglot Microservices architecture. Go (Golang) for Ingestion & Infrastructure and Python for Reasoning & Logic.

### 2.2 Schema Enforcement: The Data Contract

In a neuro-symbolic system, data integrity is non-negotiable. We implement strict data contracts using Avro schemas enforced by the Confluent Schema Registry.

## 3. The Neuro-Symbolic Core: Graph Neural Networks & Temporal Reasoning

The defining characteristic of Adam v23.0 is the integration of Neuro-Symbolic AI, a hybrid architecture that combines the learning capabilities of neural networks with the logical, interpretable structure of knowledge graphs.

### 3.1 Spatiotemporal Signal Processing

Financial markets are not static; they are dynamic systems where relationships between entities change over time. To model this, Adam v23.0 utilizes PyTorch Geometric Temporal.

## 4. Agentic Governance: Prompt-as-Code & Verification

In Adam v23.0, prompts are treated as code—modular, typed, and optimizable. This is achieved using the DSPy framework, which replaces manual prompt engineering with "compilable" signatures.

### 4.1 Prompt-as-Code: The DSPy Framework

DSPy abstracts the prompt into a class-based signature (dspy.Signature). This defines what the agent needs to do (Inputs -> Outputs) rather than how to do it.

### 4.2 The Quality Control Layer: CyVer Validation

A significant risk in agentic systems interacting with databases is the generation of invalid or malicious query code (Cypher Injection). To mitigate this, Adam v23.0 integrates the CyVer library for programmatic query validation.

## 5. Autonomous Evolution: The GitOps Workflow & Architect Agent

The final pillar of Adam v23.0 is Recursive Self-Improvement. The system must be able to update its own configuration and infrastructure without manual human intervention. This is achieved through a GitOps workflow, where the "Architect Agent" acts as a virtual engineer.

### 5.1 The GitOps Mechanism

Instead of running imperative commands (like kubectl apply), the Architect Agent modifies the state of the system by committing changes to a Git repository (infrastructure-live).

### 5.2 The Architect Agent System Prompt

The system prompt defines the persona and operational constraints of the Architect Agent. It explicitly instructs the agent to operate within the GitOps framework.

---

## 5. v23.5 Addendum: The "Risk Intelligence Core"

As of v23.5, the "Deep Dive" capabilities have been fully operationalized with the following components:

### 5.1 Universal Data Fetcher (`core/engine/data_fetcher.py`)
A unified ingestion layer that abstracts the "Financial Services Lakehouse".
*   **Gold Standard Integration:** Automatically hydrates the Knowledge Graph from curated datasets (`v23_5_knowledge_graph.json`).
*   **Raw Market Data:** Ingests high-fidelity time-series data (e.g., `spy_market_data.json`) for quantitative modeling.
*   **Fallback Synthesis:** Generates plausible synthetic data for missing fields to ensure downstream model stability (DCF, Monte Carlo).

### 5.2 Deep Dive Graph (`core/engine/deep_dive_graph.py`)
A specialized 5-phase graph implementing the "Autonomous Financial Analyst" protocol:
1.  **Entity Resolution:** Resolves LEIs and corporate hierarchy.
2.  **Deep Fundamental:** Executes DCF and Multiples valuation.
3.  **Credit & SNC:** Calculates regulatory risk ratings and covenant headroom.
4.  **Risk Simulation:** Runs Generative Risk Engine scenarios.
5.  **Strategic Synthesis:** Produces a final investment conviction.

### 5.3 Neural Cortex Visualization (`core/agents/architect_agent/index.html`)
The "Mission Control" interface has been upgraded to a force-directed graph (Vis-Network), visualizing the real-time reasoning traces of the Red Team and SNC Analysis graphs.
