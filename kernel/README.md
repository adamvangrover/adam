# Adam v3.0 / v30.1: Core Kernel Execution & Routing Strategy

## 1. Product Backlog & Prioritization

The transition to a zero-trust, deterministic Core Kernel necessitates a rigorous, phase-by-phase approach. Below is the implementation backlog prioritizing modularity, zero-trust statefulness, and mathematically rigid evaluation metrics, mapping the architecture to the 10-Layer Institutional Evaluation Framework.

### Phase 1: Foundational Schema & State Determinism
*   **Objective:** Establish the unshakeable JSON-based ground truth.
*   **Tasks:**
    *   **1.1** Define the `kernel_rpc.json` schema (Milestone 1) to standardize all core system capabilities and tool calls via JSON-RPC 2.0. (Addresses Layer 3: Agent Orchestration).
    *   **1.2** Implement `state_engine.json` to enforce strict state transitions and provide immutable ledgering of all agent actions. (Addresses Layer 5: State Transition Evaluation).
    *   **1.3** Establish `logic_rules.json` to embed JSONLogic guards natively, validating Expected Loss (EL) equations deterministically. (Addresses Layer 1: Deterministic Validation & Layer 4: Credit Risk Policy Compliance).

### Phase 2: Trace-Native Evaluation Transformation
*   **Objective:** Upgrade `adam_credit_eval_flywheel.py` from a passive script to a trace-native evaluator.
*   **Critical Path:**
    *   **2.1** Refactor the agent execution loop to output structured JSON payloads containing `TraceID`, `SpanID`, `ParentSpanID`, and step-by-step reasoning tokens.
    *   **2.2** Integrate Layer 6 (Information Gain Metric) by parsing the trace payloads to count net-new facts extracted per tool call.
    *   **2.3** Integrate Layer 8 (Calibration Metric) by logging agent confidence scores at each state transition and checking them against retrieved evidence bounds using JSONLogic.
    *   **2.4** Ensure the `GovernanceGatekeeper` intercepts all actions and enforces a minimum ConvictionScore threshold of 0.5.

### Phase 3: Institutional Intelligence & Routing Verification
*   **Objective:** Ensure multi-agent orchestration produces outputs matching the macro-physics and risk rigors of the "Market Mayhem" intelligence briefs.
*   **Tasks:**
    *   **3.1** Standardize System 1 and System 2 cognitive primitives in `prompt_matrix.jsonl`.
    *   **3.2** Implement automated tests for Layer 2 (Grounding & Hallucination Control) by cross-referencing claims directly with scraped SEC/market context.
    *   **3.3** Enforce Layer 7 (Risk-Control Evaluation) and Layer 10 (Economic Consistency) validations via deterministic post-checks on the final EL equations and generated narratives.

### Phase 4: Full Deployment & Continuous Flywheel
*   **Objective:** CI/CD integration and deployment of the neuro-symbolic engine.
*   **Tasks:**
    *   **4.1** Connect the evaluation flywheel to a daily automated CI/CD pipeline.
    *   **4.2** Benchmark Layer 9 (Rating Decision Accuracy) using ground-truth historical corporate default datasets.

---

## 2. Model Selection & Routing Strategy

To maximize efficiency and accuracy while preserving an isolated, stateful JSON runtime, the Adam v3.0 Core Kernel utilizes dynamic model routing based on task complexity and required output format.

1.  **High-Context Synthesis & Institutional Narrative (System 2):**
    *   **Model:** Gemini (e.g., Gemini-2.5-Flash or Pro).
    *   **Use Case:** Executing multi-page deep-dive credit memos, linking macro conditions to sector risks, and writing the final "Market Mayhem" style executive summaries.
    *   **Why:** Superior long-context window handling and strong reasoning capabilities for complex economic logic.

2.  **Meticulous Formatting & Code Generation:**
    *   **Model:** Claude (e.g., Claude 3.5 Sonnet).
    *   **Use Case:** Enforcing strict Pydantic models, translating reasoning into exact JSON-LD structures, or generating architectural code.
    *   **Why:** Unmatched adherence to complex output schemas and instruction following.

3.  **Rapid Tool Calling & System 1 Perception:**
    *   **Model:** GPT-4o.
    *   **Use Case:** Quick JSON-RPC executions, fast data retrieval, SEC Edgar scraping, and parsing continuous market streams (WebSocket ingestion).
    *   **Why:** Industry-leading tool-calling latency and accuracy for discrete, deterministic API interactions.

**Routing Enforcement Mechanism:**
The `Orchestrator` node acts as the dynamic router. Before dispatching a task, it analyzes the `state_engine.json` requirements and tags the `prompt_matrix.jsonl` entry with a `target_engine` parameter. The prompt execution harness then maps the target to the appropriate LLM provider client while treating the model simply as a stateless compute engine returning deterministic text/JSON. The `GovernanceGatekeeper` intercepts the response, applies `logic_rules.json`, and if bounds are met, commits the transition to the immutable ledger.
