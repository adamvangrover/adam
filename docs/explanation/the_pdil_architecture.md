# The PDIL Architecture

## Probabilistic-to-Deterministic Integration Layer

The PDIL is the defining architectural component of ADAM. It serves as an immutable bridge and firewall between the stochastic world of Large Language Models (System 1) and the rigorous, deterministic world of financial mathematics and execution (System 2).

### Why Pure LLM Pipelines Fail for Risk Control
LLMs are inherently probabilistic. They excel at semantic understanding, parsing unstructured data (like M&A announcements or 10-K filings), and generating hypotheses. However, they suffer from hallucinations and lack the capability to perform reliable, exact arithmetic or enforce strict logical bounds. In institutional finance, an LLM calculating a Value-at-Risk (VaR) or assessing a debt covenant breach is an unacceptable compliance risk.

### The Neuro-Symbolic Routing Solution
Neuro-symbolic AI combines neural networks (pattern recognition) with symbolic logic (rules and reasoning).

In ADAM, the PDIL achieves this by:
1. **Extraction (Neural):** System 1 agents extract key variables (e.g., `ebitda_usd`, `total_revenue_usd`) from unstructured text.
2. **Validation (Bridge):** The extracted data is forced into strict Pydantic schemas. The `ProvenanceHeader` guarantees that every datapoint can be traced back to its origin file and line number.
3. **Execution (Symbolic/Deterministic):** The structured, validated data is passed to Rust-based kernels or JSONLogic evaluators. These deterministic engines run the actual calculations (e.g., covenant stress tests, pricing algorithms) without any LLM involvement.

### System State & Drift
The PDIL also monitors "observed_drift"—the divergence between the expected deterministic output and the probabilistic agent's assumptions. When drift exceeds a defined threshold, the system triggers a revalidation workflow, effectively enforcing self-correction.
