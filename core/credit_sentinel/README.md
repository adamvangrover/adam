# Credit Sentinel: Distressed Debt Analytics

**Credit Sentinel** is Adam's specialized module for high-fidelity credit risk assessment. It automates the workflow of a distressed debt analyst, combining quantitative modeling with qualitative reasoning.

## 🏗️ Architecture

The module is composed of three layers:

### 1. Data Ingestion & Processing (`data_ingestion/`)
*   **Universal Ingestor:** Handles raw financial data from APIs (FMP, SEC) or documents (PDFs).
*   **ICAT Engine:** The core pipeline:
    *   **I**ngest: Fetch raw data.
    *   **C**lean: Normalize line items (e.g., mapping "Revenue" and "Total Sales" to `revenue`).
    *   **A**nalyze: Compute derived metrics.
    *   **T**ransform: Output standardized Pydantic models.

### 2. Quantitative Modeling (`models/`, `agents/ratio_calculator.py`)
*   **Ratio Calculator:** Deterministically computes key credit ratios (Leverage, Interest Coverage, Quick Ratio).
*   **Distress Classifier:** A Random Forest model (or fallback heuristic) that predicts the probability of default based on computed ratios.

### 3. Qualitative Reasoning (`agents/risk_analyst.py`)
*   **Risk Analyst Agent:** A "System 2" agent that consumes the quantitative outputs.
*   **Role:**
    *   Validates the model's prediction against market context (News, Macro).
    *   Checks for specific "red flags" (e.g., Covenant breaches).
    *   Generates a human-readable Investment Memo.

## 🚀 Workflow

1.  **Trigger:** User requests "Analyze distress risk for AMC".
2.  **Fetch:** System retrieves latest financials via `retrieve_market_data`.
3.  **Compute:** `RatioCalculator` derives metrics (e.g., Debt/EBITDA = 8.5x).
4.  **Predict:** `DistressClassifier` flags "High Risk" (Prob: 0.89).
5.  **Synthesize:** `RiskAnalyst` reads the data, sees the high leverage, checks news for "restructuring advisors", and drafts a report.

## 📦 Key Classes

*   **`RiskAnalyst`**: The primary agent interface.
*   **`DistressClassifier`**: The ML model wrapper.
*   **`RatioCalculator`**: The financial math engine.

## 🛠️ Usage (Python)

```python
from core.credit_sentinel.agents.risk_analyst import RiskAnalyst, AgentInput
from core.credit_sentinel.agents.ratio_calculator import RatioCalculator

# 1. Prepare Data
financials = {
    'ebitda': 500,
    'total_debt': 4500,  # High leverage
    # ... other fields
}

# 2. Calculate Ratios
ratios = RatioCalculator().calculate_all(financials)

# 3. Run Agent
agent = RiskAnalyst()
result = agent.execute(AgentInput(
    query="Analyze credit health",
    context={"financials": financials, "ratios": ratios}
))

print(result.answer)
```
