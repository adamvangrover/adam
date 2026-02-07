# Analysis Modules

The `core/analysis/` directory contains the specialized mathematical and analytical engines of Adam.

## 📊 Modules

### 1. Fundamental Analysis (`fundamental_analysis.py`)
Calculates intrinsic value using deterministic financial models.
*   **DCF (Discounted Cash Flow):** Projects future cash flows and discounts them back to present value.
*   **Ratio Analysis:** Computes P/E, P/B, Debt/Equity, and other key metrics from 10-K data.

### 2. Technical Analysis (`technical_analysis.py`)
Analyzes price action and market trends.
*   **Indicators:** RSI, MACD, Bollinger Bands.
*   **Pattern Recognition:** Detects "Head and Shoulders", "Double Bottom", etc.

### 3. Risk Assessment (`risk_assessment.py`)
The safety valve of the system.
*   **VaR (Value at Risk):** Estimates potential loss at a given confidence interval.
*   **Stress Testing:** Simulates portfolio performance under shock scenarios.

### 4. Explainable AI (`xai/`)
**"Trust but Verify."**
This submodule generates human-readable explanations for the models' decisions.
*   **Counterfactual Engine:** "What if interest rates were 1% higher?"
*   **Complexity Formula:** Scores the difficulty of a given analysis task.

## 🤝 Usage

These modules are typically called by **Specialized Agents** (e.g., `FundamentalAnalystAgent`), but can be used standalone for research.

```python
from core.analysis.fundamental_analysis import calculate_dcf

dcf_value = calculate_dcf(cash_flows=[100, 110, 120], discount_rate=0.10)
print(f"Intrinsic Value: ${dcf_value}")
```
