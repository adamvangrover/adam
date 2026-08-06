# AFOS Simulation Kernel Specification

## Overview
The Simulation Kernel enables institutional risk management to live and die by rigorous what-if analysis. It allows management to project the impact of policy changes, macroeconomic shocks, and portfolio restructurings before deploying them to production.

## The Counterfactual Engine

The Simulation Kernel leverages the separation of concerns provided by the other kernels to execute counterfactual scenarios:

### Example: Policy Impact Analysis
"What if we had deployed Policy Version 3.2 six months ago?"

1. **Load:** Retrieve the Historical Portfolio state from the Knowledge Kernel.
2. **Inject:** Load Policy Version 3.2 from the Governance Kernel.
3. **Replay:** Use the Execution Kernel to run the historical Evidence through the new Policy.
4. **Compare:** Generate alternative Ratings, Capital Impact, Loss Forecasts, and Regulatory Impacts.

### Example: Macroeconomic Stress Testing
"What happens to our portfolio if interest rates spike 200bps?"

1. **Define:** Create a `Scenario` object with the shock parameters.
2. **Project:** Apply the shock to the current portfolio state.
3. **Evaluate:** Run the stressed Evidence through the active Policies via the Decision Kernel.
4. **Analyze:** Aggregate the resulting Decision Graphs to forecast portfolio-wide defaults or covenant breaches.
