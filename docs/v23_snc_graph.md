# Shared National Credit (SNC) Analysis Graph

## Overview
The **SNC Analysis Graph** is a specialized component of the Adam v23 "Adaptive System". It utilizes the **Cyclical Reasoning** architecture to automate the regulatory classification of large, syndicated loans (Shared National Credits).

This system is designed to meet the high-impact need for robust, auditable credit analysis in the institutional finance market (specifically for the v22 remediation plan).

## Architecture
The graph is implemented using `langgraph` and consists of the following stateful nodes:

1.  **Analyze Structure**: Evaluates the syndicate composition (Lead Bank share, number of participants) to identify concentration or governance risks.
2.  **Assess Credit**: Performs quantitative analysis on the obligor's financials (Leverage, Liquidity, Coverage) to propose an initial regulatory rating (Pass, Special Mention, Substandard, Doubtful, Loss).
3.  **Critique**: A meta-cognitive step that reviews the proposed rating for logical consistency and compliance with the "Interagency Guidance on Leveraged Lending".
4.  **Revise**: Iteratively refines the analysis based on the critique until a robust conclusion is reached.
5.  **Human Approval**: A "Human-in-the-Loop" (HITL) checkpoint for adverse ratings (Substandard or worse).

## Usage

```python
from core.v23_graph_engine.snc_graph import snc_graph_app
from core.v23_graph_engine.states import init_snc_state

# 1. Define Input Data
obligor_id = "Titan Energy"
syndicate = {"banks": [{"name": "BigBank", "role": "Lead", "share": 0.6}]}
financials = {"ebitda": 350, "total_debt": 1800, "liquidity": 150}

# 2. Initialize State
initial_state = init_snc_state(obligor_id, syndicate, financials)

# 3. Run Graph
final_state = snc_graph_app.invoke(initial_state, config={"configurable": {"thread_id": "1"}})

print(final_state["regulatory_rating"])
print(final_state["rationale"])
```

## Artisanal Training Data
This module is supported by `data/artisanal_training_sets/artisanal_data_snc_v2.jsonl`, which provides high-quality "Few-Shot" examples for fine-tuning the underlying decision logic or for use in context stuffing.
