# Odyssey Knowledge Graph Upgrade: FIBO Integration

## Overview
This upgrade formalizes the "Odyssey" Credit Risk System by integrating it with a Financial Industry Business Ontology (FIBO) based Knowledge Graph. This ensures that risk assessments are grounded in structured, verifiable relationships between Legal Entities, Debt Instruments, and Covenants.

## 1. Schema Definition
The new schema is defined in `data/fibo_knowledge_graph_schema.json` and introduces the following node types:

*   **LegalEntity (FIBO: LegalPerson)**: The borrower (e.g., "Apple Inc.").
*   **CreditFacility (FIBO: Loan)**: The debt structure (e.g., "General Facility").
*   **Tranche (FIBO: Tranche)**: Specific debt portions (e.g., "Term Loan B").
*   **Covenant (FIBO: Covenant)**: Restrictions on the borrower (e.g., "Max Leverage < 4.0x").
*   **FinancialReport (FIBO: FinancialReport)**: Snapshots of balance sheet/income statement data.
*   **RiskModel**: The output of the Odyssey analysis (Recommendation, Confidence).

## 2. Integration Logic
The `UnifiedKnowledgeGraph` class in `core/v23_graph_engine/unified_knowledge_graph.py` has been updated with a new method:

```python
def ingest_risk_state(self, risk_state: Dict[str, Any]):
    ...
```

### Mapping Process
1.  **Ticker -> LegalEntity**: The system checks or creates a node for the borrower.
2.  **VerticalRiskGraphState -> FinancialReport**: Extracted Balance Sheet and Income Statement data are converted into a time-stamped report node.
    *   *Derived Metrics*: Leverage Ratio (Total Debt / EBITDA) is calculated automatically during ingestion.
3.  **Covenants -> Covenant Nodes**: Legal constraints are mapped to nodes and linked to the Facility via `GOVERNED_BY`.
4.  **Draft Memo -> RiskModel**: The final analyst output is stored as a node linked via `HAS_RISK_MODEL`.

## 3. Benefits
*   **Auditability**: Every risk rating can be traced back to the specific Financial Report and Covenant nodes in the graph.
*   **Contagion Analysis**: By linking Entities via Supply Chains (existing UKG) and Credit Facilities (Odyssey), we can trace how a default in one entity affects others.
*   **Reasoning**: The graph structure allows for neuro-symbolic reasoning (e.g., "Find all companies violating 'Max Leverage' covenants").

## 4. Usage
To ingest data from the Vertical Risk Agent:

```python
from core.v23_graph_engine.unified_knowledge_graph import UnifiedKnowledgeGraph
from core.vertical_risk_agent.state import VerticalRiskGraphState

ukg = UnifiedKnowledgeGraph()
risk_state: VerticalRiskGraphState = { ... } # Populated by agent
ukg.ingest_risk_state(risk_state)
```
