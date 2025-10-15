# Knowledge Graph Schema Extension for Causal Inference

## 1. Objective

To support the integration of causal inference into the Adam system, the Knowledge Graph's schema must be extended. The current schema primarily supports correlational or associational relationships (e.g., `is_related_to`, `is_a_subsidiary_of`). To enable true causal reasoning, as recommended in the Causal Modeling Whitepaper, the schema must be updated to explicitly represent causal links between entities and events.

## 2. Lead Agent

*   **Knowledge Base Agent:** Responsible for managing and updating the Knowledge Graph, including the implementation and validation of the new schema.

## 3. Proposed Schema Extensions

We propose the introduction of a new set of directed, weighted edge types (relationships) to capture the nuances of causality. These new relationships will allow the system to build a Causal Bayesian Network directly from the Knowledge Graph, where the graph's nodes represent variables and the edges represent causal influence.

### 3.1. New Causal Relationship Types

The following new relationship types will be added to the schema:

*   **`causes`**:
    *   **Description:** A directed relationship indicating that Node A is a direct cause of Node B.
    *   **Example:** `(Subprime Mortgage Defaults) -[causes]-> (Lehman Brothers Bankruptcy)`
    *   **Attributes:**
        *   `strength` (float, 0.0 to 1.0): The probabilistic strength of the causal link, representing P(B|A).
        *   `time_lag` (integer, in days): The average time delay between the cause and the effect.
        *   `evidence_source` (string): The document, report, or analysis that supports this causal claim.

*   **`prevents`**:
    *   **Description:** A directed relationship indicating that the occurrence of Node A reduces the likelihood of Node B occurring. This is a form of negative causality.
    *   **Example:** `(Federal Reserve Intervention) -[prevents]-> (Systemic Financial Collapse)`
    *   **Attributes:**
        *   `strength` (float, 0.0 to 1.0): The strength of the preventative effect.
        *   `evidence_source` (string): The source supporting this claim.

*   **`enables`**:
    *   **Description:** A directed relationship indicating that Node A creates the conditions necessary for Node B to occur, but does not directly cause it. This represents a conditional dependency.
    *   **Example:** `(Deregulation of Financial Derivatives) -[enables]-> (Creation of Complex CDOs)`
    *   **Attributes:**
        *   `condition` (string): A description of the condition that is enabled.
        *   `evidence_source` (string): The source supporting this claim.

### 3.2. Deprecation of Ambiguous Relationships

The existing generic relationship `is_related_to` should be progressively phased out where a more specific causal link can be established. While it will be kept for non-causal associations, a dedicated effort will be made by the Knowledge Base agent to re-classify existing relationships into the new causal categories where appropriate.

## 4. Implementation and Data Ingestion

*   **Schema Update:** The graph database schema (e.g., Neo4j, JanusGraph) will be updated to include these new edge types and their associated properties.
*   **NLP Agent Enhancement:** The Natural Language Processing (NLP) agents responsible for information extraction from documents will be retrained to identify and extract these specific causal phrases (e.g., "led to," "resulted in," "prevented," "made possible by").
*   **Integration with WSM v8.0:** The World Simulation Model (WSM) will be upgraded to v8.0, which will be designed to read and interpret these new causal relationships from the Knowledge Graph, using them to build its internal Bayesian Network for simulation and forecasting.

## 5. Validation and Audit

A validation process will be established where any new causal link added to the graph with a strength above a certain threshold (e.g., 0.8) must be flagged for review by a human subject-matter expert. This human-in-the-loop validation is critical to prevent the propagation of incorrect causal assumptions throughout the system.
