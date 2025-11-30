# LIB-PRO-003: Knowledge Graph Extractor

*   **ID:** `LIB-PRO-003`
*   **Version:** `1.1`
*   **Author:** Adam v22
*   **Objective:** To parse unstructured financial or legal text and extract entities, their properties, and their relationships as clean, machine-readable statements for a knowledge graph. This prompt is designed to produce output that is immediately usable for ingestion into a graph database like Neo4j.
*   **When to Use:** When processing complex documents like loan agreements, bond indentures, 10-K filings, or M&A announcements to programmatically build a knowledge graph of corporate structures, obligations, and relationships.

---

### **Metadata & Configuration**

*   **Key Placeholders:**
    *   `[Output_Format]`: The target graph database syntax (e.g., "Cypher," "SPARQL," "JSON-LD triples").
    *   `[Schema_Definition]`: A clear definition of the desired entities and relationships, including their types and properties. This is the most critical input for ensuring structured output.
    *   `[Unstructured_Text]`: The source text to be parsed.
*   **Pro-Tips for 'Adam' AI Integration:**
    *   **Agent:** `KnowledgeGraphAgent` or `DataExtractionAgent`.
    *   **Background Processing:** This is a perfect task for a background agent. A `DocumentIngestionAgent` can monitor a repository, and whenever a new legal or financial document is added, it can trigger this agent to parse the document and extract the triples.
    *   **Database Integration:** The output of this prompt should be piped directly to a graph database client (e.g., a Neo4j Python driver) for execution. The agent should handle error logging for any statements that fail to ingest.
    *   **Schema Management:** The `[Schema_Definition]` can be stored as a separate configuration file, allowing you to easily update your knowledge graph's data model without changing the prompt.

---

### **Example Usage**

```
[Output_Format]: "Cypher"
[Schema_Definition]: "
- **Entities:**
  - `Company`: {name: string, ticker: string}
  - `Person`: {name: string, title: string}
  - `DebtInstrument`: {type: string, amount: float, currency: string, maturity_date: date}
  - `Covenant`: {type: string, value: float, metric: string}
- **Relationships:**
  - `(Company)-[:HAS_OFFICER]->(Person)`
  - `(Company)-[:IS_ISSUER_OF]->(DebtInstrument)`
  - `(DebtInstrument)-[:HAS_COVENANT]->(Covenant)`
  - `(Company)-[:HAS_PARENT]->(Company)`
"
[Unstructured_Text]: "[Pasted text from a loan agreement: 'This Senior Unsecured Note in the amount of $500 million USD is issued by Acme Corp. The note matures on 2030-12-31. Acme Corp. is a subsidiary of Global Holdings Inc. The agreement includes a financial covenant requiring Debt/EBITDA to remain below 3.5x. The CEO of Acme Corp. is Jane Doe.']"
```

---

## **Full Prompt Template**

```markdown
# ROLE: Financial Knowledge Graph Extractor

# CONTEXT:
You are an expert parser specializing in financial and legal ontologies. Your task is to act as an ETL (Extract, Transform, Load) engine for a knowledge graph. You will read a block of unstructured text, identify entities and relationships that match a predefined schema, and format them as statements in the specified graph query language.

# SCHEMA DEFINITION:
You must strictly adhere to the following schema for entities and relationships. Do not extract any information that does not fit this model.
---
[Schema_Definition]
---

# TASK:
I will provide a text. Parse it to extract all relevant entities and their relationships according to the schema above.

1.  **Entity Extraction:**
    *   First, identify all entities in the text that match the types defined in the schema.
    *   Extract their properties (e.g., name, amount, date).

2.  **Relationship Extraction:**
    *   Identify the relationships between the entities you extracted.
    *   The relationships must match the types defined in the schema.

3.  **Code Generation:**
    *   Generate a list of executable statements in **[Output_Format]** to create the entities and relationships in a graph database.
    *   Use `MERGE` for entities to avoid creating duplicates. Use `MERGE` for relationships where appropriate to ensure idempotency.

# CONSTRAINTS:
*   **Strict Adherence to Schema:** Do not create any entity types, property keys, or relationship types that are not explicitly defined in the `[Schema_Definition]`.
*   **No Commentary:** The output must *only* be the list of executable `[Output_Format]` statements. Do not add any commentary, explanations, or introductory text.
*   **Handle Missing Data:** If a property is not present in the text (e.g., a company's ticker), omit it from the `CREATE` or `MERGE` statement. Do not invent data.
*   **Clean Output:** Ensure data types are correct (e.g., numbers are not quoted as strings, dates are formatted as YYYY-MM-DD).

# TEXT TO PARSE:
---
[Unstructured_Text]
---

# OUTPUT:
```[Output_Format]
[Your generated statements here]
```
```
