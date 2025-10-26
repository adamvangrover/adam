### 2.3. Knowledge Graph (FIBO) Extractor

* **ID:** `LIB-PRO-003`
* **Objective:** To parse unstructured financial or legal text and extract entities and relationships as machine-readable triples for a knowledge graph.
* **When to Use:** When processing legal documents (loan agreements, indentures) or financial reports (10-Ks) to programmatically populate a graph database (like Neo4j).
* **Key Placeholders:**
* `[Ontology_Standard]`: The target ontology (e.g., "FIBO," "custom schema").
* `[Output_Format]`: The target syntax (e.g., "Cypher," "SPARQL," "JSON-LD triples").
* `[Unstructured_Text]`: The source text to be parsed.
* **Pro-Tips for 'Adam' AI:** This is a **'UtilitySkill'** for your system. It can run in the background via your 'DataGatheringAgent', automatically parsing all new documents and ingesting the triples into your graph DB. This builds the "brain" that your 'TotalRecallAgent' (`LIB-META-005`) will query.

#### Full Template:

```
## ROLE: Financial Ontology Expert

Act as an expert parser specializing in the [Ontology_Standard] ontology.

## TASK:
I will provide a text. Parse it to extract all relevant entities (e.g., companies, people, dates, financial instruments) and their relationships.

Format the output *only* as a list of [Output_Format] triples (Subject, Predicate, Object).
- Use predicates from [Ontology_Standard] where possible (e.g., 'fibo-be:hasLegalEntityIdentifier', 'fibo-fbc:isPartyTo').
- For relationships not in the ontology, create a logical predicate (e.g., 'hasMaturityDate', 'hasLienOn').
- Do not add any commentary or explanation.

## TEXT:
[Unstructured_Text]
```
