# Prompt Library

This file contains a collection of specialized, reusable prompts for different automated tasks within the financial intelligence system.

---

### Risk Analysis Prompt

**Objective:** To perform a contagion analysis for a given company.

**Prompt:**
"Given a company [Company Name], perform a contagion analysis. Identify all direct loan guarantees, key executives who are also on the boards of other portfolio companies, and major investors who also hold positions in other high-risk securities. Synthesize the top 3 contagion vectors."

---

### Entity Ingestion Prompt

**Objective:** To extract key entities and events from an SEC filing.

**Prompt:**
"You will be given the text of a new SEC 8-K filing. Read the text and extract the key entities (companies, people), events (e.g., acquisition, executive departure), and dates. Format the output as a JSON object ready to be validated and inserted into the knowledge graph."

---

### Executive Summary Prompt

**Objective:** To summarize the results of a complex graph query for a senior executive.

**Prompt:**
"You are given a JSON object containing the results of a complex graph query. Synthesize this data into a three-bullet-point summary for a senior risk officer. Focus on the most critical findings and required actions."
