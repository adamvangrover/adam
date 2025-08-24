# Financial Digital Twin (FDT) Prompt Library

This repository contains a portable and modular prompt library created from a strategic blueprint for a Financial Digital Twin (FDT). The library breaks down the complex concepts of the FDT into a series of distinct, reusable prompts that can be used to generate, explain, or expand upon the core components of the FDT strategy.

## Library Structure

The library is organized into the following directories, each corresponding to a key area of the FDT blueprint:

-   **`executive_summary_and_strategy/`**: Prompts focusing on the high-level vision, strategic positioning, and business value of the FDT.
-   **`semantic_foundation_and_data_modeling/`**: Prompts related to creating the "common language" for the FDT using ontologies and knowledge graphs.
-   **`architecture_and_technology_stack/`**: Prompts designed to generate detailed technical specifications for the FDT platform.
-   **`ai_agents_and_analytics/`**: Prompts for the intelligent layer of the FDT, including AI agents, machine learning, and natural language interfaces.
-   **`roadmap_and_governance/`**: Prompts covering the practical implementation, governance, security, and success measurement of the FDT initiative.

Each directory contains individual markdown files, with each file representing a specific prompt. The files are numbered to maintain a logical order consistent with the original blueprint.

## Usage

These prompts can be used with large language models (LLMs) to generate detailed documentation, specifications, and explanations related to the Financial Digital Twin. They are designed to be self-contained and provide sufficient context for an LLM to produce a high-quality response.

### **Prompt 1: Generate the Executive Summary**

"Draft a concise executive summary for a strategic blueprint on implementing a **Financial Digital Twin (FDT)** for lending operations. The summary must cover:
1.  **The Problem:** The challenges of the modern financial landscape (volatility, competition) and the limitations of traditional, siloed data systems in lending.
2.  **The Solution:** The vision of the FDT as a living, virtual replica of the lending ecosystem, moving the institution from reactive reporting to predictive foresight.
3.  **Core Capabilities:** Mention real-time simulation, predictive risk analysis, automated compliance, and hyper-personalized products.
4.  **The Architecture:** Briefly describe the hybrid architecture centered on a knowledge graph and powered by an agentic framework.
5.  **The Roadmap & ROI:** Reference a phased, three-year implementation plan and state the projected business outcomes, such as a 10-15% reduction in credit losses, 80% automation of regulatory reporting, and a 5% increase in loan origination."


### **Prompt 2: Explain the Strategic Imperative**

"Explain the strategic imperative for a financial institution to transition from a traditional lending operation to an **'Intelligent Lending Ecosystem.'** Your explanation should:
1.  Describe the evolving risk landscape, including market volatility, geopolitical risks, and sophisticated fraud.
2.  Highlight the competitive pressures from agile FinTech companies.
3.  Critique the traditional, siloed approach to data management (LOS, servicing, risk systems) and explain how it leads to a reactive, backward-looking risk posture."


### **Prompt 3: Compare Legacy Data Architectures**

"Compare and contrast the **Data Warehouse** and the **Data Lake** in the context of modern financial lending operations. For each architecture, explain:
1.  Its core paradigm (e.g., 'schema-on-write' vs. 'schema-on-read').
2.  Its primary strengths and weaknesses for handling diverse financial data (structured, unstructured).
3.  Why both are ultimately inadequate for the FDT's vision of a real-time, predictive, and simulation-ready platform."


### **Prompt 4: Define the FDT Vision and Business Alignment**

"Articulate the vision for the **Financial Digital Twin (FDT)**, emphasizing the paradigm shift from **'Hindsight to Foresight.'**
1.  Define the FDT as a living, dynamic, computable model of the entire lending portfolio.
2.  Detail its three core value propositions: **Holistic Situational Awareness**, **Predictive and Prescriptive Analytics**, and **Intelligent Automation**.
3.  Align these capabilities with four core business objectives: Enhanced Risk Management, Improved Operational Efficiency, Accelerated Revenue Growth, and Robust Regulatory Compliance. Provide specific, measurable targets for each objective."


### **Prompt 5: Explain the Knowledge Graph Core**

"Explain why a **Knowledge Graph** is the ideal semantic core for a Financial Digital Twin, as opposed to a traditional relational database. Your explanation should cover:
1.  How knowledge graphs model data as a network of entities (nodes) and relationships (edges).
2.  The inefficiency of using complex `JOIN` operations in relational databases to model the interconnected nature of lending (borrower, loan, collateral, guarantor).
3.  The knowledge graph's native ability to perform rapid, multi-hop reasoning to uncover hidden risks and complex connections."


### **Prompt 6: Design a Proprietary Lending Ontology**

"Design a proprietary ontology extension for lending operations that builds upon the **Financial Industry Business Ontology (FIBO)**.
1.  State the purpose: to model concepts specific to our lending business not covered in the general FIBO standard.
2.  Outline the methodical development process: identify core concepts, define properties, and link them to the FIBO hierarchy.
3.  Provide a concrete code example in Turtle (`.ttl`) format that defines a `lending:LoanCovenant` class as a subclass of a relevant FIBO class and an object property `lending:violatesCovenant`."


### **Prompt 7: Compare Data Orchestration Tools**

"Create a comparative analysis of three data workflow orchestration tools for the FDT's integration fabric: **Apache Airflow, Dagster, and Prefect.**
1.  Structure the comparison as a markdown table with criteria such as: Core Paradigm (e.g., task-centric vs. asset-centric), Development Experience, Data Lineage support, and Local Testing.
2.  Based on the analysis, provide a clear recommendation for the FDT project, justifying the choice. Emphasize alignment with the FDT's strategic goals (e.g., why Dagster's asset-centric model is a good fit)."


### **Prompt 8: Design the Converged Data Platform**

"Design the architecture for the FDT's **converged data platform**, which combines a data lakehouse with a specialized serving layer.
1.  Describe the **Foundation Layer**: A data lakehouse (e.g., Databricks on S3/ADLS) and its role as the cost-effective, comprehensive system of record with ACID compliance.
2.  Describe the **Serving Layer** and its 'polyglot persistence' approach. Detail the purpose of each specialized database:
    * **Graph Database (e.g., Neo4j):** For the core FDT knowledge graph and relationship analysis.
    * **Time-Series Database (e.g., TimescaleDB):** For high-frequency market data.
    * **Search Index (e.g., OpenSearch):** For unstructured text data like documents and news."


### **Prompt 9: Compare Enterprise Graph Databases**

"Provide a detailed comparison of leading enterprise graph databases: **Neo4j, TigerGraph, and Amazon Neptune.**
1.  Present the comparison in a markdown table, evaluating features like Data Model, Query Language, Scalability Model, Native Graph Data Science capabilities, and Security.
2.  Conclude with a specific recommendation for the FDT, justifying the choice based on factors like ecosystem maturity, query language intuition, and the comprehensiveness of its data science library."


### **Prompt 10: Generate a Risk Analysis Cypher Query**

"You are a risk analyst using the Financial Digital Twin. Write a **Cypher query** for the Neo4j graph database to perform a multi-hop counterparty risk assessment.
**Scenario:** Assess the total exposure to a borrower named 'Global Megacorp.'
The query must:
1.  Find the borrower 'Global Megacorp.'
2.  Identify all guarantors connected to this borrower.
3.  Find any publicly traded stock owned by those guarantors.
4.  Aggregate the borrower's direct exposure from its active loans.
5.  Return a summary of the borrower's name, its total direct exposure, a list of its guarantors, and the stock symbols it's indirectly exposed to via those guarantors."


### **Prompt 11: Design the Agentic Framework**

"Design a **multi-agent system** to power the FDT's intelligent automation.
1.  Explain the shift from monolithic applications to a collaborative system of autonomous AI agents.
2.  Define the core **agent personas** and their specific responsibilities:
    * **Credit Risk Agent:** Monitors portfolio credit quality.
    * **Fraud Detection Agent:** Uses GNNs to find fraud rings.
    * **Compliance Agent:** Screens against watchlists and monitors for SAR triggers.
    * **Market Intelligence Agent:** Analyzes unstructured news and market data.
    * **Query Agent:** Provides the natural language interface.
3.  Describe how these agents would collaborate using a 'Supervisor' agent pattern to answer a complex user query, such as: 'Show me our highest-risk loans exposed to the recent downturn in commercial real estate.'"


### **Prompt 12: Explain the Text-to-Cypher Engine**

"Explain the architecture and workflow of a **Text-to-Cypher** engine that serves as the FDT's natural language interface.
The process should include:
1.  **Schema-Aware Prompting:** How the system provides the graph's schema to an LLM as context.
2.  **Few-Shot Learning:** How example question/query pairs are used to improve accuracy.
3.  **LLM-Powered Translation:** The role of the LLM (e.g., GPT-4o) in generating the Cypher query.
4.  **Secure Execution:** The critical step of executing the generated query in the database (not the LLM) to enforce user permissions.
5.  **Synthesized Response:** How the LLM synthesizes the structured data from the query result into a human-readable answer."


### **Prompt 13: Propose Advanced Analytics Capabilities**

"Outline a strategy for implementing advanced analytics and simulation in the FDT, moving from prediction to causation.
1.  **Predictive Analytics:** Describe the use of **Graph Neural Networks (GNNs)** for sophisticated fraud detection, explaining how they learn from network topology to identify coordinated fraud rings.
2.  **Causal Inference:** Explain how **Causal Inference** and Causal DAGs will be used for true 'what-if' counterfactual portfolio simulation, allowing analysts to understand *why* events happen, not just predict that they will.
3.  **Explainable AI (XAI):** Detail the necessity of XAI techniques (like GNNExplainer) to ensure model transparency for business users, auditors, and regulators."


### **Prompt 14: Create the Phased Implementation Roadmap**

"Create a three-year, phased implementation roadmap for the Financial Digital Twin. Present this as a markdown table with the following columns: **Phase**, **Timeline**, **Key Objectives**, **Core Activities**, **Key Deliverables**, and **Success Metrics (KPIs)**.
* **Phase 1 (Year 1): Foundational Layer & Core Use Case.** Focus on setting up the infrastructure, ontology, and delivering a consolidated counterparty exposure report.
* **Phase 2 (Year 2): Advanced Analytics & Agentic Capabilities.** Focus on deploying GNN models, AI agents, and a natural language interface pilot.
* **Phase 3 (Year 3): Enterprise Expansion & Causal Simulation.** Focus on full enterprise rollout and introducing causal inference for portfolio simulation."


### **Prompt 15: Design the Governance & Security Framework**

"Outline a comprehensive **governance, security, and compliance framework** for the FDT.
1.  **Data Governance:** Define the key roles (Data Owners, Data Stewards) and the function of a Data Governance Council. Mention the importance of automated data lineage.
2.  **BCBS 239 Compliance:** Explain how the FDT's architecture directly addresses key principles of BCBS 239 (e.g., Completeness, Timeliness, Adaptability).
3.  **Graph-Native Security:** Describe a multi-layered security model using Role-Based Access Control (RBAC) with fine-grained permissions at the node and property level in Neo4j. Provide a sample rule (e.g., a Loan Officer can see a loan but not the sensitive details of a sanctions screening result).
4.  **LLMOps:** Detail the strategies for managing the security of the Text-to-Cypher LLM, including prompt injection prevention and ensuring the LLM never directly accesses sensitive data."



