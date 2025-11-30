# Data Directory

This directory contains a wide variety of data files that are essential for the operation of the ADAM system. These files include knowledge bases, knowledge graphs, decision trees, ontologies, and various other datasets used for training, testing, and analysis. This document provides a comprehensive overview of each file, its purpose, and how it can be used to supercharge development, navigation, integration, and modularity.

## 1. Knowledge Base and Knowledge Graph

The knowledge base and knowledge graph are the heart of the ADAM system's knowledge management capabilities. They provide a structured and machine-readable representation of the world, which allows agents to reason about complex concepts and relationships.

### 1.1. `knowledge_base.json` and `knowledge_base_v2.json`

*   **Purpose:** These files contain the core knowledge base of the ADAM system. They define a wide range of concepts and relationships in the financial domain, including valuation methods, risk management techniques, macroeconomic indicators, and technical analysis.
*   **Schema:** The knowledge base is organized into a hierarchical structure, with each entry containing a `machine_readable` section with formulas and parameters, and a `human_readable` section with definitions, explanations, and examples.
*   **Usage:** Agents use the knowledge base to understand and reason about financial concepts. For example, the `fundamental_analyst_agent` uses the knowledge base to understand how to perform a discounted cash flow (DCF) analysis, while the `risk_assessment_agent` uses it to understand how to calculate Value at Risk (VaR).
*   **Developer Notes:** When adding new concepts to the knowledge base, it is important to follow the existing schema and to provide both machine-readable and human-readable definitions. This will ensure that the new concepts can be easily understood and used by both agents and developers.
*   **Future Ideas:** The knowledge base could be extended to include more domains, such as legal and regulatory compliance. It could also be integrated with external knowledge bases, such as DBpedia and Wikidata, to provide a more comprehensive view of the world.

### 1.2. `knowledge_graph.json`, `knowledge_graph_v2.json`, and `knowledgegraph.ttl`

*   **Purpose:** These files contain the knowledge graph of the ADAM system. The knowledge graph is a network of interconnected entities, such as companies, people, and products. It allows agents to discover and explore relationships between entities, which can be used to generate insights and to make more informed decisions.
*   **Schema:** The knowledge graph is represented in a variety of formats, including JSON and Turtle (TTL). The JSON format is easy to parse and use in Python, while the TTL format is a standard for representing RDF data and can be used with a variety of graph databases and tools.
*   **Usage:** Agents use the knowledge graph to explore relationships between entities. For example, the `geopolitical_risk_agent` could use the knowledge graph to identify companies that are exposed to geopolitical risks, while the `supply_chain_risk_agent` could use it to identify companies that are dependent on a single supplier.
*   **Developer Notes:** When adding new entities to the knowledge graph, it is important to link them to existing entities whenever possible. This will create a more connected and valuable knowledge graph.
*   **Future Ideas:** The knowledge graph could be used to build a recommendation engine, a social network analysis tool, or a fraud detection system.

## 2. Decision Trees

Decision trees are used by agents to make decisions in a structured and transparent way. They provide a clear and auditable trail of the decision-making process, which is important for regulatory compliance and for building trust with users.

### 2.1. `credit_rating_decision_tree_v2.json` and `credit_rating_decision_tree_v3.json`

*   **Purpose:** These files contain decision trees for assessing the creditworthiness of companies and for assigning credit ratings.
*   **Schema:** The decision trees are represented in a JSON format, with each node in the tree representing a decision or a factor to consider.
*   **Usage:** The `snc_analyst_agent` uses these decision trees to assess the creditworthiness of companies. The agent traverses the tree, answering questions at each node, until it reaches a leaf node that contains the credit rating.
*   **Developer Notes:** When creating new decision trees, it is important to ensure that they are well-structured and that the decision logic is sound. It is also important to provide a clear and concise explanation of the decision-making process at each node.
*   **Future Ideas:** Decision trees could be used for a variety of other tasks, such as fraud detection, loan underwriting, and portfolio management.

## 3. Ontologies and Schemas

Ontologies and schemas provide a way to define the semantic context of the data in the ADAM system. They allow agents to understand the meaning of the data and to reason about it in a more intelligent way.

### 3.1. `context_definition.jsonld` and `CACM:SaaS_DefaultRisk_v1.jsonld`

*   **Purpose:** These files contain the ontologies and schemas for the ADAM system. They define the concepts, properties, and relationships that are used to represent the data in the system.
*   **Schema:** The ontologies and schemas are represented in JSON-LD format, which is a standard for representing linked data in JSON.
*   **Usage:** Agents use the ontologies and schemas to understand the meaning of the data. For example, an agent could use the ontology to understand that "revenue" is a type of "financial metric" and that it is measured in "millions of dollars."
*   **Developer Notes:** When creating new ontologies and schemas, it is important to follow the existing standards and to reuse existing vocabularies whenever possible. This will ensure that the new ontologies and schemas are interoperable with other systems.
*   **Future Ideas:** The ontologies and schemas could be used to build a semantic search engine, a data validation tool, or a data integration pipeline.

## 4. Core System and Market Data

These files provide the core data that the ADAM system needs to operate, including user information, market data, and economic indicators.

### 4.1. `adam_core_data.json`

*   **Purpose:** This file contains core data for the ADAM system, including user profiles, world events, economic indicators, and predictive models.
*   **Usage:** This data is used to provide context for the agents and to help them make more informed decisions.
*   **Schema:**
    *   `contextual_data`: Contains user profiles, world events, knowledge graph, and industry data.
    *   `predictive_models`: Contains information about the predictive models used by the system.
    *   `real_time_data_feeds`: Contains the URLs for real-time data feeds.
    *   `system_configuration`: Contains the system's configuration settings.
*   **Developer Notes:** This file is a central repository for the system's core data. It is important to keep this file up-to-date and to ensure that the data is accurate.

### 4.2. `adam_market_baseline.json`

*   **Purpose:** This file contains a baseline of market data that can be used for simulations and for training machine learning models.
*   **Usage:** This data is used to create a realistic market environment for testing and development.
*   **Schema:**
    *   `market_baseline`: Contains the version of the baseline, simulation metadata, and data modules.
    *   `data_modules`: Contains global economic indicators, asset classes, trading strategies, loan asset valuation, and machine learning data.
*   **Developer Notes:** This file can be extended by adding new data modules and by updating the existing ones with more realistic data.

## 5. Financial Analysis Templates

These files provide templates for financial analysis and valuation.

### 5.1. `clo_analyzer.csv`

*   **Purpose:** This file contains a template for analyzing collateralized loan obligations (CLOs).
*   **Usage:** Agents can use this template to analyze the performance of CLOs and to assess their risk.
*   **Schema:** The file is a CSV file with columns for CLO tranches, tranche size, tranche coupon, underlying assets, loan details, default assumptions, recovery rate, current interest rate, loan cash flows, CLO tranche cash flows, tranche pricing, CLO valuation, CDS pricing, mark-to-market valuation, and risk metrics.
*   **Developer Notes:** This template can be customized to meet the specific needs of a particular analysis.

### 5.2. `dcf_model_template.csv` and `dcf_valuation_template.json`

*   **Purpose:** These files contain templates for creating discounted cash flow (DCF) models.
*   **Usage:** Agents can use these templates to perform DCF analysis and to value companies.
*   **Schema:** The CSV file contains a template for a DCF model in a spreadsheet format, while the JSON file contains a more structured template that can be used by agents.
*   **Developer Notes:** These templates can be customized to meet the specific needs of a particular analysis.

### 5.3. `ev_model_template.csv`

*   **Purpose:** This file contains a template for creating enterprise value (EV) models.
*   **Usage:** Agents can use this template to calculate the enterprise value of a company.
*   **Schema:** The file is a CSV file with columns for assumptions, historical data, projections, and valuation.
*   **Developer Notes:** This template can be customized to meet the specific needs of a particular analysis.

### 5.4. `deal_template.json`

*   **Purpose:** This file contains a template for structuring and analyzing deals.
*   **Usage:** Agents can use this template to evaluate potential deals and to make recommendations.
*   **Schema:** The file is a JSON object with sections for deal name, deal date, company details, transaction details, financial projections, valuation analysis, risk assessment, deal summary, due diligence checklist, deal team, next steps, and deal notes.
*   **Developer Notes:** This template can be customized to meet the specific needs of a particular deal.

## 6. Company and User Data

These files contain data about companies and users.

### 6.1. `company_data.json`

*   **Purpose:** This file contains data about public companies.
*   **Usage:** This data is used by agents to perform fundamental analysis and to assess the creditworthiness of companies.
*   **Schema:** The file is a JSON object with a key for each company. Each company object contains information about the company's name, industry, financial statements, historical prices, competitors, growth rate, discount rate, tax rate, and terminal growth rate.
*   **Developer Notes:** This file can be extended by adding more companies and by updating the existing data with more recent information.

### 6.2. `private_company_template.json`

*   **Purpose:** This file contains a template for storing data about private companies.
*   **Usage:** This template can be used to create a database of private companies.
*   **Schema:** The file is a JSON object with sections for company name, LEI, private company profile, calculated metrics, assessment, integration points, module origin, version info, and timestamp.
*   **Developer Notes:** This template can be customized to meet the specific needs of a particular analysis.

### 6.3. `example_user_portfolio.json` and `example_user_profile.json`

*   **Purpose:** These files contain example user portfolios and profiles.
*   **Usage:** This data is used for testing and development purposes.
*   **Schema:** The portfolio file contains information about the portfolio's ID, owner ID, name, creation date, last updated date, description, currency, asset allocation, risk profile, investment horizon, holdings, performance metrics, future investments, and portfolio notes. The profile file contains information about the user's personal information, professional information, preferences, interaction history, personal goals, technology proficiency, social media profiles, health data, financial data, and custom filters.
*   **Developer Notes:** These files can be used as a starting point for creating more realistic user profiles and portfolios.

## 7. Risk Data

These files contain data about risk.

### 7.1. `global_risk_appetite_barometer_20250224.csv`

*   **Purpose:** This file contains data about global risk appetite.
*   **Usage:** This data is used by agents to assess the overall risk environment and to make more informed investment decisions.
*   **Schema:** The file is a CSV file with columns for region, risk appetite score, market volatility, economic indicators, geopolitical risk, social media sentiment, and Adam's Edge commentary.
*   **Developer Notes:** This file can be updated with more recent data to provide a more accurate picture of global risk appetite.

### 7.2. `risk_rating_mapping.json` and `risk_rating_mapping_v2.json`

*   **Purpose:** These files contain mappings between different risk rating systems.
*   **Usage:** This data is used by agents to compare and to translate between different risk rating systems.
*   **Schema:** The files are JSON objects with mappings for S&P, Moody's, and SNC credit ratings, as well as a risk score mapping.
*   **Developer Notes:** These files can be updated with new rating systems and with more granular mappings.

## 8. Simulated and Training Data

These files contain simulated data and teacher outputs for training and testing machine learning models.

### 8.1. `simulated_JSONL_output_4262025.jsonl` and `simulated_JSONL_output_52225_1042.jsonl`

*   **Purpose:** These files contain simulated JSONL output from the ADAM system.
*   **Usage:** This data is used for testing and for training machine learning models.
*   **Schema:** The files are in JSONL format, with each line containing a JSON object with information about a company, its credit rating, and the rationale for the rating.
*   **Developer Notes:** This data can be used to train a machine learning model to predict credit ratings.

### 8.2. `sp500_ai_overviews.jsonl`

*   **Purpose:** This file contains AI-generated overviews of the S&P 500 companies.
*   **Usage:** This data is used to train and to evaluate the performance of the natural language generation agents.
*   **Schema:** The file is in JSONL format, with each line containing a JSON object with information about a company, its GICS sector code, its GICS industry group code, its simulated revenue, its simulated year-over-year growth, its simulated EBITDA margin, its simulated leverage, its simulated S&P rating, and a report with negative news and red flags, a company overview, and a basic credit profile.
*   **Developer Notes:** This data can be used to train a natural language generation model to generate company overviews.

### 8.3. `teacher_outputs.jsonl`

*   **Purpose:** This file contains teacher outputs for training machine learning models.
*   **Usage:** This data is used to train the machine learning models in the ADAM system using supervised learning.
*   **Schema:** The file is in JSONL format, with each line containing a JSON object with input data, a teacher rating, a teacher justification, and teacher output probabilities.
*   **Developer Notes:** This data can be used to train a machine learning model to predict credit ratings and to generate justifications for the ratings.

By providing a comprehensive and well-documented data directory, we can empower developers to build more intelligent and capable agents, and to accelerate the development of the ADAM system as a whole.
