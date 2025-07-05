# Comprehensive AI Agent & Analysis Prompt Library

This library provides a structured set of prompts for performing corporate credit risk analysis and orchestrating advanced AI agent workflows, covering the entire lifecycle from data ingestion to secure, collaborative deployment.

---

# I. Foundational & Scoping Prompts

## Entity Profile
> *This object gathers fundamental identification and contextual data. The purpose of the analysis is paramount, as it dictates the focus and depth required. An analysis for a new bond issuance will concentrate on the company's forward-looking capacity to service the proposed debt, whereas an annual surveillance review will focus on performance relative to previous expectations and covenants.*

### Task: EP01
> Provide the full legal name of the entity being analyzed, its primary ticker symbol (if public), headquarters location, and the ultimate parent entity.
- **Expected Response:** JSON object with keys: 'legal_name', 'ticker', 'hq_location', 'ultimate_parent'.

### Task: EP02
> Clearly state the purpose and scope of this credit analysis. Is it for a new debt issuance, an annual surveillance, a management assessment, or another purpose?
- **Expected Response:** Narrative statement defining the specific goal and boundaries of the analysis.

---

## Analytical Framework Setup
> *This object establishes the methodological 'rules of engagement.' Credit analysis adheres to structured frameworks published by rating agencies like S&P, Moody's, and Fitch. This selection governs the entire analytical process, from financial adjustments to risk factor weighting.*

### Task: AF01
> Select the primary credit rating agency methodology to be used for this analysis (e.g., S&P Global Ratings, Moody's, Fitch Ratings). Justify the selection.
- **Expected Response:** String value (e.g., 'S&P Global Ratings') with a brief narrative justification.

### Task: AF02
> Define the time horizon for the analysis, specifying the historical period (e.g., 2022-2024) and the forecast period (e.g., 2025-2027).
- **Expected Response:** JSON object with keys: 'historical_period_start', 'historical_period_end', 'forecast_period_start', 'forecast_period_end'.

---

## Information Gathering
> *This object serves as a structured checklist to ensure all necessary documentation is available before substantive analysis begins. The process mirrors the initial steps taken by rating agencies, who require issuers to provide a comprehensive information package. An analysis conducted with incomplete data, such as missing debt indentures, cannot properly assess structural risks and is inherently flawed.*

### Task: IG01
> Confirm receipt and list the annual and interim financial statements (10-K, 10-Q, or equivalents) for the defined historical period.
- **Expected Response:** Boolean confirmation with a list of documents received.

### Task: IG02
> Confirm receipt and list key legal and financing documents, including credit agreements, bond indentures, and major lease agreements.
- **Expected Response:** Boolean confirmation with a list of documents received.

### Task: IG03
> Confirm receipt and list qualitative documents, such as investor presentations, management discussion and analysis (MD&A), and equity research reports.
- **Expected Response:** Boolean confirmation with a list of documents received.

---

# II. Macro-Environment Risk Assessment

## Sovereign and Country Risk
> *This analysis evaluates the risks stemming from the primary countries where the company operates, generates revenue, and holds assets. For companies with significant foreign currency debt, the sovereign's own foreign currency rating can act as a 'sovereign ceiling,' effectively capping the corporate's rating due to transfer and convertibility risks.*

### Task: SCR01
> List the company's key countries of operation, ranked by percentage of revenue, assets, or EBITDA.
- **Expected Response:** A list of countries with corresponding percentages for revenue, assets, or EBITDA.

### Task: SCR02
> For the top 3 key countries, assess the economic risk, including real GDP growth trends, inflation, and currency volatility. Provide the sovereign credit rating for each.
- **Expected Response:** Narrative analysis supported by macroeconomic data and sovereign ratings.

### Task: SCR03
> For the top 3 key countries, assess the political and institutional risk, including political stability, rule of law, and institutional effectiveness.
- **Expected Response:** Qualitative narrative assessment.

### Task: SCR04
> Assess the risk of a 'sovereign ceiling' impacting the company's rating due to transfer and convertibility (T&C) risk. Does the company have significant foreign currency debt issued from a country with a low sovereign rating?
- **Expected Response:** Narrative assessment concluding with a statement on the level of sovereign ceiling risk (e.g., Low, Moderate, High).

---

## Industry Risk Analysis
> *This section evaluates the dynamics of the industry in which the company competes. The analysis must identify systemic risks and opportunities that affect all participants, such as cyclicality, competitive intensity, and long-term growth prospects. A critical modern component is the assessment of industry-wide Environmental, Social, and Governance (ESG) risks.*

### Task: IR01
> Define the company's primary industry and any significant sub-industries.
- **Expected Response:** String identifying the primary industry (e.g., 'Global Automotive Manufacturing').

### Task: IR02
> Analyze the industry's cyclicality, competitive intensity, and barriers to entry. How do these factors influence profitability and risk for participants?
- **Expected Response:** Narrative analysis covering cyclicality, competition, and barriers to entry.

### Task: IR03
> Assess the industry's long-term growth prospects and key drivers. Is the industry mature, in decline, or experiencing high growth? What are the primary demand drivers?
- **Expected Response:** Narrative analysis supported by industry growth data.

### Task: IR04
> Identify the top 3 systemic ESG-related risks and opportunities for this industry (e.g., carbon transition, water scarcity, data privacy, supply chain labor standards). Explain how these factors could impact the industry's long-term risk profile and profitability.
- **Expected Response:** Narrative identifying and explaining the impact of key industry-level ESG factors.

### Task: IR05
> Synthesize the country and industry risk assessments to determine a combined Corporate Industry and Country Risk Assessment (CICRA) score, following the selected rating agency's methodology. Justify how the interaction between country and industry factors exacerbates or mitigates overall risk.
- **Expected Response:** A single risk score (e.g., 1-Very Low Risk to 6-Very High Risk) with a detailed justification narrative.[11]

---

# III. Business Risk Profile Assessment

## Competitive Position
> *This evaluates the company's market standing and the sustainability of its competitive advantages. A dominant market share, protected by high barriers to entry, is a significant credit strength. Conversely, high customer or geographic concentration is a key vulnerability.*

### Task: CP01
> Assess the company's market share and competitive rank in its primary product lines and geographic markets. Is its position strengthening, stable, or eroding over time? Provide supporting data.
- **Expected Response:** Narrative analysis with market share data and trends.

### Task: CP02
> Analyze the company's diversification across products/services, geographies, and customers. Is there significant concentration risk in any of these areas? Quantify where possible (e.g., '% of revenue from top customer').
- **Expected Response:** Narrative analysis with supporting diversification metrics.

### Task: CP03
> Identify and evaluate the company's key competitive advantages (e.g., brand strength, proprietary technology, cost leadership, network effects, barriers to entry). How durable are these advantages?
- **Expected Response:** Qualitative assessment of competitive advantages with justification.

---

## Operational Efficiency and Profitability
> *This examines the company's ability to generate profits and cash flow. A crucial distinction is made between the absolute level of profitability and its volatility. Two companies may have the same average EBITDA margin over a five-year period, but the one with lower margin volatility is considered a better credit risk because its cash flows are more predictable and reliable for servicing debt through an economic cycle.*

### Task: OEP01
> Analyze the historical trend and level of the company's key profitability metrics (e.g., EBITDA margin, EBIT margin) over the defined historical period.
- **Expected Response:** Narrative analysis supported by a table of historical profitability ratios.

### Task: OEP02
> Assess the volatility of the company's profitability. Calculate the standard deviation or coefficient of variation of the EBITDA margin over the historical period and compare it to peers.
- **Expected Response:** A quantitative measure of volatility with a narrative explaining its credit implications.

### Task: OEP03
> Evaluate the company's cost structure and operating efficiency. Is there evidence of a durable cost advantage? How does its efficiency compare to peers?
- **Expected Response:** Qualitative assessment of the cost structure with supporting evidence.

---

## Management and Governance
> *This qualitative assessment evaluates the competence, strategy, and risk appetite of the management team, as well as the robustness of corporate governance structures. Management's financial policy is a critical indicator of future financial risk and demonstrates the link between business strategy and balance sheet management. Weak governance or a history of poor strategic execution are significant credit concerns.*

### Task: MG01
> Evaluate management's strategic competence and operational track record. Has management successfully executed on past strategic initiatives?
- **Expected Response:** Narrative assessment of management's strategy and historical performance.

### Task: MG02
> Assess management's risk appetite and financial policy. Is the financial policy viewed as conservative, moderate, or aggressive? Are shareholder returns consistently prioritized over creditor interests?
- **Expected Response:** Narrative assessment of financial policy, concluding with a characterization (e.g., 'Aggressive').

### Task: MG03
> Evaluate the quality and robustness of corporate governance. Consider board independence, transparency of financial reporting, and any history of related-party transactions or regulatory issues.
- **Expected Response:** Qualitative assessment of governance structures and practices.

---

## Group and Ownership Structure
> *This analysis considers the influence of the company's parent or controlling shareholders. A subsidiary's rating can be positively influenced by a strong parent or negatively impacted by a weak parent that may extract resources. The analysis must consider specific methodologies for group structures and government-related entities (GREs).*

### Task: GOS01
> Identify the company's parent entity or key controlling shareholders. Describe the ownership structure.
- **Expected Response:** Narrative description of the ownership structure.

### Task: GOS02
> Assess the potential for positive or negative intervention from the parent/controlling shareholder. Consider the parent's credit quality, strategic importance of the subsidiary, and any history of support or resource extraction.
- **Expected Response:** Narrative assessment concluding on the likely direction and strength of group influence.

### Task: GOS03
> If the company is a Government-Related Entity (GRE), assess the likelihood of extraordinary government support based on the relevant rating agency methodology.
- **Expected Response:** Narrative analysis applying the GRE framework, concluding on the likelihood of support.

---

# IV. Financial Risk Profile Assessment

## Financial Statement Adjustments
> *This is the most critical step in quantitative analysis. Standard adjustments for items like operating leases and pension deficits create an analytically 'clean' set of financials that provide a more accurate picture of a company's leverage and obligations.*

### Task: FSA01
> Calculate the present value of operating lease commitments and add the result to reported debt to arrive at lease-adjusted debt. Add lease-related interest back to reported EBITDA.
- **Expected Response:** Table showing reported debt, lease adjustment, and lease-adjusted debt. Separate calculation for adjusted EBITDA.

### Task: FSA02
> Calculate the after-tax pension and Other Post-Employment Benefit (OPEB) deficits and add them to reported debt.
- **Expected Response:** Table showing reported debt, pension/OPEB adjustment, and resulting adjusted debt.

### Task: FSA03
> Identify and quantify any material non-recurring items (e.g., restructuring costs, asset sale gains) from the historical period. Adjust reported EBITDA to reflect a normalized, ongoing earnings capacity.
- **Expected Response:** Table listing non-recurring items and their impact on reported EBITDA to arrive at adjusted EBITDA.

---

## Historical Financial Analysis
> *This involves calculating and interpreting key credit ratios over the historical period using the adjusted financial figures. The focus is on leverage, coverage, and cash flow metrics, which are central to assessing debt repayment capacity.*

### Task: HFA01
> Using the fully adjusted financials, calculate key leverage ratios (e.g., Adjusted Debt / Adjusted EBITDA, Adjusted FFO / Adjusted Debt) for the defined historical period.
- **Expected Response:** Table of historical leverage ratios.

### Task: HFA02
> Using the fully adjusted financials, calculate key coverage ratios (e.g., Adjusted EBITDA / Adjusted Interest Expense) for the defined historical period.
- **Expected Response:** Table of historical coverage ratios.

### Task: HFA03
> Analyze the historical trends in the calculated credit ratios. Explain the key drivers of any significant improvement or deterioration.
- **Expected Response:** Narrative analysis explaining the trends observed in the historical credit metrics.

---

## Cash Flow Analysis
> *A deeper dive into the composition, quality, and sustainability of a company's cash flow, which is often considered the single most important consideration in credit analysis. This includes analyzing working capital trends and the cash conversion cycle.*

### Task: CFA01
> Analyze the quality and composition of Cash Flow from Operations (CFO). How much is driven by non-cash charges versus core earnings? Is it volatile?
- **Expected Response:** Narrative analysis of CFO quality and stability.

### Task: CFA02
> Analyze historical working capital trends. Is the company experiencing a consistent cash drain or benefit from working capital changes? What does this imply about operational management?
- **Expected Response:** Narrative analysis supported by a table of historical working capital movements.

### Task: CFA03
> Calculate historical Free Operating Cash Flow (FOCF) and Discretionary Cash Flow (DCF). Assess the company's ability to generate cash after capital expenditures and dividends.
- **Expected Response:** Table showing historical calculation of FOCF and DCF with a narrative assessment.

---

## Financial Forecasting and Stress Testing
> *Credit ratings are inherently forward-looking opinions. This section moves from historical analysis to projecting future performance. A critical concept here is the development of a 'rating case' forecast. This is distinct from a company's often-optimistic 'management case.' The rating case incorporates more conservative assumptions about growth and profitability to assess debt service capacity 'through the cycle'.*

### Task: FFS01
> Develop a 'rating case' financial forecast for the defined forecast period. Clearly state the key assumptions for revenue growth, profitability margins, and capital expenditures. These assumptions should be more conservative than management's public guidance.
- **Expected Response:** A full projected financial statement model (IS, BS, CF) with a separate table listing and justifying key assumptions.

### Task: FFS02
> Define and apply a 'downside stress test' scenario to the rating case forecast. This should model a plausible negative event (e.g., recession, sharp input cost increase). State the stress assumptions clearly.
- **Expected Response:** A second set of projected financial statements under the stress scenario, with assumptions clearly defined.

### Task: FFS03
> Analyze the trajectory of key credit metrics (leverage, coverage) under both the rating case and the downside stress test. How resilient is the company's financial profile?
- **Expected Response:** Table comparing projected credit metrics under both scenarios, with a narrative discussing financial resilience.

---

## Financial Flexibility and Liquidity
> *This assesses the company's ability to meet near-term obligations and manage unexpected cash shortfalls. It involves analyzing the debt maturity profile, available liquidity sources, and covenant headroom under credit facilities. A potential covenant breach is a significant credit event that can trigger defaults.*

### Task: FFL01
> Analyze the company's near-term liquidity position. Calculate sources (cash, FFO, available credit lines) versus uses (short-term debt, working capital needs, capex, dividends) over the next 12-24 months.
- **Expected Response:** A sources and uses of liquidity table with a concluding statement on the adequacy of the liquidity position (e.g., Strong, Adequate, Weak).

### Task: FFL02
> Provide a schedule of the company's debt maturities for the next 5 years and beyond. Are there any large, upcoming maturity towers that pose a refinancing risk?
- **Expected Response:** A table of debt maturities by year, with a narrative assessment of refinancing risk.

### Task: FFL03
> Identify the key financial covenants in the company's main credit facilities. Calculate the current and projected covenant headroom under the rating case and stress case forecasts.
- **Expected Response:** Table listing key covenants, their required levels, and the calculated headroom (in %) under both forecast scenarios.

---

# V. Synthesis, Rating, and Reporting

## Peer Analysis
> *A company's credit metrics are only meaningful when placed in the context of its peers. This systematic comparison helps to normalize for industry-specific characteristics and highlights areas of relative strength or weakness.*

### Task: PA01
> Identify a group of 3-5 publicly rated peer companies. Justify their selection based on business mix, scale, and geography.
- **Expected Response:** List of peer companies with their credit ratings and a brief justification for their inclusion.

### Task: PA02
> Create a table comparing the subject company's business risk profile (market position, diversification, profitability) against the selected peers.
- **Expected Response:** Table with qualitative comparisons (e.g., 'Stronger', 'In-line', 'Weaker') for key business risk factors across the peer group.

### Task: PA03
> Create a table comparing the subject company's key historical and projected financial metrics (leverage, coverage) against the selected peers.
- **Expected Response:** Table with quantitative credit metrics for the subject company and its peers.

---

## Risk Profile Synthesis
> *This is where the two main pillars of the analysis—Business Risk and Financial Risk—are formally combined to derive an initial, or 'anchor,' credit assessment.*

### Task: RPS01
> Based on the preceding analysis (competitive position, diversification, profitability), synthesize and assign a single Business Risk Profile assessment (e.g., Excellent, Strong, Satisfactory, Fair, Weak, Vulnerable). Justify the assessment.
- **Expected Response:** A single adjectival score with a detailed justification narrative.

### Task: RPS02
> Based on the preceding analysis (historical and projected financial metrics), synthesize and assign a single Financial Risk Profile assessment (e.g., Minimal, Modest, Intermediate, Significant, Aggressive, Highly Leveraged). Justify the assessment.
- **Expected Response:** A single adjectival score with a detailed justification narrative.

### Task: RPS03
> Using the selected rating agency's Business Risk / Financial Risk matrix, combine the two profile assessments to determine the 'anchor' credit rating.
- **Expected Response:** A single rating category (e.g., 'bbb', 'bb+') derived from the matrix.

---

## Modifying Factors and Notching
> *The anchor rating is adjusted for other material factors. A particularly strong or weak liquidity profile can warrant an adjustment. For specific debt instruments, recovery analysis determines whether the instrument rating should be at, above, or below the issuer's overall credit rating based on its security and seniority in the capital structure.*

### Task: MFN01
> Assess the company's liquidity profile as a potential modifying factor. Does the liquidity position (Strong, Adequate, Weak) warrant a notch up or down from the anchor rating?
- **Expected Response:** Narrative assessment concluding with a notching decision (e.g., '+1 notch', 'no adjustment', '-1 notch').

### Task: MFN02
> Assess other potential modifiers, such as financial policy, governance, or group support. Justify any further notching adjustments to the anchor rating.
- **Expected Response:** Narrative assessment of any other modifiers and their impact on the rating.

### Task: MFN03
> For a specific debt instrument, conduct a recovery analysis to determine if its rating should be notched up or down from the final Issuer Credit Rating based on its collateral and seniority.
- **Expected Response:** A recovery rating (e.g., '1+', '3', '5') and a corresponding instrument rating.

---

## Rating Recommendation
> *This is the final, actionable output. It includes the recommended rating, a forward-looking outlook, and a concise rationale. The outlook (Stable, Positive, Negative) is a critical component, communicating the likely direction of the rating over the next 12-24 months and is based on the potential for identified risks or opportunities to materialize.*

### Task: RR01
> State the final recommended Issuer Credit Rating (ICR) after all adjustments.
- **Expected Response:** A final credit rating (e.g., 'BBB-').

### Task: RR02
> Assign a rating outlook (e.g., Stable, Positive, Negative, Developing). Justify the outlook based on the potential for specific risks or opportunities to materialize over the next 12-24 months.
- **Expected Response:** A rating outlook with a brief justification.

### Task: RR03
> Write a concise rating rationale (2-3 paragraphs) summarizing the key credit strengths and weaknesses that support the final rating and outlook.
- **Expected Response:** A well-structured narrative summarizing the core credit story.

---

## Credit Report Generation
> *This final object provides prompts to assemble the full narrative report from the preceding analytical components, ensuring a professional and comprehensive final deliverable consistent with industry standards.*

### Task: CRG01
> Assemble an executive summary that includes the final rating recommendation, outlook, and a high-level overview of the business and financial risk profiles and key credit considerations.
- **Expected Response:** A 1-page executive summary narrative.

### Task: CRG02
> Compile the full, detailed credit report by sequencing the narrative outputs from all preceding analytical sections in a logical, professional format.
- **Expected Response:** A single, comprehensive document containing the full analysis.

---

# VI. Meta-Instructions & Tooling

## System Behavior
> *Defines the LLM's core operational parameters, persona, and versioning for the task.*

### Task: SYS01
> Adopt the persona of a senior credit analyst with 15 years of experience at a major rating agency. All subsequent responses should be formal, data-driven, and reference established credit methodologies.
- **Expected Response:** Confirmation of persona adoption, e.g., 'Persona adopted. Ready to proceed with analysis.'

### Task: SYS02
> Set the master version for this entire analysis session to v1.0. All generated artifacts, data, and reports should be tagged with this version.
- **Expected Response:** Confirmation of version initialization, e.g., 'Analysis session v1.0 initialized.'

### Task: SYS03
> Generate a configuration file in YAML format for a new data processing job. The configuration should include 'source_bucket', 'destination_table', 'job_name', and a list of 'data_quality_checks' such as 'not_null' and 'unique_values'.
- **Expected Response:** A code block containing a valid YAML configuration file.

---

## Knowledge Integration (RAG)
> *Controls how the model interacts with and synthesizes information from external knowledge sources.*

### Task: RAG01
> For the next prompt, exclusively use the following documents from the knowledge base as your context for Retrieval-Augmented Generation (RAG): [list of document IDs, e.g., 'doc_10K_2024.pdf', 'doc_credit_agreement_2023.pdf']. Do not use general knowledge.
- **Expected Response:** Confirmation of RAG source configuration, e.g., 'RAG context locked to provided documents.'

---

## Knowledge Graph & Ontology
> *Controls how the model interacts with and synthesizes information from external knowledge sources.*

### Task: KGRAPH01
> Based on the 'entity_profile' and 'group_and_ownership_structure' sections, generate a formal ontology in OWL format defining the relationships between the company, its parent, its subsidiaries, and key executives.
- **Expected Response:** A code block containing the ontology in OWL (Web Ontology Language) format.

### Task: KGRAPH02
> Populate a knowledge graph using the ontology from KGRAPH01 and the information from the credit report. Represent entities and relationships as Cypher statements for import into a Neo4j database.
- **Expected Response:** A code block containing a series of Cypher `CREATE` or `MERGE` statements.

---

## Decision Tree Modeling
> *Prompts for generating and executing structured analytical models like decision trees.*

### Task: DT01
> Generate a decision tree in Python using scikit-learn that models the final rating recommendation. Use the following as features: CICRA score, competitive position assessment, profitability volatility, and projected Debt/EBITDA. The tree should output a rating category.
- **Expected Response:** A Python code snippet defining and training a `DecisionTreeClassifier`.

---

## Coding & Automation
> *Prompts for generating code, scripts, and instructions for development automation tools.*

### Task: CODE01
> Write a complete Python script named 'data_validator.py' that takes a CSV file path as a command-line argument. The script must use the pandas library to read the CSV and verify that the 'EBITDA' column contains no negative values. It should print a success or failure message.
- **Expected Response:** A complete, executable Python script within a single code block.

### Task: CODE02
> Generate the high-level pseudocode and comments for a Python function that will be completed by a coding copilot. The function, named 'calculate_dscr', should take 'net_operating_income' and 'total_debt_service' as inputs and return the Debt Service Coverage Ratio. Include type hints and a docstring.
- **Expected Response:** A Python function stub with detailed comments and pseudocode, ready for a copilot tool to complete.

---

## External Tooling (API & CLI)
> *Prompts for generating commands to interact with external tools like APIs and Command-Line Interfaces.*

### Task: API01
> Generate a Python script to call an external API at 'https://api.marketdata.com/v1/quotes' to retrieve the latest stock price for the company's ticker. The API key is stored in the environment variable 'MARKET_DATA_API_KEY'.
- **Expected Response:** A Python code snippet using the `requests` library to make the specified API call.

### Task: CLI01
> Generate the gcloud CLI command to download the latest financial reports for the company from the Google Cloud Storage bucket 'gs://financial-reports-archive/' into the local directory './reports'.
- **Expected Response:** A shell command snippet, e.g., `gcloud storage cp gs://financial-reports-archive/COMPANY_TICKER/* ./reports/`

### Task: CLI02
> Generate a single-line terminal command that finds all '.log' files in the '/var/log' directory, searches for lines containing the word 'ERROR', and saves the resulting lines to a file named 'error_summary.txt' in the home directory.
- **Expected Response:** A single, complete shell command utilizing pipes, e.g., `grep -r 'ERROR' /var/log/*.log > ~/error_summary.txt`.

---

## Agentic Workflow (SDK & MCP)
> *Prompts for defining and executing multi-step, agentic tasks, including coordination between multiple agents.*

### Task: AGENT01
> Initialize as a research agent. Your primary goal is to complete the 'Macro-Environment Risk Assessment' stage. You have access to the following tools: [web_search, knowledge_base_query]. Acknowledge when the goal is complete.
- **Expected Response:** Confirmation of agent initialization and goal understanding.

### Task: AGENT02
> Decompose the goal 'Complete the Financial Risk Profile Assessment' into a sequence of logical steps using the available prompts (FSA01, HFA01, etc.) and tools [code_interpreter, api_caller].
- **Expected Response:** A numbered list of steps or a plan in JSON format.

### Task: MCP01
> This is a multi-agent task. Define roles for 'AnalystAgent' (executes analysis prompts) and 'ReviewerAgent' (critiques outputs for quality and accuracy). The 'ReviewerAgent' must approve the output of each stage before the 'AnalystAgent' can proceed.
- **Expected Response:** A JSON object defining the roles, responsibilities, and interaction protocol for the agents.

---

## Prompt-to-Prompt / A2A
> *Prompts for defining and executing multi-step, agentic tasks, including coordination between multiple agents.*

### Task: A2A01
> Upon completion of the current prompt, analyze its output. If the 'leverage' metric is above 4.0x, the next prompt you must execute is FFL03 (Evaluate Covenant Headroom). Otherwise, the next prompt is PA01 (Select Peer Group). Formulate and output ONLY the JSON for the next prompt to be executed.
- **Expected Response:** A single, complete JSONL object representing the next prompt in the dynamic chain.

---

## Federated Operations
> *Prompts for designing and orchestrating federated learning or federated analytics workflows on decentralized data.*

### Task: FED01
> Design a high-level federated analysis plan to calculate the average leverage ratio across three different, isolated corporate groups (GroupA, GroupB, GroupC) without moving their raw financial data. The plan should specify the local computation on each node and the central aggregation step. Use TensorFlow Federated (TFF) constructs as a reference.
- **Expected Response:** A narrative or multi-step plan outlining the federated process, including `tff.federated_computation` and aggregation logic.

---

# VII. Advanced Analytics & Orchestration

## Simulation & Stochastic Modeling
> *Prompts for creating and running advanced simulations to model uncertainty and complex system dynamics.*

### Task: SIM01
> Using the 'rating case' forecast as a baseline, generate a Python script to run a 10,000-iteration Monte Carlo simulation on the company's Free Cash Flow. Assume EBITDA margin and revenue growth are normally distributed with means equal to the baseline forecast and standard deviations derived from historical data. The output should be a histogram of potential cash flow outcomes and the probability of cash flow being negative.
- **Expected Response:** A Python script using libraries like NumPy and Matplotlib to perform the Monte Carlo simulation and generate visualizations.

### Task: SIM02
> Generate a causal loop diagram in DOT language (for Graphviz) that models the key feedback loops in the company's business. Include nodes for 'Capital Investment', 'Asset Base', 'Revenue Capacity', 'Profitability', and 'Cash Flow for Reinvestment'. Indicate reinforcing ('R') and balancing ('B') loops.
- **Expected Response:** A code block containing a system dynamics model in DOT language.

---

## Explainability & Audit (XAI)
> *Prompts for interpreting model behavior, ensuring transparency, and creating audit trails.*

### Task: XAI01
> For the decision tree model created in task DT01, generate a Python script using the SHAP (SHapley Additive exPlanations) library to explain the prediction for a specific hypothetical company with high debt and strong profitability. The output should be a SHAP force plot visualizing the feature contributions.
- **Expected Response:** A Python script that loads the model, creates a sample data point, and generates a SHAP force plot to explain its prediction.

### Task: XAI02
> Generate a counterfactual analysis narrative. Based on the final model, determine the minimum improvement in 'Projected Debt/EBITDA' and the minimum change in 'Competitive Position Assessment' that would have been required to achieve a one-notch rating upgrade.
- **Expected Response:** A narrative analysis explaining the specific changes in key variables required to alter the outcome.

### Task: XAI03
> Review the full log of prompts and responses in this session and generate a Markdown summary for an audit trail. The summary must list each major analytical step, the data or documents used, and the resulting conclusion, providing a traceable path from raw data to final rating.
- **Expected Response:** A structured Markdown document detailing the analytical process flow for audit purposes.

---

## Advanced Knowledge Synthesis
> *Prompts for advanced reasoning, hypothesis testing, and synthesis across multiple data types and sources.*

### Task: AKS01
> Form a hypothesis based on the initial financial data. Then, generate a multi-step plan to validate or refute this hypothesis. The plan must explicitly state which documents to search (using RAG), which API calls to make, and which calculations to perform. Hypothesis Example: 'The company's declining gross margins are primarily caused by rising input costs rather than pricing pressure.'
- **Expected Response:** A JSON object with a 'hypothesis' string and a 'validation_plan' array of steps.

### Task: AKS02
> Synthesize information from the following multimodal sources to assess the company's brand strength: [Text: MD&A section on market position, Image: Chart of market share from investor deck, Table: Customer satisfaction scores from attached CSV]. Conclude with a qualitative assessment.
- **Expected Response:** A narrative synthesis that explicitly references how the text, image, and tabular data collectively support the final conclusion on brand strength.

---

## Workflow Orchestration
> *Prompts for automating and orchestrating complex workflows, including defining human-in-the-loop checkpoints.*

### Task: ORC01
> Generate a GitHub Actions workflow file (`.github/workflows/analysis.yml`) that triggers on a push to the 'main' branch. The workflow should execute the entire chain of analysis scripts (e.g., `data_validator.py`, `monte_carlo.py`) in sequence.
- **Expected Response:** A complete, valid YAML file for a GitHub Actions workflow.

### Task: ORC02
> Define a human-in-the-loop (HITL) checkpoint. After the 'anchor' rating is determined in task RPS03, the workflow must pause. Generate a request for human review containing the anchor rating and the business/financial risk profiles. The agent may only proceed to MFN01 after receiving an explicit 'approved' signal.
- **Expected Response:** A JSON object defining the HITL trigger condition, the data payload for the human reviewer, and the required approval signal.

---

## Dynamic Reporting & Communication
> *Prompts for creating interactive, audience-specific reports and presentations.*

### Task: DYN01
> Generate a Python script for an interactive dashboard using Streamlit or Plotly Dash. The dashboard must allow a user to select different stress test scenarios (e.g., 'Recession', 'Input Cost Shock') from a dropdown menu and see the projected leverage and coverage ratios update dynamically in a chart.
- **Expected Response:** A complete, executable Python script for an interactive dashboard.

### Task: DYN02
> Generate three distinct summaries of the final rating recommendation (RR03): 1) An 'Executive Summary' for the CEO (max 100 words, focuses on strategic implications). 2) A 'Portfolio Manager Briefing' (focuses on risk factors, outlook, and covenant details). 3) A 'Methodology Note' for a junior analyst (explains key adjustments and model choices).
- **Expected Response:** A JSON object with three keys ('executive_summary', 'pm_briefing', 'methodology_note'), each containing the tailored narrative.

### Task: DYN03
> Generate a JSON structure representing a slide deck for the rating committee presentation. The structure should include keys for 'title', 'presenter', and an array of 'slides'. Each slide object should have a 'title', 'talking_points' (an array of strings), and an optional 'visualization_type' (e.g., 'bar_chart', 'line_chart'). Populate it with the key findings of this analysis.
- **Expected Response:** A structured JSON object representing the entire presentation.

---

# VIII. Model Governance, Deployment & Lifecycle Management

## Model & Artifact Versioning
> *Prompts for managing versions of models, data, and code using integrated version control systems.*

### Task: GOV01
> Generate the git commands to create a new branch named 'feature/rating_model_v2', add the serialized decision tree model file ('rating_model.pkl') and the final credit report ('credit_report_v1.pdf') to the branch, commit them with the message 'feat: Add version 2 of rating model and initial report', and push the branch to the remote repository.
- **Expected Response:** A sequence of shell commands for git.

### Task: GOV02
> Generate a DVC (Data Version Control) command to start tracking the 'historical_financials.csv' file and associate it with the current git commit, ensuring data-to-code lineage.
- **Expected Response:** A shell command snippet for DVC, e.g., `dvc add data/historical_financials.csv`.

---

## Deployment & CI/CD
> *Prompts for automating the deployment of models and analysis pipelines into production or staging environments.*

### Task: DEP01
> Generate a Dockerfile to containerize the 'rating_model.pkl' and the API script created in task API01. The container should expose port 8080 and run the API server upon startup. Ensure all necessary Python dependencies from a 'requirements.txt' file are installed.
- **Expected Response:** A complete, valid Dockerfile within a code block.

### Task: DEP02
> Generate a Kubernetes deployment manifest in YAML (`deployment.yaml`) to deploy the container image created from the Dockerfile in DEP01. The deployment should specify 2 replicas and include a liveness probe that checks the API's '/health' endpoint every 30 seconds.
- **Expected Response:** A complete, valid Kubernetes deployment YAML file.

---

## Performance Monitoring & Alerting
> *Prompts for setting up real-time monitoring of model performance, data drift, and system health.*

### Task: MON01
> Generate a JSON configuration for a monitoring alert. The alert should trigger if the 95th percentile latency of the rating prediction API exceeds 500ms over a 5-minute window. The alert should be sent to the '#credit-ops-alerts' Slack channel via a webhook URL stored in the 'SLACK_WEBHOOK' environment variable.
- **Expected Response:** A JSON object defining the alert conditions and notification channel.

### Task: MON02
> Design a data drift detection job. Generate the pseudocode for a script that runs daily. The script should calculate the statistical distribution (e.g., mean, standard deviation) of key input features from the live prediction requests over the last 24 hours and compare it to the distribution of the training data using the Kolmogorov-Smirnov test. If the p-value for any feature is below 0.05, it should log a 'Drift Detected' warning.
- **Expected Response:** Detailed pseudocode or a Python script outline for the data drift detection job.

---

## Retraining & Lifecycle Hooks
> *Prompts for defining automated triggers and policies for model retraining, testing, and promotion.*

### Task: RET01
> Define a model retraining policy. Generate a JSON object that specifies the conditions under which the rating model should be automatically retrained. The policy should include two triggers: 1) A 'time-based' trigger to retrain every 90 days. 2) A 'performance-based' trigger if the model's prediction accuracy on a validation set drops below 85%.
- **Expected Response:** A JSON object defining the retraining policy with its triggers.

### Task: RET02
> Generate the configuration for a CI/CD pipeline hook. Upon successful completion of a retraining job, the new candidate model must be automatically benchmarked against the currently deployed production model on a hold-out 'challenger' dataset. The new model can only be promoted to staging if its F1-score is at least 2% higher than the production model's score.
- **Expected Response:** A YAML or JSON configuration snippet defining the post-retraining benchmarking and promotion gate.

---

## Ethical & Compliance Guardrails
> *Prompts for defining and enforcing ethical guidelines, fairness checks, and regulatory compliance within the AI system.*

### Task: ETH01
> Generate a test case in Python to check the rating model for fairness. Using a hypothetical dataset with a 'region' feature, the test should calculate the Demographic Parity Difference for the 'Investment Grade' prediction outcome between the 'North America' and 'Europe' groups. The test fails if the absolute difference is greater than 10%.
- **Expected Response:** A Python script or function that implements the specified fairness test.

### Task: ETH02
> Define a compliance guardrail as a JSON policy object. This policy must prevent the agent from executing any prompt that involves processing Personally Identifiable Information (PII) unless the prompt explicitly references a 'compliance_approval_code'. The policy should also specify a regex pattern for detecting common PII like email addresses and phone numbers.
- **Expected Response:** A JSON object detailing the PII detection pattern and the compliance check logic.

---

# IX. Human-Computer Interaction & Collaboration

## User Preference & Adaptation
> *Prompts for tailoring the AI's behavior, verbosity, and output format to individual user needs and expertise levels.*

### Task: HCI01
> Set my user profile to 'Expert'. For all subsequent responses, minimize conversational filler, use dense technical language, and provide outputs directly in their final format (e.g., code, JSON) without narrative explanation unless explicitly requested.
- **Expected Response:** Confirmation of user profile change, e.g., 'Expert mode enabled.'

### Task: HCI02
> I am a novice user. For the next task, 'Explain the results of the Monte Carlo simulation (SIM01)', please use an analogy and avoid statistical jargon like 'standard deviation' or 'p-value'. Focus on the business implications of the potential outcomes.
- **Expected Response:** A simplified, analogy-driven narrative explanation of the simulation results.

---

## Collaborative Workspace Management
> *Prompts for managing shared analytical sessions, tracking contributions, and resolving conflicts between multiple users.*

### Task: COL01
> Initialize a new collaborative analysis session for the 'Peer Analysis' stage. Invite users 'analyst_jane@example.com' and 'manager_bob@example.com'. All prompts and outputs within this session should be logged with user attribution.
- **Expected Response:** Confirmation of session creation and user invitations, returning a unique session ID.

### Task: COL02
> User 'analyst_jane@example.com' has proposed a peer group in task PA01. User 'manager_bob@example.com' has proposed a different peer group. Present a side-by-side comparison of the two proposed peer groups and highlight the key differences in their business mix and financial metrics to facilitate a decision.
- **Expected Response:** A Markdown table or comparative narrative summarizing the two conflicting inputs.

### Task: COL03
> Lock the 'Financial Statement Adjustments' (FSA01-FSA03) section of the analysis. No further changes can be made to these tasks by any user without explicit override from a user with 'Team Lead' permissions.
- **Expected Response:** Confirmation that the specified analytical section has been locked.

---

## Feedback & Reinforcement Learning
> *Prompts for capturing user feedback to improve the AI's future performance and fine-tune its models.*

### Task: FBK01
> The rationale provided in RR03 was not persuasive. The causal link between the company's competitive position and its financial forecast was unclear. Use this feedback to regenerate the rationale, placing greater emphasis on that specific connection. Register this feedback instance for model improvement.
- **Expected Response:** A revised narrative for the rating rationale that explicitly incorporates the user's feedback.

### Task: FBK02
> The peer group selected in PA01 was excellent and highly relevant. Create a positive reinforcement signal for the selection logic used. Associate this successful outcome with the input features: industry='specialty chemicals', revenue_size='<$1B', geo_focus='North America'.
- **Expected Response:** Confirmation that a positive feedback signal has been logged for reinforcement learning, associating the successful output with the specified input conditions.

---

# X. Security, Privacy & Access Control

## Permissions & Role-Based Access (RBAC)
> *Prompts for defining and enforcing granular access controls over tasks, data, and system capabilities.*

### Task: SEC01
> Define a new user role named 'Junior Analyst'. Generate a JSON RBAC policy that grants this role 'read-only' access to all stages up to 'V. Synthesis, Rating, and Reporting', and explicit 'deny' access to all subsequent stages (VI-X). The role is also denied access to any task involving the 'delete' or 'deploy' verbs.
- **Expected Response:** A JSON object representing the Role-Based Access Control policy.

### Task: SEC02
> The current user is requesting to execute task DEP02 (Kubernetes Deployment). Verify if the user's role has the necessary 'execute' permission for the 'deployment_and_cicd' section. Provide a confirmation or denial message based on the current RBAC policy.
- **Expected Response:** A confirmation or denial string, e.g., 'ACCESS DENIED: User role 'Junior Analyst' lacks 'execute' permission for section 'deployment_and_cicd'.'

---

## Data Privacy & Anonymization
> *Prompts for handling sensitive data, performing anonymization, and ensuring compliance with privacy regulations.*

### Task: PRIV01
> Before processing the attached document 'employee_census.csv', run a PII scan and generate a data masking plan. The plan should identify columns containing names, addresses, and social security numbers, and specify a masking technique for each (e.g., 'hash', 'redact', 'substitute_with_placeholder').
- **Expected Response:** A JSON object representing the data masking plan.

### Task: PRIV02
> Generate a differential privacy query. Apply a Laplace mechanism with a specified privacy budget (epsilon) of 1.0 to a query that calculates the average salary from the 'employee_census.csv' file. Generate the Python code to execute this differentially private query.
- **Expected Response:** A Python script using a differential privacy library (e.g., Google's diff-privlib, OpenDP) to perform the noisy query.

---

## Security Auditing & Logging
> *Prompts for logging security-sensitive events and generating reports for compliance and forensic analysis.*

### Task: AUD01
> A request to access a sensitive document was denied. Create a high-priority security event log entry. The log must be in JSON format and include a timestamp, the requesting user's ID, the target resource ('doc_merger_prospectus.pdf'), the result ('ACCESS_DENIED'), and the ID of the RBAC policy that blocked the request.
- **Expected Response:** A single JSON object representing the structured security event log.

### Task: AUD02
> Generate a security report for the last 7 days. The report should summarize: 1) The number of failed login attempts by user. 2) A list of all access requests to resources tagged as 'highly_sensitive'. 3) All actions performed by users with the 'Administrator' role. The output should be a formatted Markdown file.
- **Expected Response:** A structured Markdown report containing the requested security audit summary.
