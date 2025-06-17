````markdown
# Adam v19.1 User Guide

This guide provides comprehensive instructions on how to use Adam v19.1, the advanced financial analytics system. It covers various aspects, including:

* Accessing and utilizing the knowledge graph
* Interacting with the API
* Running different analysis modules
* Interpreting results and generating reports
* Customizing strategies and settings

## Knowledge Graph

Adam v17.0's knowledge graph is a rich repository of financial concepts, models, and data, organized in a structured and interconnected manner. It enables Adam to perform in-depth analysis, provide context-aware insights, and generate actionable recommendations.

### Accessing the Knowledge Graph

You can access the knowledge graph through the following methods:

* **API:** The Adam v17.0 API provides endpoints for retrieving and updating information in the knowledge graph. See the API section for more details and examples.
* **Direct Access:** You can also directly access the knowledge graph data stored in the `data/knowledge_graph.json` file. This file is structured in JSON format and can be easily parsed and queried using various tools and libraries. Here's an example of how to access the knowledge graph data using Python:

```python
import json

with open('data/knowledge_graph.json', 'r') as f:
    knowledge_graph = json.load(f)

# Access the nodes and edges
nodes = knowledge_graph['Valuation']['DCF']['machine_readable']['nodes']
edges = knowledge_graph['Valuation']['DCF']['machine_readable']['edges']

# Print the nodes and edges
print(nodes)
print(edges)
````

### Utilizing the Knowledge Graph

The knowledge graph can be used for various purposes, including:

  * **Understanding Financial Concepts:** Explore the definitions, explanations, and relationships between different financial concepts. Each node in the knowledge graph represents a concept, and the edges represent the relationships between them. For example, you can explore the concept of "Discounted Cash Flow (DCF)" and its relationship to other concepts like "Free Cash Flow" and "Discount Rate."
  * **Conducting Research:** Use the knowledge graph to research specific topics or areas of interest. You can traverse the graph to find related concepts and explore their connections. For example, you can start with the concept of "Market Risk" and traverse the graph to explore different types of market risks, such as "Interest Rate Risk" and "Equity Price Risk."
  * **Validating Information:** Verify the accuracy and consistency of financial data and models. The knowledge graph provides a structured representation of financial knowledge that can be used for validation purposes. For example, you can check if the formula for calculating "Debt-to-Equity Ratio" is consistent with the definition in the knowledge graph.
  * **Developing New Agents and Modules:** Leverage the knowledge graph to develop new agents and modules that can enhance Adam's capabilities. The knowledge graph can serve as a foundation for building new AI components that can reason about financial information. For example, you can use the knowledge graph to develop an agent that specializes in analyzing ESG (Environmental, Social, and Governance) factors.

## API

The Adam v17.0 API provides a unified interface for interacting with the system. It allows you to access various functionalities, including:

  * **Retrieving Data:** Get data from the knowledge graph, market data feeds, and other sources.
  * **Running Analysis Modules:** Execute different analysis modules, such as market sentiment analysis, fundamental analysis, and technical analysis.
  * **Generating Reports:** Create customized reports based on the analysis results.
  * **Managing Agents:** Control the behavior and interactions of different agents within the system.

### API Documentation

Detailed API documentation is available in the `docs/api_docs.yaml` file. It outlines the available endpoints, request parameters, and response formats. You can use tools like Swagger UI to visualize and interact with the API documentation.

### API Examples

Here are a few examples of how to use the API:

  * **Get the DCF valuation for AAPL:**

    ```bash
    curl -X POST /api/v1 \
         -H "Content-Type: application/json" \
         -d '{"module": "valuation", "action": "get_dcf_valuation", "parameters": {"company": "AAPL", "forecast_period": 5}}'
    ```

  * **Get the latest market sentiment for the technology sector:**

    ```bash
    curl -X POST /api/v1 \
         -H "Content-Type: application/json" \
         -d '{"module": "market_sentiment", "action": "get_sentiment", "parameters": {"sector": "technology"}}'
    ```

  * **Update the risk-free rate in the knowledge graph:**

    ```bash
    curl -X POST /api/v1 \
         -H "Content-Type: application/json" \
         -d '{"module": "knowledge_graph", "action": "update_node", "parameters": {"node_id": "risk_free_rate", "new_value": 0.02}}'
    ```

## Analysis Modules

Adam v19.1 provides various analysis modules that can be used to gain insights into financial markets and make informed investment decisions. These modules include:

  * **Market Sentiment Analysis:** Analyzes news articles, social media, and financial forums to gauge investor sentiment towards different assets and markets.
  * **Macroeconomic Analysis:** Monitors and interprets key economic indicators (e.g., GDP, inflation, interest rates) to assess the macroeconomic environment and its potential impact on investments.
  * **Geopolitical Risk Analysis:** Identifies and analyzes geopolitical risks (e.g., political instability, trade wars) and their potential impact on financial markets.
  * **Fundamental Analysis:** Performs in-depth fundamental analysis of companies, including financial statement analysis, valuation modeling, and risk assessment.
  * **Technical Analysis:** Analyzes price charts, technical indicators, and patterns to identify trading opportunities and generate trading signals.

### SNC Analyst Agent

The Shared National Credit (SNC) Analyst Agent is designed to assess the credit risk of large, syndicated loans based on established regulatory guidelines. It provides a regulatory rating (e.g., Pass, Special Mention, Substandard) along with a detailed justification for its assessment.

**Key XAI Features:**

*   **Detailed Justifications:** The agent's primary output includes a human-readable rationale that explains the factors leading to the assigned rating. These justifications now incorporate references to specific sections or codes from relevant regulatory handbooks (e.g., Comptroller's Handbook, OCC Guidelines), making the reasoning more transparent and auditable.
*   **XAI Execution Trace:** For an in-depth understanding of the agent's decision-making process, users can request a full XAI execution trace.
    *   If interacting with the agent programmatically, this is typically done by passing an `include_xai_trace=True` parameter to its execution method.
    *   This trace provides a step-by-step log of the agent's operations, including data inputs, skill invocations (internal reasoning steps), parameters passed, intermediate assessments, and how these contribute to the final rating and rationale.
    *   The trace is provided in a structured format (JSON) and allows for a granular review of the entire analysis.
*   **Accessing the Output:** The agent's output is a structured dictionary containing the `rating`, `rationale`, and optionally, the `xai_trace`. For example:
    ```json
    {
      "rating": "Substandard",
      "rationale": "Substandard rating influenced by SK assessments (Non-Accrual, Collateral, or Repayment indicating weaknesses, citing CHB Paying Capacity Sec 3.2, OCC Non-Accrual Reg 12.3(a)). Fallback: Insufficient liquidity and interest coverage.",
      "xai_trace": { /* ... detailed trace data ... */ }
    }
    ```

### Running Analysis Modules

You can run analysis modules through the API or by directly calling the corresponding Python scripts in the `core/modules` directory.

**Example using API:**

```bash
curl -X POST /api/v1 \
     -H "Content-Type: application/json" \
     -d '{"module": "fundamental_analysis", "action": "analyze_company", "parameters": {"company": "MSFT"}}'
```

**Example using Python script:**

```bash
python core/modules/fundamental_analysis.py --company MSFT
```

### Interpreting Results

The results of the analysis modules are typically presented in a structured format, such as JSON or CSV. You can then use these results to generate reports, create visualizations, or develop custom trading strategies.

**Example:**

The `fundamental_analysis.py` script might output a JSON file containing the following information:

```json
{
  "company": "MSFT",
  "revenue": 168088000000,
  "net_income": 61271000000,
  "eps": 8.04,
  "pe_ratio": 35.5,
  "debt_to_equity": 0.45,
  //... other financial metrics
}
```

You can then use this data to generate a report or create a visualization of MSFT's financial performance.

## Customizing Strategies and Settings

Adam v19.1 allows you to customize various aspects of the system, including:

  * **Investment Strategies:** Define your own investment strategies based on Adam's insights and your risk tolerance and investment goals. You can use the API or configuration files to specify your investment preferences and constraints.
  * **Agent Behavior:** Configure the behavior and interactions of different agents within the system. You can adjust the parameters of each agent to fine-tune their analysis and decision-making processes.
  * **Data Sources:** Add or remove data sources to customize the information that Adam uses for its analysis. You can connect to different financial data providers, databases, or APIs to enrich Adam's knowledge base.
  * **Alerting:** Set up alerts to be notified of specific events or market conditions. You can define custom alert rules based on various factors, such as price movements, news events, or sentiment changes.

### Configuration File

The `config/` directory contains modular YAML files for customizing various settings for Adam v19.1 (e.g., `agents.yaml`, `system.yaml`). Refer to these files and any `example_*.yaml` files for detailed instructions and examples. The main `config/config.yaml` is deprecated for direct edits.

## Contributing

Contributions to Adam v19.1 are welcome\! Please check the [CONTRIBUTING.md](https://www.google.com/url?sa=E&source=gmail&q=CONTRIBUTING.md) file for guidelines on how to contribute to the project.

## Support and Feedback

If you have any questions or feedback, please feel free to reach out to the Adam v19.1 development team. You can submit issues or pull requests on the GitHub repository or contact the developers directly via email or other communication channels.

```
```
