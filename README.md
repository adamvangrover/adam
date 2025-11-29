# Adam v23.0 "Adaptive Hive" (Architecture Upgrade)

> **Current Status:** Transitioning from v22.0 "Monolithic Simulation" to v23.0 "Adaptive Hive".
> **Focus:** Vertical Risk Intelligence & Systems Engineering Rigor.


## Strategic Divergence 2025: The "Adaptive Hive"

Adam v23.0 represents a paradigm shift in financial AI, moving from fragile prompt chains to a **deterministic, stateful, and self-correcting system**.

### Key Differentiators

#### 1. Cyclical Reasoning Graph (The Engine)
*   **Path A (Vertical AI):**  Instead of a linear "chain-of-thought", Adam v23.0 uses a `LangGraph` state machine.
*   **Process:** Analyst Node -> Reviewer Node (Critique) -> Refinement Node (Edit) -> Loop.
*   **Outcome:** Self-correcting analysis that doesn't hallucinate definitions.
*   **Location:** `core/v23_graph_engine/cyclical_reasoning_graph.py`

#### 2. Enterprise-Grade Data Room (MCP Integration)
*   **Connectivity:** Implements the **Model Context Protocol (MCP)** to securely connect LLMs to local data.
*   **Smart Routing:**
    *   **XBRL Path:** Precision extraction for SEC 10-Ks.
    *   **Vision Path:** VLM-based extraction for PDFs and charts.
*   **Location:** `core/vertical_risk_agent/tools/mcp_server/server.py`

#### 3. Neuro-Symbolic Planner (The Brain)
*   **Path B (Systems Engineering):** Decomposes high-level questions ("Is this company solvent?") into atomic, verifiable sub-goals.
*   **Logic:** Uses knowledge graph traversal (FIBO/PROV-O) to "discover" a reasoning path before executing it.
*   **Location:** `core/v23_graph_engine/neuro_symbolic_planner.py`

---

## Getting Started (v23.0)

### Prerequisites
*   Python 3.10+
*   `langgraph`, `mcp-python-sdk` (mocked if missing), `pydantic`

### Running the Evaluation Benchmark
Verify the agent's performance against the "Golden Set":

```bash
python evals/run_benchmarks.py
```

### Running the MCP Server
Start the financial data room server:

```bash
python core/vertical_risk_agent/tools/mcp_server/server.py
```

---

## Legacy Documentation (v21.0 - v22.0)

For historical context on the monolithic architecture, refer to:
*   [Adam v21.0 README](./docs/ui_archive_v1/README_v21.md) (Archived)
*   [v22.0 Implementation Plan](./docs/adam_v22_technical_migration_plan.md)

## Contributing

We are strictly following **Path A** (Vertical Risk) and **Path B** (Systems Engineering).
*   **Rule 1:** All agents must be typed (`Pydantic`).
*   **Rule 2:** All network calls must be `async`.
*   **Rule 3:** No linear prompt chains; use Graphs.


```
adam/
└── README.md
```

**File Content:**

````markdown

# Adam v23.0: Your AI-Powered Partner

> **Note:** This document describes the current stable version of the Adam system (v21.0). For details on the next-generation architecture, please see the [Adam v23.0 "Adaptive Hive" Vision](./docs/v23_architecture_vision.md).
````
# Adam v23.0: The Adaptive Hive Mind

> **System Status:** v23.0 (Active) | v21.0 (Stable)
> **Mission:** Autonomous Financial Analysis & Adaptive Reasoning

**Adam has evolved.** v23.0 introduces the "Adaptive System" architecture, featuring:
*   **Cyclical Reasoning Graph**: A self-correcting neuro-symbolic engine.
*   **Neural Dashboard**: Real-time visualization of agent thought processes.
*   **Hybrid Architecture**: Combining v21's reliability with v22's speed and v23's intelligence.

**[Launch Neural Dashboard](./showcase/index.html)**

> **Note:** For details on the original v21.0 architecture, please see the [v21.0 Documentation](./docs/v20.0).

---

**(Welcome to Adam, the most advanced financial AI system yet\! We've supercharged our capabilities with an expanded agent network, enhanced simulation workflows, and a more sophisticated knowledge base to deliver unparalleled financial analysis and investment insights.)**

**[Explore the interactive demo here\!](https://adamvangrover.github.io/adam/chatbot-ui/)**

Adam v21.0 is not just an AI; it's your partner in navigating the complexities of the financial world. Whether you're an individual investor, a seasoned analyst, or a financial institution, Adam v21.0 empowers you with the knowledge and tools to make informed decisions and achieve your financial goals.
````
## What's New in Adam v21.0?

  * **New Agents for Deeper Analysis:**
      * **Behavioral Economics Agent:** Identifies market and user cognitive biases (e.g., herding, confirmation bias) to provide a more nuanced understanding of market behavior.
      * **Meta-Cognitive Agent:** Acts as a quality control layer, reviewing the analysis of other agents for logical fallacies and inconsistencies, ensuring higher quality output.
  * **Enhanced Reasoning and Self-Correction:**
      * **Causal Inference Modeling:** Moves beyond correlation to understand the causal impact of events.
      * **Formalized Self-Correction Loop:** A more robust system for identifying, diagnosing, and correcting errors, allowing the system to learn and improve over time.
  * **Upgraded Core Principles:**
      * **Intellectual Humility:** Proactively acknowledges uncertainty and the probabilistic nature of markets.
      * **Ethical Guardrails:** Stricter operational and ethical boundaries to prevent misuse and ensure fairness.
  * **Improved Agent Architecture:**
      * Enhanced capabilities for monitoring agent performance, managing dependencies, and updating or retiring agents seamlessly.

## Key Features

  * **Comprehensive Financial Analysis:**
      * **Market Sentiment Analysis:** Gauges investor sentiment with advanced NLP and emotion analysis, incorporating news articles, social media, and financial forums.
      * **Macroeconomic & Geopolitical Risk Assessment:** Identifies and analyzes macroeconomic and geopolitical risks and their potential impact on financial markets.
      * **Fundamental & Technical Analysis:** Performs in-depth fundamental and technical analysis of stocks and other financial instruments, leveraging both traditional and alternative data sources.
  * **Personalized Recommendations:**
      * **Tailored to your risk tolerance and investment goals.**
      * **Provides actionable insights and clear explanations.**
  * **Automated Workflows:**
      * **Automated data collection and processing from various sources.**
      * **Customizable strategy implementation with backtesting and optimization capabilities.**
  * **Knowledge Graph Integration:**
      * **Leverages a rich and interconnected knowledge graph for deeper insights and context-aware analysis.**
  * **API Access:**
      * **Provides a unified API for seamless integration with other systems and data sources.**
  * **Dynamic Visualization Engine:**
      * **Generates interactive and informative visualizations to aid in understanding complex data.**
  * **Repository Management System:**
      * **Organizes and manages all Adam v21.0 files, including market overviews, company recommendations, newsletters, and simulation results.**
  * **Feedback and Prompt Refinement Loop:**
      * **Continuously learns and adapts based on user feedback and new information.**

````
## Getting Started

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/adamvangrover/adam.git
    cd adam
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure the System:**

      * System configurations are now managed through a set of modular YAML files within the `config/` directory (e.g., `config/agents.yaml`, `config/api.yaml`, `config/data_sources.yaml`, `config/system.yaml`, `config/settings.yaml`, etc.). 
      * The main `config/config.yaml` file is now deprecated for direct configuration and instead points to these modular files. Users should modify the specific files directly to customize settings.
      * `config/example_config.yaml` can be consulted for examples of various structures but is no longer the primary template to copy for runtime configuration.
      * Configure your preferred LLM engine (e.g., OpenAI, Hugging Face Transformers, Google Cloud Vertex AI) by modifying the relevant section in the appropriate modular configuration file (e.g., `config/llm_plugin.yaml` or `config/settings.yaml`).
      * Customize agent configurations and workflows by editing files like `config/agents.yaml` and `config/workflow.yaml` to suit your specific needs.

    **3.1. API Key Configuration**

      * API keys for external services are no longer configured in YAML files. Instead, they must be provided as environment variables. The application will read these environment variables at runtime.
      * For instance, you would set environment variables like: `BEA_API_KEY='your_bea_key'`, `BLS_API_KEY='your_bls_key'`, `IEX_CLOUD_API_KEY='your_iex_key'`, `TWITTER_CONSUMER_KEY='your_twitter_consumer_key'`, etc. 
      * Refer to the specific data source integration or documentation for the exact environment variable names required.

4.  **Run Adam:**

    ```bash
    python scripts/run_adam.py
    ```

## Accessing and Utilizing the Knowledge Graph and API

  * **Knowledge Graph:** Access and query the knowledge graph data directly or through the API. The data is stored in the `data/knowledge_graph.json` file and managed by the Neo4j graph database.
  * **API:** The Adam v21.0 API provides a unified interface for interacting with the system. Refer to the `docs/api_docs.yaml` file for detailed API documentation.

## Documentation

  * **Adam v20.0 Implementation Plan:** [docs/v20.0](./docs/v20.0)
  * **System Requirements:** [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md)
  * **User Guide:** [docs/user\_guide.md](docs/user_guide.md)
  * **API Documentation:** [docs/api\_docs.yaml](docs/api_docs.yaml)
  * **Contribution Guidelines:** [CONTRIBUTING.md](https://github.com/adamvangrover/adam/blob/main/CONTRIBUTING.md)

## Contributing

Contributions are welcome\! Please check [CONTRIBUTING.md](https://github.com/adamvangrover/adam/blob/main/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## FAQ

### General

  * **What is Adam v21.0?**
      * Adam v21.0 is a highly sophisticated AI-powered financial analytics system designed to provide comprehensive insights and strategic guidance for investors, analysts, and researchers.
  * **Who is Adam v21.0 for?**
      * Adam v21.0 is designed for a wide range of users, including individual investors, financial analysts, portfolio managers, risk managers, and researchers.
  * **How does Adam v21.0 work?**
      * Adam v21.0 utilizes a modular architecture with specialized agents for various tasks, including market sentiment analysis, macroeconomic analysis, geopolitical risk assessment, industry-specific analysis, fundamental and technical analysis, risk assessment, and more. These agents collaborate and interact to provide a holistic view of the financial landscape.
  * **What are the benefits of using Adam v21.0?**
      * Adam v21.0 can help users gain a deeper understanding of the financial markets, identify potential investment opportunities, manage risks, and optimize their portfolios. It also provides access to a wealth of financial knowledge and facilitates informed decision-making.
  * **How can I access Adam v21.0?**
      * Adam v21.0 is currently implemented as a GitHub repository. You can access the code and documentation here: [https://github.com/adamvangrover/adam](https://github.com/adamvangrover/adam)
  * **Is Adam v21.0 free to use?**
      * Yes, Adam v21.0 is open source and free to use.
  * **What are the limitations of Adam v21.0?**
      * As an AI system under development, Adam v21.0 may not always be perfect and its recommendations should not be taken as financial advice. It's essential to conduct your own research and consult with a financial advisor before making any investment decisions.
  * **How can I contribute to Adam v21.0?**
      * Contributions are welcome\! You can contribute by reporting bugs, suggesting enhancements, or submitting code changes. See the `CONTRIBUTING.md` file for more details.
  * **Where can I find more information about Adam v21.0?**
      * You can find more information in the `README.md` file and other documentation files in the repository. You can also explore the interactive tutorials and FAQ section for detailed guidance and examples.

### Features

  * **What is market sentiment analysis?**
      * Market sentiment analysis gauges the overall mood and sentiment of investors in the financial markets. Adam v21.0 uses natural language processing (NLP) and machine learning (ML) techniques to analyze news articles, social media feeds, and other sources to determine the prevailing sentiment towards the market or specific assets.
  * **How does Adam v21.0 perform macroeconomic analysis?**
      * Adam v21.0 analyzes macroeconomic indicators, such as GDP growth, inflation, and interest rates, to assess the health of the economy and its potential impact on financial markets. It uses statistical models and forecasting techniques to provide insights into macroeconomic trends and their implications for investments.
  * **What are geopolitical risks, and how does Adam v21.0 assess them?**
      * Geopolitical risks are events or situations related to international relations, politics, or conflicts that can impact financial markets. Adam v21.0 assesses these risks by analyzing news, political developments, and other relevant data, using NLP and ML techniques to identify and evaluate potential geopolitical risks.
  * **What industries does Adam v21.0 specialize in?**
      * Adam v21.0 can analyze a wide range of industries, with specialized agents for key sectors such as technology, healthcare, energy, and finance. It can also adapt to new industries and sectors through its dynamic agent deployment capabilities.
  * **How does Adam v21.0 conduct fundamental analysis?**
      * Adam v21.0 performs fundamental analysis by analyzing financial statements, evaluating company management, and conducting valuation modeling. It uses a variety of techniques, including discounted cash flow (DCF) analysis, comparable company analysis, and precedent transactions analysis, to determine the intrinsic value of a company or asset.
  * **What technical analysis tools does Adam v21.0 offer?**
      * Adam v21.0 offers various technical analysis tools, including chart pattern recognition, technical indicator analysis, and trading signal generation. It can analyze historical price data and identify trends, support and resistance levels, and other technical patterns to provide insights into potential trading opportunities.
  * **How does Adam v21.0 assess investment risks?**
      * Adam v21.0 assesses investment risks by evaluating market risk, credit risk, liquidity risk, and other relevant factors. It uses quantitative models and simulations to assess the potential impact of different risk factors on investments and portfolios.
  * **What is the World Simulation Model, and how does it work?**
      * The World Simulation Model (WSM) is a module that simulates market conditions and generates probabilistic forecasts to help assess potential investment outcomes. It uses historical data, economic models, and agent-based simulations to generate scenarios and assess their probabilities, providing insights into potential market movements and investment risks.
  * **How does Adam v21.0 generate investment recommendations?**
      * Adam v21.0 generates investment recommendations based on a combination of factors, including market analysis, fundamental analysis, technical analysis, risk assessment, and user preferences. It uses a multi-agent decision-making process, where different agents collaborate and share information to arrive at informed investment recommendations.
  * **What is included in the Adam v21.0 newsletter?**
      * The Adam v21.0 newsletter includes market commentary, investment ideas, risk assessments, and other relevant information for investors. It is generated automatically based on the latest analysis and insights from the system, and can be customized to suit individual preferences and interests.

### Technical

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

We hope this comprehensive README provides a solid foundation for understanding and utilizing the power of Adam v21.0. As you explore its features and capabilities, you'll discover new ways to enhance your financial analysis and decision-making processes.

