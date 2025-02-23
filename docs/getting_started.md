# Getting Started with Adam v17.0

This guide will walk you through the process of setting up Adam v17.0 and running your first analysis.

## Prerequisites

*   Python 3.7+
*   pip (Python package installer)

## Installation

1.  **Clone the Repository:**

    ```bash
    git clone [https://github.com/adamvangrover/adam.git](https://github.com/adamvangrover/adam.git)  # Replace with your actual repo URL if different
    cd adam
    ```

2.  **Navigate to the Core Directory:**

    ```bash
    cd core
    ```

3.  **Install Required Packages:**

    ```bash
    pip install -r requirements.txt  # If a requirements file exists (recommended)
    # Or install individual packages:
    pip install numpy pandas matplotlib  # Example packages - adjust as needed
    ```

4.  **Knowledge Base Setup:**

    *   The Knowledge Base is stored in the `data/knowledge_base.json` file. A sample file has been provided. You can customize this file with your own data.  Ensure the `data/` directory is at the root of your Adam project, alongside the `core/` directory.

## Running an Analysis

The following sections will demonstrate how to perform a basic stock analysis using Adam v17.0.

## Example 1: Analyzing Tech Innovators Inc. (Simulated)

This example demonstrates how to use Adam v17.0 to analyze the *simulated* performance of "Tech Innovators Inc."  Remember, this example uses simulated data.  Real-world data integration will be covered in a later section.

1.  **Import Necessary Modules:**

    ```python
    from core.market_sentiment_agent import MarketSentimentAgent
    from core.fundamental_analyst_agent import FundamentalAnalystAgent
    from core.technical_analyst_agent import TechnicalAnalystAgent
    import json
    import matplotlib.pyplot as plt
    import os  # For creating the output directory
    ```

2.  **Load the Knowledge Base:**

    ```python
    with open("../data/knowledge_base.json", "r") as f:  # Adjust path if necessary
        knowledge_base = json.load(f)
    ```

3.  **Initialize Agents:**

    ```python
    sentiment_agent = MarketSentimentAgent(knowledge_base)
    fundamental_analyst = FundamentalAnalystAgent(knowledge_base)
    technical_analyst = TechnicalAnalystAgent(knowledge_base)
    ```

4.  **Simulate Data (Placeholder):**

    ```python
    # In a real-world scenario, this data would come from a live data feed.
    # For this example, we'll use simulated data.
    simulated_stock_data = {
        "price_history": [100, 105, 110, 108, 112, 115, 120],
        "earnings_per_share": 10,
        "analyst_sentiment": "positive"
    }
    ```

5.  **Perform Analysis:**

    ```python
    sentiment_result = sentiment_agent.analyze(simulated_stock_data["analyst_sentiment"])
    fundamental_result = fundamental_analyst.analyze(simulated_stock_data["earnings_per_share"])
    technical_result = technical_analyst.analyze(simulated_stock_data["price_history"])
    ```

6.  **Access Knowledge Base Information:**

    ```python
    pe_ratio_definition = knowledge_base["PriceToEarningsRatio"]["definition"]
    print(f"Price-to-Earnings Ratio Definition: {pe_ratio_definition}")

    analyst_sentiment_interpretation = knowledge_base["AnalystSentiment"]["interpretation"][simulated_stock_data["analyst_sentiment"]]
    print(f"Analyst Sentiment Interpretation: {analyst_sentiment_interpretation}")
    ```

7.  **Visualize Results:**

    ```python
    # Create the output directory if it doesn't exist
    output_dir = "../outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.plot(simulated_stock_data["price_history"])
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title("Tech Innovators Inc. (Simulated)")
    plt.savefig(os.path.join(output_dir, "tech_innovators_price.png"))  # Save to output directory
    plt.show()
    ```

8.  **Output Summary:**

    ```python
    print("Market Sentiment Analysis:", sentiment_result)
    print("Fundamental Analysis:", fundamental_result)
    print("Technical Analysis:", technical_result)
    ```

## Next Steps

Explore the other agents and modules within the `core/` directory.  Contribute to the project by adding new agents, improving documentation, or providing feedback.  Stay tuned for updates on real-world data integration and more advanced features.
