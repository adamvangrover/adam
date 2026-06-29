# 02 - Your First Ingestion Pipeline

This tutorial walks you through pulling a 10-K from the Edgar ingestion module and passing it through the basic sentiment engine.

## Step 1: Initialize the Edgar Ingestion Module
Ensure your local ADAM instance is running.

Run the ingestion script to pull the latest 10-K for a specific ticker (e.g., AAPL):
```bash
uv run python adam_ingest/edgar_pull.py --ticker AAPL --form 10-K
```

## Step 2: Observe the Data
The unstructured regulatory filing is temporarily downloaded to `data/raw/AAPL_10-K.txt`.

## Step 3: Run the Sentiment Engine
Now, pass this unstructured data through the System 1 sentiment engine to extract semantic vectors and sentiment scores.

```bash
uv run python adam_semantic/analyze_sentiment.py --input data/raw/AAPL_10-K.txt
```

You should see an output containing a structured Pydantic object with conviction scores, anomaly flags, and semantic insights. This demonstrates the transformation of unstructured regulatory text into structured intelligence.
