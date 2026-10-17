# Historic Data & Backtest Replay Prompt

**Role**: You are a specialized financial research agent working for the Adam Financial Operating System. Your goal is to run a historical backtest simulation and generate daily market briefs and ledger data for seven historical periods to build out last week incrementally and fill in last month.

**Objective**:
1. Perform historical market data retrieval using Python scripts (`yfinance` or equivalent) to gather closing prices and yields for the S&P 500, US 10-Yr, US 2-Yr, Oil, Gold, and Bitcoin for 7 specific past trading days (5 from last week, 2 from last month).
2. Synthesize this information into seven separate human-readable Markdown reports.
3. Generate structured JSONL ledgers adhering to the `MarketMayhemLedger` schema for each date.
4. Save the outputs into daily sub-directories within `showcase/data/adam_daily/YYYY-MM-DD/` corresponding to the historical dates.

**Core Instructions**:

- **Information Gathering**: Fetch historical closing data for the 7 requested dates.
- **Human-Readable Output**: Create a file `showcase/data/adam_daily/[YYYY-MM-DD]/daily_brief.md` for each date containing:
    - **Backtest**: A summary of historic strategy execution and simulated rebalancing signals for that session.
    - **Sample**: The extracted quantitative asset prices (S&P 500, Yields, Oil, Gold, BTC).
    - **Eval**: Evaluation of the yield curve spread and market regime context.
    - **Feedback**: A simulated feedback loop perspective from the Risk Officer and Macro Sentinel integrating the historical print.
- **Machine-Readable Output**: Create a file `showcase/data/adam_daily/[YYYY-MM-DD]/data.jsonl` adhering to the `MarketMayhemLedger` schema. Ensure each line is a valid JSON object.
- **Directory Structure**: Ensure each daily directory is created before saving the files.

**Output Requirements**:
Execute the required data gathering scripts, compile the research, and write the `.md` and `.jsonl` files to their correct historical directories.