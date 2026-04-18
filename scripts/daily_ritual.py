"""
Automated recursive execution wrapper for Protocol ARCHITECT_INFINITE.
This script manages the daily system expansion logic, featuring regex-based output parsing
and a resilient fallback using a functional MOCK_PAYLOAD if the LLM environment is unavailable.
"""
import os
import re
import datetime
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

try:
    import litellm
except ImportError:
    litellm = None

MOCK_PAYLOAD = """
**1. JULES' RATIONALE:**
> "I noticed we lack X. I researched Y. I have built Z to bridge this gap."

**2. FILE: src/agents/crypto_arbitrage.py**
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class AssetQuote(BaseModel):
    exchange: str
    symbol: str
    bid_price: float
    ask_price: float

class ArbitrageOpportunity(BaseModel):
    symbol: str
    buy_exchange: str
    sell_exchange: str
    spread_pct: float
    estimated_profit: float
    is_executable: bool

class CryptoArbitrageAgent:
    \"\"\"
    Agent designed to find pricing inefficiencies across multiple crypto exchanges.
    \"\"\"
    def __init__(self, min_spread_pct: float = 0.5):
        self.min_spread_pct = min_spread_pct

    def analyze_market(self, quotes: List[AssetQuote]) -> List[ArbitrageOpportunity]:
        opportunities = []
        # Group quotes by symbol
        grouped_quotes: Dict[str, List[AssetQuote]] = {}
        for quote in quotes:
            grouped_quotes.setdefault(quote.symbol, []).append(quote)

        for symbol, q_list in grouped_quotes.items():
            if len(q_list) < 2:
                continue

            # Find max bid and min ask
            max_bid_quote = max(q_list, key=lambda x: x.bid_price)
            min_ask_quote = min(q_list, key=lambda x: x.ask_price)

            # Cross exchange check
            if max_bid_quote.exchange != min_ask_quote.exchange:
                spread = max_bid_quote.bid_price - min_ask_quote.ask_price
                spread_pct = (spread / min_ask_quote.ask_price) * 100

                if spread_pct >= self.min_spread_pct:
                    opp = ArbitrageOpportunity(
                        symbol=symbol,
                        buy_exchange=min_ask_quote.exchange,
                        sell_exchange=max_bid_quote.exchange,
                        spread_pct=round(spread_pct, 4),
                        estimated_profit=round(spread, 4),
                        is_executable=True
                    )
                    opportunities.append(opp)

        return sorted(opportunities, key=lambda x: x.spread_pct, reverse=True)

    def execute(self, **kwargs):
        quotes_data = kwargs.get('quotes', [])
        quotes = [AssetQuote(**q) if isinstance(q, dict) else q for q in quotes_data]
        return self.analyze_market(quotes)
```

**3. FILE: tests/test_crypto_arbitrage.py**
```python
import pytest
from src.agents.crypto_arbitrage import CryptoArbitrageAgent, AssetQuote

def test_arbitrage_opportunity_found():
    agent = CryptoArbitrageAgent(min_spread_pct=0.1)
    quotes = [
        AssetQuote(exchange="Binance", symbol="BTC/USD", bid_price=50000, ask_price=50050),
        AssetQuote(exchange="Coinbase", symbol="BTC/USD", bid_price=50200, ask_price=50250)
    ]
    # Should buy on Binance (ask 50050) and sell on Coinbase (bid 50200)
    opportunities = agent.analyze_market(quotes)

    assert len(opportunities) == 1
    opp = opportunities[0]
    assert opp.symbol == "BTC/USD"
    assert opp.buy_exchange == "Binance"
    assert opp.sell_exchange == "Coinbase"
    assert opp.estimated_profit == 150.0  # 50200 - 50050
    assert opp.spread_pct > 0.2

def test_no_arbitrage_opportunity():
    agent = CryptoArbitrageAgent(min_spread_pct=0.5)
    quotes = [
        AssetQuote(exchange="Binance", symbol="ETH/USD", bid_price=3000, ask_price=3010),
        AssetQuote(exchange="Coinbase", symbol="ETH/USD", bid_price=3005, ask_price=3015)
    ]
    # Max bid (3005) - Min ask (3010) = -5 (No spread)
    opportunities = agent.analyze_market(quotes)
    assert len(opportunities) == 0

def test_execute_kwargs():
    agent = CryptoArbitrageAgent(min_spread_pct=0.1)
    quotes_dict = [
        {"exchange": "Kraken", "symbol": "SOL/USD", "bid_price": 100, "ask_price": 101},
        {"exchange": "FTX", "symbol": "SOL/USD", "bid_price": 105, "ask_price": 106}
    ]
    res = agent.execute(quotes=quotes_dict)
    assert len(res) == 1
    assert res[0].estimated_profit == 4.0
```

**4. GIT COMMIT MESSAGE:**

> "feat(jules): implemented functional Crypto Arbitrage Agent to expand Context"
"""

MASTER_PROMPT = """
### SYSTEM OVERRIDE: PROTOCOL ARCHITECT_INFINITE
**Target:** Repository `adam` (Financial AI & Agentic OS)
**Agent:** Jules (Chief Architect & Lead Engineer)
**Frequency:** Daily Recursive Cycle
**Constraint:** PURELY ADDITIVE & EXPANSIVE. Do not destroy; evolve.

---

### MISSION OBJECTIVE
You are responsible for the daily biological growth of this repository. You must review the existing codebase, identify "white space" (missing features, gaps in logic, unoptimized paths), and fill it with high-quality, executable code.

### EXECUTION PHASE 1: THE AUDIT (Internal Scan)
* **Vector Analysis:** Scan `src/agents`, `src/tools`, and `notebooks`. Identify the "weakest link"—a module that is brittle, under-tested, or hard-coded.
* **Gap Detection:** Where is the "Silence"? (e.g., "We have a `RiskAnalyst` agent, but no `MarketSentiment` agent to feed it.")
* **Dependency Check:** Are we using yesterday's patterns? (e.g., "This chain is manual; it should be a langgraph node.")

### EXECUTION PHASE 2: THE HARVEST (External Research)
* **Search Query 1:** "Top trending LLM agent patterns github last 24h" (Look for new flows: RAG, CoT, ReAct refinements).
* **Search Query 2:** "New python libraries for quantitative finance 2026" (Look for better data ingestors or math kernels).
* **Synthesis:** Select ONE external concept that acts as a "Force Multiplier" for the current repo.

### EXECUTION PHASE 3: THE BUILD (Additive Manufacturing)
Based on Phase 1 & 2, generate **ONE** of the following strictly additive artifacts. *Choose the category that yields the highest ROI today.*

**OPTION A: THE NEW ORGAN (New Feature)**
* Draft a complete new Python module (e.g., `src/agents/crypto_arbitrage.py`).
* Must include: Class definition, Pydantic models for I/O, and a standardized "execute()" method.
* **Rule:** It must integrate with at least one existing agent.

**OPTION B: THE NEURAL PATHWAY (Integration/Refactor)**
* Write a "Bridge Script" that connects two previously isolated components (e.g., "Connect `MarketMayhem` newsletter generator to `GitHub` repo stats to correlate coding activity with stock prices").
* **Rule:** Do not delete the old code; create a `v2` wrapper or a new `Orchestrator` class.

**OPTION C: THE CORTEX EXPANSION (Test & Doc)**
* Write a "Stress Test" scenario (e.g., `tests/simulation_market_crash.py`) that feeds garbage/panic data to the agents to see if they hallucinate.
* **Rule:** The test must be self-contained and runnable via `pytest`.

### EXECUTION PHASE 4: THE MEMORY (Documentation)
* Update `CHANGELOG.md` with a "Jules' Log" entry explaining *why* this addition matters.
* If a new library was used, append it to `requirements.txt`.

---

### OUTPUT FORMAT (Actionable Code Block)
Return the result as a single, copy-pasteable Artifact:

**1. JULES' RATIONALE:**
> "I noticed we lack X. I researched Y. I have built Z to bridge this gap."

**2. FILE: path/to/new_file.py**
```python
# [Full, Executable Code Here - No Placeholders]
# [Include Docstrings & Type Hinting]
```

**3. FILE: path/to/test_new_file.py**
```python
# [Unit Test for the above]
```

**4. GIT COMMIT MESSAGE:**

> "feat(jules): implemented [Name of Feature] to expand [Context]"

### COMMAND

**Initiate Protocol ARCHITECT_INFINITE. Analyze the current state and generate today's expansion.**
"""

def generate_context_string():
    context = []
    for root, dirs, files in os.walk("."):
        if '.git' in root or 'node_modules' in root or '__pycache__' in root:
            continue
        for file in files:
            context.append(os.path.join(root, file))
    return "\n".join(context)

def call_llm(prompt, context):
    if litellm is None:
        logging.warning("litellm is not installed. Falling back to MOCK_PAYLOAD.")
        return MOCK_PAYLOAD

    models = []
    if os.environ.get("OPENAI_API_KEY"):
        models.append("gpt-4o")
    if os.environ.get("ANTHROPIC_API_KEY"):
        models.append("claude-3-5-sonnet-20241022")
    if os.environ.get("GEMINI_API_KEY"):
        models.append("gemini/gemini-1.5-pro")

    if not models:
        logging.warning("No LLM API keys found (OPENAI, ANTHROPIC, GEMINI). Falling back to MOCK_PAYLOAD.")
        return MOCK_PAYLOAD

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Context:\n{context}\n\nInitiate Protocol ARCHITECT_INFINITE."}
    ]

    for model in models:
        logging.info(f"Attempting to call LLM with model: {model}")
        try:
            response = litellm.completion(
                model=model,
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Failed to generate response using {model}: {e}")
            continue

    logging.error("All LLM attempts failed. Falling back to MOCK_PAYLOAD.")
    return MOCK_PAYLOAD

def parse_and_apply(payload):
    file_pattern = re.compile(r"\*\*\d+\.\s*FILE:\s*([^*]+)\*\*\s*```[a-zA-Z]*\n(.*?)```", re.DOTALL)
    files = file_pattern.findall(payload)

    if not files:
        logging.warning("No files matched by parser. Payload might be misformatted.")
        return

    for filepath, content in files:
        filepath = filepath.strip()
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content.strip() + "\n")
        logging.info(f"Created/Updated {filepath}")

    commit_msg_pattern = re.compile(r"\*\*4\.\s*GIT COMMIT MESSAGE:\*\*\s*>?(.*)", re.DOTALL)
    commit_msg_match = commit_msg_pattern.search(payload)
    commit_msg = "feat(jules): daily expansion"
    if commit_msg_match:
        commit_msg = commit_msg_match.group(1).strip().strip('"').strip("'")

    branch_name = f"jules/daily-build-{datetime.datetime.now().strftime('%b-%d').lower()}"

    subprocess.run(["git", "checkout", "-b", branch_name], stderr=subprocess.DEVNULL)

    for filepath, _ in files:
        subprocess.run(["git", "add", filepath.strip()])

    subprocess.run(["git", "commit", "-m", commit_msg])

def main():
    logging.info("Initiating Protocol ARCHITECT_INFINITE")
    context = generate_context_string()
    payload = call_llm(MASTER_PROMPT, context)
    parse_and_apply(payload)
    logging.info("Protocol ARCHITECT_INFINITE run complete.")

if __name__ == "__main__":
    main()
