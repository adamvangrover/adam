"""
Script: Universal Market Snapshot Generator
===========================================

Purpose:
  1. Universal External Pull: Simulates fetching high-fidelity market data (Ticks, Order Books, News).
  2. Time Series Snapshot: Generates a JSONL file with sequential events for fine-tuning or replay.
  3. Static Showcase: Exports a aggregated state to `window.ADAM_STATE` JSON for the frontend.

Usage:
  python scripts/generate_market_snapshot.py --symbols AAPL BTC-USD ES=F --duration 1h --interval 1s

"""

import argparse
import json
import random
import time
import os
from datetime import datetime, timedelta
import math
from typing import List, Dict, Any
from enum import Enum

# Constants
OUTPUT_JSONL = "data/snapshots/market_snapshot_v1.jsonl"
SHOWCASE_DATA = "showcase/data/market_state.json"
SHOWCASE_JS = "showcase/js/repo_data.js" # We will append/overwrite this

class EventType(Enum):
    PRICE_TICK = "PRICE_TICK"
    ORDER_BOOK_UPDATE = "ORDER_BOOK_UPDATE"
    NEWS_HEADLINE = "NEWS_HEADLINE"
    SYSTEM_ALERT = "SYSTEM_ALERT"

class SyntheticMarketSource:
    """
    Generates realistic-looking financial data.
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.state = {
            s: {
                "price": 150.0 if "AAPL" in s else (4000.0 if "ES" in s else 30000.0),
                "volatility": 0.0001,
                "trend": 0.0
            } for s in symbols
        }

    def generate_tick(self, timestamp: datetime) -> List[Dict]:
        events = []
        for symbol, data in self.state.items():
            # Random Walk
            shock = random.normalvariate(0, 1)
            drift = data["trend"]
            change = data["price"] * (drift + data["volatility"] * shock)
            new_price = data["price"] + change
            data["price"] = new_price

            # Create Tick Event
            events.append({
                "type": EventType.PRICE_TICK.value,
                "timestamp": timestamp.isoformat(),
                "symbol": symbol,
                "data": {
                    "price": round(new_price, 2),
                    "volume": random.randint(100, 5000)
                }
            })

            # Occasionally update order book (simulated L2)
            if random.random() < 0.2:
                spread = new_price * 0.0005
                events.append({
                    "type": EventType.ORDER_BOOK_UPDATE.value,
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "data": {
                        "bids": [
                            {"price": round(new_price - spread * 1, 2), "size": random.randint(100, 1000)},
                            {"price": round(new_price - spread * 2, 2), "size": random.randint(500, 2000)},
                            {"price": round(new_price - spread * 3, 2), "size": random.randint(1000, 5000)},
                        ],
                        "asks": [
                            {"price": round(new_price + spread * 1, 2), "size": random.randint(100, 1000)},
                            {"price": round(new_price + spread * 2, 2), "size": random.randint(500, 2000)},
                            {"price": round(new_price + spread * 3, 2), "size": random.randint(1000, 5000)},
                        ]
                    }
                })

        return events

class NewsGenerator:
    """Generates synthetic financial news."""
    TEMPLATES = [
        "{symbol} beats earnings expectations by {percent}%.",
        "Regulatory concerns mount for {symbol} amid new probe.",
        "Analyst upgrades {symbol} to 'Overweight'.",
        "Market volatility spikes ahead of Fed meeting.",
        "{symbol} announces strategic partnership for AI expansion."
    ]

    def generate(self, symbol: str, timestamp: datetime) -> Dict:
        text = random.choice(self.TEMPLATES).format(
            symbol=symbol,
            percent=random.randint(5, 20)
        )
        sentiment = "POSITIVE" if "beats" in text or "upgrades" in text else "NEGATIVE"
        if "volatility" in text: sentiment = "NEUTRAL"

        return {
            "type": EventType.NEWS_HEADLINE.value,
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "data": {
                "headline": text,
                "sentiment": sentiment,
                "source": "FlashWire"
            }
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "BTC-USD", "ES=F"])
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    market = SyntheticMarketSource(args.symbols)
    news_gen = NewsGenerator()

    start_time = datetime.now() - timedelta(hours=24)
    current_time = start_time

    all_events = []

    print(f"Generating snapshot for {args.symbols} ({args.steps} steps)...")

    # 1. Generate Time Series
    with open(OUTPUT_JSONL, 'w') as f:
        for i in range(args.steps):
            # Market Ticks
            ticks = market.generate_tick(current_time)
            for tick in ticks:
                f.write(json.dumps(tick) + "\n")
                all_events.append(tick)

            # Random News
            if random.random() < 0.05:
                symbol = random.choice(args.symbols)
                news = news_gen.generate(symbol, current_time)
                f.write(json.dumps(news) + "\n")
                all_events.append(news)

            current_time += timedelta(seconds=15) # 15s interval

    print(f"Saved time series to {OUTPUT_JSONL}")

    # 2. Generate Static Showcase State
    # We aggregate the final state and some history

    showcase_state = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "description": "Synthetic Market State for Adam v23.5 Showcase"
        },
        "market_data": {},
        "news_feed": []
    }

    # Populate latest state
    for symbol in args.symbols:
        # Get history
        history = [e for e in all_events if e["symbol"] == symbol and e["type"] == EventType.PRICE_TICK.value]
        last_event = history[-1] if history else None

        if last_event:
            showcase_state["market_data"][symbol] = {
                "price": last_event["data"]["price"],
                "volume": last_event["data"]["volume"],
                "change_pct": round(((last_event["data"]["price"] - history[0]["data"]["price"]) / history[0]["data"]["price"]) * 100, 2),
                "history": [h["data"]["price"] for h in history[-50:]] # Last 50 points for sparklines
            }

    # Populate News
    news_events = [e for e in all_events if e["type"] == EventType.NEWS_HEADLINE.value]
    showcase_state["news_feed"] = [n["data"] for n in news_events[-10:]] # Last 10 news items

    # Write to JSON
    with open(SHOWCASE_DATA, 'w') as f:
        json.dump(showcase_state, f, indent=2)
    print(f"Saved showcase state to {SHOWCASE_DATA}")

    # Write to JS for easy frontend loading (optional, but requested by pattern)
    js_content = f"window.MARKET_SNAPSHOT = {json.dumps(showcase_state, indent=2)};"
    with open("showcase/js/market_snapshot.js", "w") as f:
        f.write(js_content)
    print("Saved showcase JS adapter to showcase/js/market_snapshot.js")

if __name__ == "__main__":
    main()
