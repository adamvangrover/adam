"""
Script: Universal Market Snapshot Generator (Market Mayhem Edition)
=================================================================

Purpose:
  1. Universal External Pull: Simulates fetching high-fidelity market data (Ticks, Order Books, News).
  2. Time Series Snapshot: Generates a JSONL file with sequential events for fine-tuning or replay.
  3. Static Showcase: Exports a aggregated state to `window.ADAM_STATE` JSON for the frontend.

Usage:
  python scripts/generate_market_snapshot.py --duration 1h --interval 1s

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
SHOWCASE_JS = "showcase/js/repo_data.js"

# Market Mayhem Data (Dec 12, 2025)
MARKET_DATA = {
    "ES=F": {"price": 6827.41, "trend": -0.0001}, # S&P 500
    "YM=F": {"price": 48458.05, "trend": 0.0002}, # Dow Jones
    "NQ=F": {"price": 23195.17, "trend": -0.0003}, # Nasdaq 100
    "CL=F": {"price": 57.52, "trend": -0.0004},   # Crude Oil
    "GC=F": {"price": 4313.56, "trend": 0.0001},  # Gold
    "BTC-USD": {"price": 92400.00, "trend": 0.0005}, # Bitcoin
    "WMT": {"price": 185.20, "trend": 0.0001},     # Walmart (Guessing price based on context)
    "BK": {"price": 95.00, "trend": 0.0002},       # BNY Mellon
    "CRML": {"price": 32.50, "trend": 0.0005},     # Critical Metals Corp
    "VOLT": {"price": 165.00, "trend": 0.0003},    # Volta Motors (Approaching target 185)
    "AWAV": {"price": 380.00, "trend": 0.0004},    # AlphaWave (Approaching target 420)
    "PLTR": {"price": 58.00, "trend": 0.0003},     # Palantir
    "NFLX": {"price": 950.00, "trend": -0.0001},   # Netflix
    "WBD": {"price": 12.50, "trend": 0.0020},      # Warner Bros Discovery
    "IBM": {"price": 210.00, "trend": 0.0002},     # IBM
    "LULU": {"price": 420.00, "trend": 0.0010},    # Lululemon
    "AVGO": {"price": 1400.00, "trend": -0.0020},  # Broadcom
    "NVDA": {"price": 1100.00, "trend": -0.0010},  # NVIDIA
    "AMD": {"price": 180.00, "trend": -0.0010},    # AMD
    "LEN": {"price": 190.00, "trend": 0.0001},     # Lennar
    "FDX": {"price": 310.00, "trend": 0.0000},     # FedEx
    "NKE": {"price": 85.00, "trend": 0.0000},      # Nike
}

NEWS_HEADLINES = [
    {"headline": "Fed Cuts Rates to 3.50%-3.75%, signaling potential pause in 2026.", "sentiment": "NEUTRAL", "source": "Central Bank Wire"},
    {"headline": "Broadcom plunges 11% on AI chip margin warning, dragging down semi sector.", "sentiment": "NEGATIVE", "source": "MarketWatch"},
    {"headline": "Netflix to acquire Warner Bros. Discovery for ~$82B in mega-merger.", "sentiment": "POSITIVE", "source": "DealBook"},
    {"headline": "IBM acquires Confluent for $11B to bolster Watsonx AI stack.", "sentiment": "POSITIVE", "source": "TechCrunch"},
    {"headline": "Lululemon jumps 10% on raised outlook; high-end consumer remains resilient.", "sentiment": "POSITIVE", "source": "Retail Dive"},
    {"headline": "Gold smashes through $4,300 as investors seek hedge against policy uncertainty.", "sentiment": "POSITIVE", "source": "Commodities Weekly"},
    {"headline": "Bitcoin surges past $92k as crypto rally continues.", "sentiment": "POSITIVE", "source": "CryptoDesk"},
    {"headline": "Walmart unveils 'Trend-to-Product' Agentic AI system for inventory optimization.", "sentiment": "POSITIVE", "source": "TechCrunch"},
    {"headline": "BNY Mellon integrates Gemini Enterprise into 'Risk Intelligence Core'.", "sentiment": "POSITIVE", "source": "FinTech News"},
    {"headline": "Copper diverges from Tech rally, signaling physical economy softness.", "sentiment": "NEGATIVE", "source": "Macro Insights"}
]

class EventType(Enum):
    PRICE_TICK = "PRICE_TICK"
    ORDER_BOOK_UPDATE = "ORDER_BOOK_UPDATE"
    NEWS_HEADLINE = "NEWS_HEADLINE"
    SYSTEM_ALERT = "SYSTEM_ALERT"

class SyntheticMarketSource:
    """
    Generates realistic-looking financial data based on Market Mayhem scenario.
    """
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.state = {}
        for s in symbols:
            base_data = MARKET_DATA.get(s, {"price": 100.0, "trend": 0.0})
            self.state[s] = {
                "price": base_data["price"],
                "volatility": 0.0002,
                "trend": base_data["trend"]
            }

    def generate_tick(self, timestamp: datetime) -> List[Dict]:
        events = []
        for symbol, data in self.state.items():
            # Random Walk with Drift
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

    def __init__(self):
        self.headlines = NEWS_HEADLINES
        self.index = 0

    def generate(self, symbol: str, timestamp: datetime) -> Dict:
        # Round robin through the defined headlines
        item = self.headlines[self.index % len(self.headlines)]
        self.index += 1

        return {
            "type": EventType.NEWS_HEADLINE.value,
            "timestamp": timestamp.isoformat(),
            "symbol": symbol, # Note: Headline might not match symbol strictly, but for mock stream it's fine
            "data": {
                "headline": item["headline"],
                "sentiment": item["sentiment"],
                "source": item["source"]
            }
        }

def main():
    parser = argparse.ArgumentParser()
    # Default to all symbols in our dataset
    default_symbols = list(MARKET_DATA.keys())
    parser.add_argument("--symbols", nargs="+", default=default_symbols)
    parser.add_argument("--steps", type=int, default=200) # Increased steps for better history
    args = parser.parse_args()

    market = SyntheticMarketSource(args.symbols)
    news_gen = NewsGenerator()

    # Set time to Market Mayhem date: Dec 12, 2025
    start_time = datetime(2025, 12, 11, 9, 30, 0)
    current_time = start_time

    all_events = []

    print(f"Generating snapshot for {len(args.symbols)} symbols ({args.steps} steps)...")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    os.makedirs(os.path.dirname(SHOWCASE_DATA), exist_ok=True)
    os.makedirs(os.path.dirname(SHOWCASE_JS), exist_ok=True)

    # 1. Generate Time Series
    with open(OUTPUT_JSONL, 'w') as f:
        for i in range(args.steps):
            # Market Ticks
            ticks = market.generate_tick(current_time)
            for tick in ticks:
                f.write(json.dumps(tick) + "\n")
                all_events.append(tick)

            # Inject News (Higher frequency for showcase)
            if random.random() < 0.1:
                symbol = random.choice(args.symbols)
                news = news_gen.generate(symbol, current_time)
                f.write(json.dumps(news) + "\n")
                all_events.append(news)

            current_time += timedelta(minutes=5) # 5 min interval for wider range

    print(f"Saved time series to {OUTPUT_JSONL}")

    # 2. Generate Static Showcase State
    # We aggregate the final state and some history

    showcase_state = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "description": "Real-Time Market State: Market Mayhem Dec 2025"
        },
        "market_data": {},
        "news_feed": []
    }

    # Populate latest state
    for symbol in args.symbols:
        # Get history
        history = [e for e in all_events if e["symbol"] == symbol and e["type"] == EventType.PRICE_TICK.value]
        if history:
            last_event = history[-1]
            first_event = history[0]

            showcase_state["market_data"][symbol] = {
                "price": last_event["data"]["price"],
                "volume": last_event["data"]["volume"],
                "change_pct": round(((last_event["data"]["price"] - first_event["data"]["price"]) / first_event["data"]["price"]) * 100, 2),
                "history": [h["data"]["price"] for h in history[-50:]] # Last 50 points for sparklines
            }

    # Populate News - grab the unique ones generated
    news_events = [e for e in all_events if e["type"] == EventType.NEWS_HEADLINE.value]
    # Deduplicate by headline
    seen_headlines = set()
    unique_news = []
    for n in news_events:
        if n["data"]["headline"] not in seen_headlines:
            unique_news.append(n["data"])
            seen_headlines.add(n["data"]["headline"])

    showcase_state["news_feed"] = unique_news[-15:] # Last 15 unique items

    # Write to JSON
    with open(SHOWCASE_DATA, 'w') as f:
        json.dump(showcase_state, f, indent=2)
    print(f"Saved showcase state to {SHOWCASE_DATA}")

    # Write to JS for easy frontend loading
    js_content = f"window.MARKET_SNAPSHOT = {json.dumps(showcase_state, indent=2)};"
    with open("showcase/js/market_snapshot.js", "w") as f:
        f.write(js_content)
    print("Saved showcase JS adapter to showcase/js/market_snapshot.js")

if __name__ == "__main__":
    main()
