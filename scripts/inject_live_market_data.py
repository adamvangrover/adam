import json
import random
import datetime
import math
import os

DATA_FILE = "showcase/data/sp500_market_data.json"
LIVE_FILE = "showcase/data/market_live.json"

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return []
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def simulate_market_move(price, volatility=0.02):
    change = price * random.normalvariate(0, volatility)
    return price + change

def inject_live_data():
    data = load_data()
    if not data:
        return

    live_snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "market_status": "OPEN",
        "tickers": {}
    }

    print(f"Injecting live data for {len(data)} tickers...")

    for item in data:
        ticker = item["ticker"]
        history = item["price_history"]
        last_price = history[-1] if history else item["current_price"]

        # Simulate 5-10 new data points representing intraday movement
        num_points = random.randint(5, 10)
        new_prices = []
        current_p = last_price

        for _ in range(num_points):
            current_p = simulate_market_move(current_p, volatility=0.005) # Low vol for intraday
            new_prices.append(round(current_p, 2))

        # Append to history (keep history manageable, e.g., last 500 points)
        item["price_history"].extend(new_prices)
        if len(item["price_history"]) > 500:
            item["price_history"] = item["price_history"][-500:]

        # Update current stats
        final_price = new_prices[-1]
        start_price = history[0] if history else final_price # Or previous close

        # Calculate change based on the very first point in history vs now, or just last move?
        # Typically change_pct is vs previous day close. Let's assume history[-num_points-1] was prev close.
        # For simplicity, let's just calculate change vs the price before this injection

        prev_price = last_price
        change = final_price - prev_price
        change_pct = (change / prev_price) * 100

        item["current_price"] = final_price
        item["change_pct"] = round(change_pct, 2)

        live_snapshot["tickers"][ticker] = {
            "price": final_price,
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "volume": random.randint(1000, 50000) # Simulated volume
        }

    # Save updated full data
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    # Save live snapshot
    with open(LIVE_FILE, "w") as f:
        json.dump(live_snapshot, f, indent=2)

    print(f"Successfully updated {DATA_FILE} and created {LIVE_FILE}")

if __name__ == "__main__":
    inject_live_data()
