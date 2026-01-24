import json
import random
import math
import datetime
import os

# Configuration for "Live" Snapshot (Proxy Data since real-time fetch failed)
SNAPSHOT = {
    "SPX": {"price": 5845.00, "volatility": 0.012},
    "NDX": {"price": 20150.00, "volatility": 0.015},
    "BTC": {"price": 64500.00, "volatility": 0.035},
    "GOLD": {"price": 2650.00, "volatility": 0.008},
    "OIL": {"price": 74.50, "volatility": 0.020},
    "US10Y": {"price": 4.15, "volatility": 0.010}
}

def generate_geometric_brownian_motion(start_price, days=365, drift=0.0005, volatility=0.01):
    """
    Generates a backward-looking price history ending at start_price.
    """
    prices = [start_price]
    current = start_price

    # We generate backwards
    for _ in range(days):
        shock = random.gauss(0, volatility)
        # Inverse geometric step
        prev = current / math.exp((drift - 0.5 * volatility**2) + shock)
        prices.insert(0, round(prev, 2))
        current = prev

    return prices

def generate_history():
    history = {}
    end_date = datetime.datetime.now()

    for symbol, params in SNAPSHOT.items():
        prices = generate_geometric_brownian_motion(
            params["price"],
            days=365,
            drift=0.0002, # Slight upward trend assumption
            volatility=params["volatility"]
        )

        # Format as list of {date, price}
        data_points = []
        for i, p in enumerate(prices):
            d = end_date - datetime.timedelta(days=(365 - i))
            data_points.append({
                "date": d.strftime("%Y-%m-%d"),
                "close": p,
                "volume": int(random.random() * 1000000)
            })

        history[symbol] = data_points

    # Save to disk
    output_path = os.path.join(os.path.dirname(__file__), 'generated_history.json')
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Generated history for {list(history.keys())} at {output_path}")

if __name__ == "__main__":
    generate_history()
