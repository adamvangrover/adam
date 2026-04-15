import json
import random
import os
import datetime

def generate_historical_spx(days=100, start_val=4800, volatility=0.015):
    data = []
    current = start_val
    for i in range(days):
        change = current * random.gauss(0.0005, volatility)
        current += change
        data.append(round(current, 2))
    return data

def generate_forward_cones(base_data, days=30):
    last_val = base_data[-1]
    base_cone = [last_val]
    bull_cone = [last_val]
    bear_cone = [last_val]

    for i in range(1, days + 1):
        # Base case: slight upward drift
        base_val = base_cone[-1] * (1 + random.gauss(0.0005, 0.005))
        base_cone.append(round(base_val, 2))

        # Bull case: higher upward drift, higher vol
        bull_val = bull_cone[-1] * (1 + random.gauss(0.002, 0.01))
        bull_cone.append(round(bull_val, 2))

        # Bear case: negative drift, high vol
        bear_val = bear_cone[-1] * (1 + random.gauss(-0.0015, 0.015))
        bear_cone.append(round(bear_val, 2))

    return base_cone, bull_cone, bear_cone

def main():
    historical_days = 90
    future_days = 30

    today = datetime.date.today()
    labels = [(today - datetime.timedelta(days=historical_days - i)).strftime("%Y-%m-%d") for i in range(historical_days)]
    labels += [(today + datetime.timedelta(days=i)).strftime("%Y-%m-%d") for i in range(future_days + 1)]

    spx_hist = generate_historical_spx(historical_days, 5000, 0.01)

    base_c, bull_c, bear_c = generate_forward_cones(spx_hist, future_days)

    # Pad historical data to match forward cone lengths for Chart.js
    spx_hist_padded = spx_hist + [None] * future_days

    # Pad cones with None for historical period
    base_c_padded = [None] * (historical_days - 1) + base_c
    bull_c_padded = [None] * (historical_days - 1) + bull_c
    bear_c_padded = [None] * (historical_days - 1) + bear_c

    data = {
        "synthesis": {
            "narrative": "System detects an imminent volatility expansion pattern across major indices. Credit spreads are tightening to historic lows while VIX remains artificially suppressed. Base case assumes a continuation of the rally, but tail risks (Bear Cone) are elevated due to systemic leverage.",
            "sentiment": "neutral-bullish",
            "early_warning_indicators": [
                "VIX compression < 13",
                "High Yield spreads < 300bps",
                "Tech concentration > 35%"
            ]
        },
        "outlook": {
            "labels": labels,
            "historical_spx": spx_hist_padded,
            "base_cone": base_c_padded,
            "bull_cone": bull_c_padded,
            "bear_cone": bear_c_padded
        },
        "portfolio": [
            {"asset": "S&P 500 (SPY)", "pos": 15.5, "conviction": 85, "status": "HOLD"},
            {"asset": "US 10Y Treasury (IEF)", "pos": -2.1, "conviction": 40, "status": "REDUCE"},
            {"asset": "Gold (GLD)", "pos": 5.4, "conviction": 70, "status": "ACCUMULATE"},
            {"asset": "Bitcoin (BTC)", "pos": 12.0, "conviction": 90, "status": "OVERWEIGHT"},
            {"asset": "Broadly Syndicated Loans", "pos": 8.5, "conviction": 75, "status": "HOLD"}
        ]
    }

    os.makedirs("showcase/data", exist_ok=True)
    with open("showcase/data/market_mayhem_data.json", "w") as f:
        json.dump(data, f, indent=4)

    print("Generated market_mayhem_data.json")

if __name__ == "__main__":
    main()