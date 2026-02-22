import json
import os

DATA_FILE = "showcase/data/sp500_market_data.json"

UPDATES = {
    "AAPL": 230.00,
    "MSFT": 420.00,
    "NVDA": 140.00,
    "AMZN": 200.00,
    "GOOGL": 175.00,
    "META": 580.00,
    "TSLA": 320.00
}

def update_baseline():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return

    with open(DATA_FILE, "r") as f:
        data = json.load(f)

    for item in data:
        ticker = item.get("ticker")
        if ticker in UPDATES:
            new_price = UPDATES[ticker]
            old_price = item.get("current_price", 0)

            # Special handling for NVDA split if old price is high
            if ticker == "NVDA" and old_price > 500:
                print(f"Applying 10:1 split adjustment for {ticker} history...")
                item["price_history"] = [p / 10.0 for p in item["price_history"]]
                old_price = old_price / 10.0

            item["current_price"] = new_price
            item["price_history"].append(new_price)

            # Recalculate change_pct based on new price vs previous
            prev_price = item["price_history"][-2] if len(item["price_history"]) > 1 else old_price
            change = new_price - prev_price
            change_pct = (change / prev_price) * 100
            item["change_pct"] = round(change_pct, 2)

            print(f"Updated {ticker}: {old_price} -> {new_price}")

    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Successfully updated {DATA_FILE}")

if __name__ == "__main__":
    update_baseline()
