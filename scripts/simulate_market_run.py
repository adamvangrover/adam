import json
import random
import datetime
import os
import math

OUTPUT_DIR = "showcase/data"
HTML_OUTPUT = "showcase/simulation_dashboard.html"

def generate_simulation_data(days=30):
    start_date = datetime.date.today() - datetime.timedelta(days=days)

    data = {
        "metadata": {
            "simulation_id": f"SIM-{random.randint(1000,9999)}",
            "start_date": start_date.isoformat(),
            "end_date": datetime.date.today().isoformat(),
            "scenario": "Tech Recovery & Volatility",
            "agent_version": "v23.5"
        },
        "timeline": []
    }

    spx = 5200.0
    ndx = 18000.0
    portfolio = 1000000.0
    cash = 200000.0 # Starting cash
    holdings = {"NVDA": 100, "MSFT": 200, "AAPL": 300} # Mock initial holdings

    sentiment_base = 50.0

    for i in range(days):
        date = start_date + datetime.timedelta(days=i)

        # Simulate Market Movement (Random Walk with Drift)
        spx_change = random.gauss(0.0005, 0.01) # Mean 0.05%, Std 1%
        ndx_change = random.gauss(0.0008, 0.015) # Higher volatility for tech

        spx *= (1 + spx_change)
        ndx *= (1 + ndx_change)

        # Simulate Sentiment
        sentiment_noise = random.gauss(0, 5)
        sentiment_base += random.gauss(0, 2)
        sentiment = max(0, min(100, sentiment_base + sentiment_noise))

        # Agent Decision Logic (Simple Mock)
        actions = []
        if sentiment > 70 and cash > 50000:
            # Bullish: Buy
            symbol = random.choice(list(holdings.keys()))
            qty = random.randint(10, 50)
            price = random.uniform(100, 1000) # Mock price
            cost = qty * price
            if cash >= cost:
                cash -= cost
                holdings[symbol] += qty
                actions.append(f"BUY {qty} {symbol} @ ${price:.2f} (Sentiment High)")
        elif sentiment < 30:
            # Bearish: Sell
            symbol = random.choice(list(holdings.keys()))
            if holdings[symbol] > 0:
                qty = random.randint(1, int(holdings[symbol] * 0.5))
                price = random.uniform(100, 1000)
                proceeds = qty * price
                cash += proceeds
                holdings[symbol] -= qty
                actions.append(f"SELL {qty} {symbol} @ ${price:.2f} (Sentiment Low)")

        # Calculate Portfolio Value (Mocked based on index correlation + cash)
        # Using NDX as proxy for holding value change
        portfolio_holdings_val = (portfolio - 200000) * (1 + ndx_change * 1.2) # Beta 1.2
        portfolio = portfolio_holdings_val + 200000 # Keep initial cash constant for this simple math, adjusted by simulated trades is hard without real prices.
        # Let's just simulate portfolio value directly for the chart
        portfolio = portfolio * (1 + (ndx_change * 1.1) + random.gauss(0, 0.002))

        day_data = {
            "date": date.isoformat(),
            "spx": round(spx, 2),
            "ndx": round(ndx, 2),
            "portfolio_value": round(portfolio, 2),
            "sentiment": round(sentiment, 2),
            "actions": actions,
            "system_status": "OPTIMAL" if sentiment > 40 else "WARNING"
        }
        data["timeline"].append(day_data)

    return data

def generate_html(data):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ADAM v23.5 :: SIMULATION RUN</title>
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="css/market_mayhem_tiers.css">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="js/nav.js" defer></script>
    <style>
        .sim-container { display: grid; grid-template-columns: 300px 1fr; gap: 20px; padding: 20px; height: calc(100vh - 60px); }
        .sim-sidebar { background: rgba(0,0,0,0.3); border-right: 1px solid #333; padding: 20px; overflow-y: auto; }
        .sim-main { overflow-y: auto; }
        .sim-log-item { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin-bottom: 5px; color: #aaa; border-bottom: 1px solid #222; padding-bottom: 5px; }
        .sim-log-date { color: #00f3ff; }
        .sim-log-action { color: #fff; }
        .chart-wrapper { background: rgba(0,0,0,0.5); border: 1px solid #333; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .stat-card { background: rgba(0, 243, 255, 0.05); border: 1px solid #00f3ff; padding: 15px; margin-bottom: 10px; border-radius: 4px; }
        .stat-label { font-size: 0.7rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
        .stat-value { font-size: 1.5rem; font-family: 'JetBrains Mono'; color: #fff; }
        .controls { display: flex; gap: 10px; margin-bottom: 20px; }
        .cyber-btn-small { background: transparent; border: 1px solid #00f3ff; color: #00f3ff; padding: 5px 10px; cursor: pointer; font-family: 'JetBrains Mono'; font-size: 0.8rem; transition: all 0.2s; }
        .cyber-btn-small:hover { background: rgba(0, 243, 255, 0.1); }
    </style>
</head>
<body class="tier-high">
    <div class="doc-header-ui" style="display: flex; justify-content: space-between; align-items: center; padding: 10px 20px; background: #050b14; border-bottom: 1px solid #333;">
        <span style="color: #00f3ff; font-family: 'JetBrains Mono';">ADAM v23.5 // SIMULATION ENGINE</span>
        <div>
            <a href="market_mayhem_archive.html" class="cyber-btn-small" style="text-decoration:none;">EXIT SIMULATION</a>
        </div>
    </div>

    <div class="sim-container">
        <aside class="sim-sidebar">
            <h3 style="color: #fff; font-family: 'JetBrains Mono'; border-bottom: 1px solid #00f3ff; padding-bottom: 10px;">RUN LOG</h3>
            <div id="simulation-log">
                <!-- Logs injected via JS -->
            </div>
        </aside>

        <main class="sim-main">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h2 style="margin: 0;">SIMULATION DASHBOARD: <span id="sim-scenario-name">LOADING...</span></h2>
                <div class="controls">
                    <button class="cyber-btn-small" onclick="startReplay()"><i class="fas fa-play"></i> REPLAY</button>
                    <button class="cyber-btn-small" onclick="pauseReplay()"><i class="fas fa-pause"></i> PAUSE</button>
                    <button class="cyber-btn-small" onclick="resetReplay()"><i class="fas fa-undo"></i> RESET</button>
                </div>
            </div>

            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;">
                <div class="stat-card">
                    <div class="stat-label">Current Date</div>
                    <div class="stat-value" id="stat-date">--</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Portfolio Value</div>
                    <div class="stat-value" id="stat-portfolio">--</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Sentiment</div>
                    <div class="stat-value" id="stat-sentiment">--</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">S&P 500</div>
                    <div class="stat-value" id="stat-spx">--</div>
                </div>
            </div>

            <div class="chart-wrapper">
                <canvas id="performanceChart"></canvas>
            </div>

            <div class="chart-wrapper">
                <canvas id="sentimentChart"></canvas>
            </div>
        </main>
    </div>

    <!-- Data Injection -->
    <script id="sim-data" type="application/json">
        __SIM_DATA_PLACEHOLDER__
    </script>

    <script src="js/simulation_viewer.js"></script>
</body>
</html>
"""
    return html.replace("__SIM_DATA_PLACEHOLDER__", json.dumps(data))

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Generating simulation data...")
    data = generate_simulation_data()

    json_path = os.path.join(OUTPUT_DIR, "simulation_run_data.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Data saved to {json_path}")

    print("Generating simulation dashboard...")
    html_content = generate_html(data)
    with open(HTML_OUTPUT, 'w') as f:
        f.write(html_content)
    print(f"Dashboard saved to {HTML_OUTPUT}")

if __name__ == "__main__":
    main()
