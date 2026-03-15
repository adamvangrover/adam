import json
import os
import random
import math
from datetime import datetime, timedelta

# Define directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'market_mayhem_data.json')

def simulate_historic_sp500(start_date, end_date, start_val=2100.0, trend=0.04, vol=0.15):
    """Simulate some generic but somewhat realistic looking price data for historic years."""
    data = []
    current_date = start_date
    current_val = start_val
    
    while current_date <= end_date:
        # Ignore weekends
        if current_date.weekday() < 5:
            # Random walk with drift
            daily_drift = trend / 252
            daily_vol = vol / math.sqrt(252)
            
            shock = random.gauss(0, 1)
            
            # Simple jump diffusion for market crashes (e.g. 2020)
            if current_date.year == 2020 and current_date.month in [2, 3]:
                shock -= 2.0  # Big drops
                daily_vol *= 3
            elif current_date.year == 2022:
                daily_drift = -0.15 / 252 # Bear market
            else:
                daily_drift = trend / 252
                daily_vol = vol / math.sqrt(252)
            
            return_val = daily_drift + daily_vol * shock
            current_val = current_val * math.exp(return_val)
            
            # Monthly tracking for smaller payload
            if current_date.day == 15 or current_date == end_date:
                data.append({
                    "date": current_date.strftime("%Y-%m-%d"),
                    "value": round(current_val, 2)
                })
                
        current_date += timedelta(days=1)
        
    return data

def simulate_forward_cones(last_price, current_date, projection_years=3):
    """Generate forward projections for the chart."""
    base_case = []
    bull_case = []
    bear_case = []

    # Projection params
    base_return = 0.08
    bull_return = 0.15
    bear_return = -0.05
    volatility = 0.16

    end_date = current_date + timedelta(days=365 * projection_years)
    dt = current_date
    
    base_val = last_price
    bull_val = last_price
    bear_val = last_price
    
    days_passed = 0
    total_days = 365 * projection_years

    while dt <= end_date:
        if dt.weekday() < 5:
            # Deterministic paths over time + slight noise
            year_frac = days_passed / 365.0
            
            base_val = last_price * math.exp(base_return * year_frac) + random.gauss(0, last_price * 0.01)
            
            # Cones widen over time
            cone_width = last_price * volatility * math.sqrt(year_frac)
            
            bull_val = base_val + cone_width + (last_price * (bull_return - base_return) * year_frac)
            bear_val = base_val - cone_width + (last_price * (bear_return - base_return) * year_frac)

            # Sample quarterly for charting
            if dt.day == 1 and dt.month in [1, 4, 7, 10]:
                date_str = dt.strftime("%Y-%m-%d")
                base_case.append({"date": date_str, "value": round(base_val, 2)})
                bull_case.append({"date": date_str, "value": round(bull_val, 2)})
                bear_case.append({"date": date_str, "value": round(bear_val, 2)})
                
        dt += timedelta(days=1)
        days_passed += 1

    return base_case, bull_case, bear_case

def generate_market_data():
    """Generates the primary dataset used by the Market Mayhem dashboard."""
    print("Generating simulated market data and ML outlooks...")
    
    # Establish Timeline
    now = datetime(2026, 3, 14) # Context of Adam v26
    start_history = datetime(2016, 1, 1)

    historic_sp500 = simulate_historic_sp500(start_history, now, start_val=2043.94)
    last_close = historic_sp500[-1]['value']
    
    base, bull, bear = simulate_forward_cones(last_close, now)

    payload = {
        "metadata": {
            "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_version": "v26.1-qmc",
            "base_index": "S&P 500"
        },
        "indicators": {
            "oil_volatility": {"status": "Red Alert", "value": 85.4, "trend": "up"},
            "bond_yield_spread": {"status": "Green", "value": 0.45, "trend": "flat"},
            "supply_chain_disruption": {"status": "Amber", "value": 62.1, "trend": "down"}
        },
        "synthesis": {
            "status": "NEUTRAL",
            "active_monitoring": True,
            "narrative": "New, current-date AI synthesis narrative indicates economic growing at an appropriate rate. All market indicators show information throughput at optimal levels. Supreme consensus counts credit capabilities adjusting to their remote-role in US synthesis, and extreme outliers have a risk premium adverse to current fixed income yields. Positioning should remain neutral with high liquidity buffers.",
        },
        "portfolio": [
            {"asset": "MSFT", "size": "1.0K", "entry": 26.00, "current": 14.35, "pnl_pct": 414.74, "risk": "high"},
            {"asset": "META", "size": "1.3K", "entry": 28.00, "current": 59.43, "pnl_pct": 59.43, "risk": "high"},
            {"asset": "AVGO", "size": "2.3K", "entry": 27.00, "current": 2.03, "pnl_pct": -0.35, "risk": "low"},
            {"asset": "ABT", "size": "1.0K", "entry": 26.00, "current": 38.33, "pnl_pct": 108.82, "risk": "low"}
        ],
        "chart_data": {
            "historic_sp500": historic_sp500,
            "forward_base": base,
            "forward_bull": bull,
            "forward_bear": bear,
            "key_spikes": [
                {"date": "2020-03-15", "label": "COVID Crash", "value": 2400},
                {"date": "2022-10-15", "label": "Rate Shock Low", "value": 3500},
                {"date": "2024-11-05", "label": "US Election", "value": 5800}
            ]
        },
        "sentiment_index": {
            "value": 26, # 0-100 where < 20 is extreme fear, > 80 extreme greed
            "label": "Fear",
            "active": True
        }
    }

    # Save to JSON
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
        
    print(f"Market Data successfully saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_market_data()
