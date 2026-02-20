import json
import os
import random
import datetime
import uuid
import hashlib

DATA_DIR = 'showcase/data'
PORTFOLIO_DIR = os.path.join(DATA_DIR, 'portfolios')
SCENARIO_DIR = os.path.join(DATA_DIR, 'market_scenarios')

def generate_investment_ecosystem():
    # Load market data
    with open(os.path.join(DATA_DIR, 'sp500_market_data.json'), 'r') as f:
        market_data = json.load(f)

    # --- Generate Portfolios ---
    print("Generating Portfolios...")

    # 1. Global Macro Fund
    generate_portfolio("Global_Macro_Opportunities_Fund", market_data,
                       strategy="Balanced", sector_bias=None,
                       aum=5000000000)

    # 2. Tech Disruption Fund
    generate_portfolio("AI_Disruption_Alpha_Fund", market_data,
                       strategy="Aggressive Growth", sector_bias="Technology",
                       aum=1500000000)

    # 3. Income & Dividend Fund
    generate_portfolio("Secure_Income_Yield_Fund", market_data,
                       strategy="Conservative", sector_bias="Consumer Staples",
                       aum=3200000000)

    # --- Generate Scenarios ---
    print("Generating Scenarios...")

    # 1. Yield Curve Inversion / Hard Landing
    generate_scenario("Hard_Landing_2026", "Yield Curve Inversion & Consumer Credit Crunch",
                      factors={"Technology": -0.25, "Financials": -0.15, "Consumer Discretionary": -0.20, "Utilities": 0.05, "Consumer Staples": -0.05})

    # 2. AI Boom Continued / Soft Landing
    generate_scenario("AI_Supercycle_2026", "Productivity Boom & Falling Inflation",
                      factors={"Technology": 0.30, "Financials": 0.10, "Consumer Discretionary": 0.15, "Energy": 0.05})

    # 3. Oil Shock / Stagflation
    generate_scenario("Stagflation_Shock", "Geopolitical Crisis & Supply Chain Disruption",
                      factors={"Energy": 0.40, "Industrials": -0.10, "Technology": -0.15, "Consumer Discretionary": -0.25})

    print("Investment Ecosystem generated.")

def generate_portfolio(name, market_data, strategy, sector_bias, aum):
    holdings = []
    cash = aum * random.uniform(0.02, 0.10) # 2-10% Cash
    invested_capital = aum - cash

    # Filter universe
    if sector_bias:
        universe = [x for x in market_data if x['sector'] == sector_bias]
        # Fallback if too small
        if len(universe) < 5: universe = market_data
    else:
        universe = market_data

    # Pick 10-20 stocks (or max available)
    max_count = len(universe)
    if max_count < 10:
        count = max_count
    else:
        count = random.randint(10, min(20, max_count))

    picks = random.sample(universe, count)

    # Allocation
    total_weight = 0
    weights = []
    for _ in range(count):
        w = random.uniform(1, 10)
        weights.append(w)
        total_weight += w

    current_value = 0

    for i, stock in enumerate(picks):
        weight = weights[i] / total_weight
        allocation = invested_capital * weight
        qty = int(allocation / stock['current_price'])
        cost_basis = stock['current_price'] * random.uniform(0.8, 1.1) # Bought somewhere within 20% range

        pos_val = qty * stock['current_price']
        pnl = pos_val - (qty * cost_basis)
        pnl_pct = (pnl / (qty * cost_basis))

        holdings.append({
            "ticker": stock['ticker'],
            "name": stock['name'],
            "sector": stock['sector'],
            "quantity": qty,
            "avg_cost": round(cost_basis, 2),
            "current_price": stock['current_price'],
            "market_value": round(pos_val, 2),
            "pnl": round(pnl, 2),
            "pnl_pct": round(pnl_pct, 4),
            "weight": round(weight, 4)
        })
        current_value += pos_val

    # Audit Trail
    audit = {
        "fund_manager": f"PM-{random.randint(100,999)}",
        "compliance_officer": f"COMP-{random.randint(100,999)}",
        "last_rebalance": datetime.date.today().isoformat(),
        "risk_check": "PASS",
        "limits_check": {
            "max_position_size": "10% (PASS)",
            "liquidity_coverage": "100% (PASS)",
            "leverage_ratio": "1.0x (PASS)"
        },
        "audit_id": str(uuid.uuid4()),
        "data_hash": hashlib.sha256(json.dumps(holdings).encode()).hexdigest()[:16]
    }

    portfolio = {
        "fund_name": name.replace('_', ' '),
        "strategy": strategy,
        "aum": round(current_value + cash, 2),
        "cash": round(cash, 2),
        "nav": round((current_value + cash) / 1000000, 2), # Mock NAV
        "holdings": holdings,
        "audit_trail": audit
    }

    with open(os.path.join(PORTFOLIO_DIR, f"{name}.json"), 'w') as f:
        json.dump(portfolio, f, indent=2)

def generate_scenario(name, description, factors):
    scenario = {
        "id": name,
        "title": name.replace('_', ' '),
        "description": description,
        "created_at": datetime.datetime.now().isoformat(),
        "author": "Macro Strategy Team",
        "impact_factors": factors,
        "probability": round(random.uniform(0.10, 0.40), 2)
    }

    with open(os.path.join(SCENARIO_DIR, f"{name}.json"), 'w') as f:
        json.dump(scenario, f, indent=2)

if __name__ == "__main__":
    generate_investment_ecosystem()
