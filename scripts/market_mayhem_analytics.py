import random
import hashlib

def get_hash_seed(text):
    """Generates a deterministic integer seed from text."""
    return int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16)

def generate_risk_matrix(item_text, item_type):
    """Generates a list of strategic risks based on content themes."""
    seed = get_hash_seed(item_text)
    random.seed(seed)

    potential_risks = [
        {"risk": "Geopolitical Escalation", "impact": "HIGH", "prob": "MED", "mitigant": "Hedge with OTM Puts on SPX, Long Gold"},
        {"risk": "Inflation Resurgence", "impact": "CRITICAL", "prob": "LOW", "mitigant": "Short Duration Bonds (TBT), Long Energy (XLE)"},
        {"risk": "Liquidity Crunch", "impact": "HIGH", "prob": "LOW", "mitigant": "Raise Cash Levels > 15%, Hold T-Bills"},
        {"risk": "AI Capex Disappointment", "impact": "MED", "prob": "MED", "mitigant": "Rotate into Value/Industrials, Short NVDA/SMH"},
        {"risk": "Credit Spread Widening", "impact": "HIGH", "prob": "LOW", "mitigant": "Buy CDS Protection (CDX.IG), Short HYG"},
        {"risk": "Regulatory Crackdown", "impact": "MED", "prob": "HIGH", "mitigant": "Reduce Big Tech Exposure, Long Utilities"},
        {"risk": "Dollar Devaluation", "impact": "MED", "prob": "MED", "mitigant": "Long Bitcoin, Long Emerging Markets (EEM)"},
        {"risk": "Supply Chain Shock", "impact": "MED", "prob": "LOW", "mitigant": "Long Commodities (DBC), Long Shipping"},
        {"risk": "Consumer Recession", "impact": "HIGH", "prob": "MED", "mitigant": "Long Staples (XLP), Short Discretionary (XLY)"}
    ]

    # Select 3-5 risks deterministically
    num_risks = random.randint(3, 5)
    selected_risks = random.sample(potential_risks, num_risks)

    # Sort by Impact (CRITICAL > HIGH > MED)
    priority = {"CRITICAL": 0, "HIGH": 1, "MED": 2, "LOW": 3}
    selected_risks.sort(key=lambda x: priority[x["impact"]])

    return selected_risks

def generate_arbitrage_opportunities(item_text):
    """Generates pair trades and sector arbitrage ideas."""
    seed = get_hash_seed(item_text)
    random.seed(seed + 1)

    sectors = ["Tech", "Energy", "Financials", "Healthcare", "Industrials", "Materials", "Consumer", "Utilities", "Real Estate"]

    # Generate 1-2 Arbitrage Ideas
    ideas = []

    # Idea 1: Sector Pair
    long_sec = random.choice(sectors)
    short_sec = random.choice([s for s in sectors if s != long_sec])
    spread = round(random.uniform(1.5, 4.5), 1)
    ideas.append({
        "type": "SECTOR_ARB",
        "strategy": f"Long {long_sec} / Short {short_sec}",
        "thesis": f"Valuation gap of {spread}σ implies mean reversion. {long_sec} showing superior free cash flow yield.",
        "horizon": "2-4 Weeks",
        "conviction": f"{random.randint(65, 95)}%"
    })

    # Idea 2: Volatility Arb (Optional)
    if random.random() > 0.5:
        ideas.append({
            "type": "VOL_ARB",
            "strategy": "Short VIX / Long Skew",
            "thesis": "Implied volatility premium is overpriced relative to realized vol. Selling covered calls on index.",
            "horizon": "1 Week",
            "conviction": f"{random.randint(60, 85)}%"
        })

    return ideas

def generate_institutional_flows(item_text):
    """Generates mock 13F / Flow data for sectors."""
    seed = get_hash_seed(item_text)
    random.seed(seed + 2)

    sectors = ["Technology", "Energy", "Financials", "Healthcare", "Discretionary", "Staples", "Utilities", "Real Estate", "Materials", "Comms"]

    flows = {}
    for sector in sectors:
        # Generate a flow value between -100 (Heavy Sell) and +100 (Heavy Buy)
        # Bias towards slight positive for growth sectors if sentiment is high (simulated)
        bias = 10 if sector in ["Technology", "Comms"] else -5
        val = int(random.gauss(bias, 40))
        val = max(-100, min(100, val))
        flows[sector] = val

    return flows

def determine_market_regime(item_text):
    """Classifies the market regime based on text keywords."""
    text_lower = item_text.lower()

    if "inflation" in text_lower and "growth" in text_lower:
        return "REFLATION / BOOM"
    elif "inflation" in text_lower and ("recession" in text_lower or "slow" in text_lower):
        return "STAGFLATION"
    elif "growth" in text_lower and "tech" in text_lower:
        return "SECULAR GROWTH"
    elif "crash" in text_lower or "panic" in text_lower:
        return "CRISIS / LIQUIDITY SHOCK"
    else:
        return "LATE CYCLE VOLATILITY"

def generate_advanced_analytics(item):
    """Main entry point to generate all advanced data."""
    full_text = f"{item['title']} {item['summary']} {item.get('full_body', '')}"

    return {
        "risks": generate_risk_matrix(full_text, item['type']),
        "arbitrage": generate_arbitrage_opportunities(full_text),
        "flows": generate_institutional_flows(full_text),
        "regime": determine_market_regime(full_text)
    }
