import json
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.afos_pd_engine import calculate_pd

OBLIGOR_DATA = [
    {"name": "Microsoft Corporation", "ticker": "MSFT", "lei": "INR2ETJR104677DF8053", "totalDebt": 108500, "stDebt": 14200, "ebitda": 145200, "interest": 3400, "cash": 84500, "assets": 530000, "eqVol": 0.195, "oas": 36, "currentPrice": 420.50, "marketCap": 3120000, "sharesOut": 7420, "capexExposure": 0.35, "seatExposure": 0.15, "powerExposure": 0.25},
    {"name": "Amazon.com, Inc.", "ticker": "AMZN", "lei": "ZXF8T627K71182390141", "totalDebt": 152000, "stDebt": 22500, "ebitda": 128000, "interest": 4100, "cash": 88000, "assets": 585000, "eqVol": 0.228, "oas": 48, "currentPrice": 205.20, "marketCap": 2150000, "sharesOut": 10477, "capexExposure": 0.40, "seatExposure": 0.05, "powerExposure": 0.35},
    {"name": "Alphabet Inc.", "ticker": "GOOGL", "lei": "5493006MHB84DD0ZWV18", "totalDebt": 31500, "stDebt": 6800, "ebitda": 132000, "interest": 450, "cash": 115000, "assets": 435000, "eqVol": 0.212, "oas": 28, "currentPrice": 182.40, "marketCap": 2280000, "sharesOut": 12500, "capexExposure": 0.30, "seatExposure": 0.05, "powerExposure": 0.20},
    {"name": "Meta Platforms, Inc.", "ticker": "META", "lei": "254900A1A28A81149B90", "totalDebt": 42800, "stDebt": 4200, "ebitda": 84500, "interest": 1250, "cash": 65200, "assets": 262000, "eqVol": 0.264, "oas": 42, "currentPrice": 608.00, "marketCap": 1540000, "sharesOut": 2533, "capexExposure": 0.45, "seatExposure": 0.02, "powerExposure": 0.25},
    {"name": "Salesforce, Inc.", "ticker": "CRM", "lei": "549300H2824L025Z4926", "totalDebt": 14100, "stDebt": 2500, "ebitda": 13800, "interest": 480, "cash": 14800, "assets": 102000, "eqVol": 0.275, "oas": 74, "currentPrice": 302.50, "marketCap": 285000, "sharesOut": 942, "capexExposure": 0.05, "seatExposure": 0.50, "powerExposure": 0.02},
    {"name": "Adobe Inc.", "ticker": "ADBE", "lei": "549300R3L9P9O9L8N884", "totalDebt": 6200, "stDebt": 1800, "ebitda": 8900, "interest": 190, "cash": 8100, "assets": 31500, "eqVol": 0.290, "oas": 62, "currentPrice": 472.00, "marketCap": 210000, "sharesOut": 445, "capexExposure": 0.05, "seatExposure": 0.45, "powerExposure": 0.02},
    {"name": "Oracle Corporation", "ticker": "ORCL", "lei": "549300V5Q8D2P2U5S631", "totalDebt": 88500, "stDebt": 9200, "ebitda": 24200, "interest": 3650, "cash": 11200, "assets": 142000, "eqVol": 0.251, "oas": 108, "currentPrice": 152.80, "marketCap": 420000, "sharesOut": 2748, "capexExposure": 0.30, "seatExposure": 0.25, "powerExposure": 0.20},
    {"name": "Constellation Energy Corp.", "ticker": "CEG", "lei": "5493006N8J2P4K8L9M12", "totalDebt": 9800, "stDebt": 1400, "ebitda": 4600, "interest": 460, "cash": 1850, "assets": 38200, "eqVol": 0.342, "oas": 112, "currentPrice": 228.00, "marketCap": 82000, "sharesOut": 360, "capexExposure": 0.15, "seatExposure": 0.00, "powerExposure": 0.50},
    {"name": "NextEra Energy, Inc.", "ticker": "NEE", "lei": "5493008E8B6K5D4C3B21", "totalDebt": 78500, "stDebt": 14800, "ebitda": 19200, "interest": 3100, "cash": 2800, "assets": 182000, "eqVol": 0.220, "oas": 122, "currentPrice": 81.60, "marketCap": 168000, "sharesOut": 2058, "capexExposure": 0.20, "seatExposure": 0.00, "powerExposure": 0.45},
    {"name": "S&P 500 Broad Non-Financial Remainder", "ticker": "SPX_CORE", "lei": "AGGREGATE-COMPOSITE", "totalDebt": 6608900, "stDebt": 1102800, "ebitda": 2569800, "interest": 301070, "cash": 1739750, "assets": 41984300, "eqVol": 0.178, "oas": 134, "currentPrice": 75.20, "marketCap": 36865000, "sharesOut": 490226, "capexExposure": 0.10, "seatExposure": 0.12, "powerExposure": 0.10}
]

def calculate_target_price(ob):
    state = {
      'seatChurn': -10,
      'capexWriteoff': 15,
      'powerLatency': 12,
      'rfRate': 4.25,
      'spreadShock': 0,
      'refiBump': 120
    }

    seatDragFactor = (state['seatChurn'] / 100) * ob['seatExposure']
    capexDragFactor = (state['capexWriteoff'] / 100) * ob['capexExposure'] * 0.45
    powerDragFactor = (state['powerLatency'] / 36) * ob['powerExposure'] * 0.15

    totalEbitdaDelta = seatDragFactor - capexDragFactor - powerDragFactor

    valuationDecline = (totalEbitdaDelta * 1.2) - ((state['rfRate'] - 4.25) * 0.05) - (state['spreadShock'] / 1000)
    targetCap = max(ob['marketCap'] * (1 + valuationDecline), ob['marketCap'] * 0.4)
    targetPrice = targetCap / ob['sharesOut']

    return targetPrice

os.makedirs("reports", exist_ok=True)
with open("reports/sovereign_obligor_pd_ratings.md", "w") as f:
    f.write("# Sovereign Obligor Intelligence - PD & Credit Report\n\n")
    f.write("A quantitative review of S&P 500 obligors using the AFOS v30.1 Dual-World PD Engine.\n\n")

    for ob in OBLIGOR_DATA:
        res = calculate_pd(
            name=ob["name"],
            ticker_or_lei=f"{ob['ticker']} / {ob['lei']}",
            E=ob["marketCap"],
            sigma_E=ob["eqVol"],
            s=ob["oas"],
            Total_Debt=ob["totalDebt"],
            STD=ob["stDebt"],
            EBITDA=ob["ebitda"],
            Interest=ob["interest"],
            FCF=ob["ebitda"] - ob["interest"],
            Total_Assets=ob["assets"],
            Cash=ob["cash"]
        )

        target_price = calculate_target_price(ob)
        price_delta_pct = ((target_price - ob['currentPrice']) / ob['currentPrice']) * 100

        f.write(f"## {ob['name']} ({ob['ticker']})\n")
        f.write(f"**LEI:** {ob['lei']} | **Current Price:** ${ob['currentPrice']:.2f} | **1-Year Price Target:** ${target_price:.2f} ({price_delta_pct:+.1f}%) | **Market Cap:** ${ob['marketCap']:,}M\n\n")

        f.write("### Model Inputs & Assumptions\n")
        f.write(f"- Total Debt: ${ob['totalDebt']:,}M\n")
        f.write(f"- ST Debt: ${ob['stDebt']:,}M\n")
        f.write(f"- EBITDA: ${ob['ebitda']:,}M\n")
        f.write(f"- Interest Expense: ${ob['interest']:,}M\n")
        f.write(f"- Cash: ${ob['cash']:,}M\n")
        f.write(f"- Total Assets: ${ob['assets']:,}M\n")
        f.write(f"- Equity Volatility: {ob['eqVol']}\n")
        f.write(f"- Unsecured Spread: {ob['oas']} bps\n")
        f.write(f"- Shares Outstanding: {ob['sharesOut']:,}M\n")
        f.write(f"- Exposure Weights (CapEx: {ob['capexExposure']}, Seat: {ob['seatExposure']}, Power: {ob['powerExposure']})\n")
        f.write("- Q1 2026 Baseline Assumptions (Seat Churn: -10%, CapEx Writeoff: +15%, Power Latency: +12m)\n\n")

        f.write("### Dual-World PD Assessment Outputs\n")
        f.write(f"- **Execution State:** {res['dual_world_arbitration']['execution_state']}\n")
        f.write(f"- **Structural Merton PD:** {res['model_pillars']['structural_merton_pd'] * 100:.2f}%\n")
        f.write(f"- **Market Spread PD:** {res['model_pillars']['market_spread_pd'] * 100:.2f}%\n")
        f.write(f"- **Fundamental Logit PD:** {res['model_pillars']['fundamental_logit_pd'] * 100:.2f}%\n")
        f.write(f"- **Champion PIT PD:** {res['dual_world_arbitration']['champion_pit_pd'] * 100:.2f}%\n")
        f.write(f"- **Challenger Counterfactual PD:** {res['dual_world_arbitration']['challenger_counterfactual_pd'] * 100:.2f}%\n")
        f.write(f"- **Final Obligor PIT PD:** {res['final_capital_assessment']['final_obligor_pit_pd'] * 100:.2f}%\n")
        f.write(f"- **Basel III Stressed PD:** {res['final_capital_assessment']['basel_vasicek_999_stressed_pd'] * 100:.2f}%\n\n")
        f.write(f"**Supervisory Directive:** {res['final_capital_assessment']['actionable_supervisory_directive']}\n\n")
        f.write("---\n\n")

print("Report generated at reports/sovereign_obligor_pd_ratings.md")
