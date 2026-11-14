#!/usr/bin/env python3
"""
CLI Runner for EVE v2.0 (EarlyValueEngine Institutional Underwriting Engine).
Usage:
    python scripts/run_eve_engine.py --ticker MSFT
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adam_finance.eve_engine import EarlyValueEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="EVE v2.0 Single-Name Institutional Underwriting Engine")
    parser.add_argument("--ticker", type=str, default="MSFT", help="Ticker symbol to underwrite (default: MSFT)")
    parser.add_argument("--price", type=float, default=529.76, help="Current market price (default: 529.76 for MSFT)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON memo instead of human-readable digest")

    args = parser.parse_args()

    engine = EarlyValueEngine(ticker=args.ticker)
    data = {
        "revenue_bn": 331.839,
        "revenue_growth_pct": 17.8,
        "gaap_operating_income_bn": 125.0,
        "da_bn": 24.0,
        "sbc_bn": 12.0,
        "ocf_bn": 182.935,
        "capex_bn": 115.948,
        "shares_outstanding_bn": 7.43,
        "invested_capital_bn": 350.0,
        "tax_rate": 0.21,
        "market_price": args.price,
    }

    memo = engine.execute_underwriting(data)

    if args.json:
        print(json.dumps(memo, indent=2))
        return

    print("================================================================================")
    print(f"EVE v2.0 — INSTITUTIONAL SINGLE-NAME DOSSIER | TICKER: {args.ticker}")
    print(f"VALUATION DATE: {memo['valuation_date']} | STATUS: {memo['status']}")
    print("================================================================================")
    print(f"MARKET PRICE:           ${memo['corridor']['market_price']}")
    print(f"WEIGHTED EVE VALUE:     ${memo['corridor']['weighted_eve_value']}")
    print(f"MARGIN OF SAFETY:       {memo['corridor']['margin_of_safety_pct']}%")
    print(f"IC DECISION:            {memo['corridor']['ic_action']}")
    print("--------------------------------------------------------------------------------")
    print("FIVE VALUATION CLOCKS:")
    for clock, val in memo["corridor"]["clocks"].items():
        print(f"  - {clock.upper():20s}: {val}")
    print("--------------------------------------------------------------------------------")
    print("OPERATING ECONOMICS:")
    econ = memo["scrubbed_economics"]
    print(f"  - Revenue: ${econ['revenue_bn']}B (+{econ['revenue_growth_pct']}%)")
    print(f"  - EVE Cash EBITDA: ${econ['eve_cash_ebitda_bn']}B (GAAP EBITDA: ${econ['gaap_ebitda_bn']}B)")
    print(f"  - OCF / CapEx: ${econ['ocf_bn']}B / ${econ['capex_bn']}B (FCF: ${econ['fcf_bn']}B)")
    print(f"  - ROIC: {econ['roic_pct']}%")
    print("--------------------------------------------------------------------------------")
    print("RED-TEAM THESIS BREAKERS & ADVERSARIAL CASES:")
    for k, v in memo["red_team_adversarial"]["vectors"].items():
        print(f"  - {k}: ${v['implied_value']} ({v['description']})")
    print("--------------------------------------------------------------------------------")
    print(f"DIGEST SHA-256: {memo['memo_sha256_digest']}")
    print("================================================================================")


if __name__ == "__main__":
    main()
