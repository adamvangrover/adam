#!/usr/bin/env python3
"""
CLI Runner for EVE v2.0 (Early Value Engine)
Executes institutional single-name underwriting protocol for Microsoft (MSFT) or custom ticker data.
"""

import argparse
import sys
from datetime import datetime, timezone
import json

from adam_finance.eve_engine import EarlyValueEngine


def get_msft_default_data() -> dict:
    """
    Returns baseline MSFT 10-K financial metrics for EVE v2.0 underwriting.
    """
    return {
        "ticker": "MSFT",
        "company_name": "Microsoft Corporation",
        "valuation_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "current_price": 420.00,
        "shares_outstanding": 7430.0,
        "revenue": 245122.0,
        "gaap_operating_income": 109433.0,
        "d_and_a": 22400.0,
        "sbc": 10800.0,
        "cash_capex": 44500.0,
        "free_cash_flow": 74071.0,
        "tax_rate": 0.18,
        "invested_capital": 285000.0,
        "total_debt": 59000.0,
        "cash_and_equivalents": 75500.0,
        "lease_claims": 16000.0,
        "hybrid_claims": 0.0,
        "minority_interests": 0.0,
        "risk_free_rate": 0.042,
        "equity_risk_premium": 0.050,
        "beta": 0.88,
        "pre_tax_cost_of_debt": 0.046,
        "growth_rate": 0.13,
        "terminal_growth": 0.03,
        "terminal_roic": 0.28,
    }


def main():
    parser = argparse.ArgumentParser(description="EVE v2.0 Institutional Underwriting CLI")
    parser.add_argument("--ticker", type=str, default="MSFT", help="Company Ticker (default: MSFT)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON results")
    args = parser.parse_args()

    company_data = get_msft_default_data()
    company_data["ticker"] = args.ticker

    engine = EarlyValueEngine(company_data)
    result = engine.run_underwriting_protocol()

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        memo = result["phase5_memo_and_gate"]["institutional_memo"]
        sha256 = result["phase5_memo_and_gate"]["sha256_digest"]
        print(memo)
        print(f"\nSHA-256 AUDIT DIGEST: {sha256}")
        print(f"VALUATION STATUS:     {result['phase5_memo_and_gate']['integrity_gate_status']}")


if __name__ == "__main__":
    main()
