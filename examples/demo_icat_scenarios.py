import sys
import os
import json
from pprint import pprint

# Add repo root to path
sys.path.append(os.getcwd())

from core.engine.icat import ICATEngine

def main():
    print("Initializing ICAT Engine...")
    engine = ICATEngine()

    filepath = "showcase/data/comprehensive_scenarios.json"
    print(f"Loading scenarios from {filepath}...")

    # Load raw to get keys
    with open(filepath, 'r') as f:
        scenarios = json.load(f)

    for scenario_key in scenarios.keys():
        print(f"\n{'='*60}")
        print(f"Running Scenario: {scenario_key}")
        print(f"{'='*60}")

        try:
            # We use scenario_key as 'ticker' because the file is structured as a dict of scenarios
            result = engine.analyze(
                ticker=scenario_key,
                source="file",
                filepath=filepath,
                scenario_name=scenario_key
            )

            print(f"Ticker: {result.ticker}")
            print(f"Generated At: {result.generated_at}")

            print("\n--- Credit Metrics ---")
            cm = result.credit_metrics
            print(f"Rating: {cm.credit_rating}")
            print(f"PD (1yr): {cm.pd_1yr:.2%}")
            print(f"LGD: {cm.lgd:.2%}")
            print(f"LTV: {cm.ltv:.2f}x")
            print(f"Net Leverage: {cm.net_leverage:.2f}x")
            print(f"DSCR: {cm.dscr:.2f}x")
            print(f"Z-Score: {cm.z_score:.2f}")

            print("\n--- Valuation Metrics ---")
            vm = result.valuation_metrics
            print(f"Enterprise Value: ${vm.enterprise_value:,.2f}")
            print(f"Equity Value: ${vm.equity_value:,.2f}")
            print(f"DCF Value: ${vm.dcf_value:,.2f} ({vm.terminal_value_method})")

            if result.lbo_analysis:
                print("\n--- LBO Analysis ---")
                lbo = result.lbo_analysis
                print(f"IRR: {lbo.irr:.2%}")
                print(f"MoM Multiple: {lbo.mom_multiple:.2f}x")
                print(f"Entry Equity: ${lbo.equity_value_entry:,.2f}")
                print(f"Exit Equity: ${lbo.equity_value_exit:,.2f}")
                print(f"Debt Paydown: ${lbo.debt_paydown:,.2f}")

            if result.carve_out_impact:
                print("\n--- Carve-Out Analysis ---")
                print(f"Valuation Impact: ${result.carve_out_impact:,.2f}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
