import re

with open("scripts/generate_real_unified_memos.py", "r") as f:
    py = f.read()

# Replace monte_carlo_forecasts logic to include the new metrics-based p10, p50, p90
mc_logic = """
        monte_carlo_forecasts = []
        import random
        base_fcf = fcf / 1e6
        base_rev = projected_rev[0]

        sims_fcf = []
        sims_rev = []
        for _ in range(100):
            scenario_fcf = [base_fcf]
            scenario_rev = [base_rev]
            for _ in range(5):
                scenario_fcf.append(scenario_fcf[-1] * (1 + growth + random.uniform(-0.05, 0.05)))
                scenario_rev.append(scenario_rev[-1] * (1 + growth + random.uniform(-0.02, 0.03)))
            monte_carlo_forecasts.append(scenario_fcf[1:])
            sims_fcf.append(scenario_fcf[1])
            sims_rev.append(scenario_rev[1])

        sims_fcf.sort()
        sims_rev.sort()

        monte_carlo_metrics = {
            "iterations": 100,
            "metrics": {
                "revenue_2025": {
                    "p10": sims_rev[10],
                    "p50": sims_rev[50],
                    "p90": sims_rev[90]
                },
                "fcf_2025": {
                    "p10": sims_fcf[10],
                    "p50": sims_fcf[50],
                    "p90": sims_fcf[90]
                }
            }
        }
"""

py = re.sub(
    r"        monte_carlo_forecasts = \[\]\n        import random\n        base_fcf = fcf / 1e6\n        for _ in range\(100\):\n            scenario_fcf = \[base_fcf\]\n            for _ in range\(5\):\n                scenario_fcf.append\(scenario_fcf\[-1\] \* \(1 \+ growth \+ random.uniform\(-0.05, 0.05\)\)\)\n            monte_carlo_forecasts.append\(scenario_fcf\[1:\]\)",
    mc_logic.strip('\n'),
    py
)

new_schema_addition = """
        # Build the new enriched schema nodes
        financials_node = {
            "historicals": {
                "revenue_2023": hist_records[1]["revenue"] if len(hist_records) > 1 else 0,
                "revenue_2024": hist_records[0]["revenue"] if len(hist_records) > 0 else 0,
                "ebitda_margin": (ebitda / rev_last) if rev_last else 0,
                "net_debt_to_ebitda": (debt/ebitda) if ebitda > 0 else 0,
                "fcf_conversion": (fcf/ebitda) if ebitda > 0 else 0
            },
            "consensus_estimates": {
                "revenue_2025": projected_rev[0] if projected_rev else 0,
                "revenue_2026": projected_rev[1] if len(projected_rev) > 1 else 0,
                "eps_2025": target_mean / 15.0 if target_mean else 0, # mock eps
                "eps_2026": target_mean / 13.0 if target_mean else 0
            },
            "monte_carlo_forecasts": monte_carlo_metrics
        }

        # Format sensitivity matrix correctly
        formatted_sensitivity = []
        for i, w in enumerate(dcf_sensitivity["wacc_range"]):
            for j, g in enumerate(dcf_sensitivity["growth_range"]):
                formatted_sensitivity.append({
                    "wacc": w,
                    "tgr": g,
                    "implied_price": dcf_sensitivity["implied_prices"][i][j]
                })

        valuation_node = {
            "baseCaseEV": ev / 1e6,
            "dcfSensitivityMatrix": formatted_sensitivity
        }

        reg_pd = 0.0015
        reg_lgd = 0.35
        regulatory_node = {
            "facilityRatings": [
                {
                    "facility": "Term Loan B",
                    "internalRating": "A" if (debt/ebitda if ebitda > 0 else 0) < 2 else "BBB",
                    "pd": reg_pd,
                    "lgd": reg_lgd,
                    "el": reg_pd * reg_lgd,
                    "rr": "RR1"
                }
            ],
            "basel_iii_rwa_impact": "Standard corporate exposure RWA."
        }
"""

# Insert the new nodes before "memo = {"
py = py.replace("""        # Build the final object
        memo = {""", new_schema_addition + "\n        # Build the final object\n        memo = {")

# Append new nodes to the memo dict
dict_addition = """
            "companyName": name,
            "financials": financials_node,
            "valuation": valuation_node,
            "regulatoryAnalysis": regulatory_node,
            "peers": [f"{sector[:3].upper()}-P1", f"{sector[:3].upper()}-P2", f"{sector[:3].upper()}-P3"],
"""

py = py.replace("""            "borrower_details": {""", dict_addition + """            "borrower_details": {""")

with open("scripts/generate_real_unified_memos.py", "w") as f:
    f.write(py)
