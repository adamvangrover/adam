import re

with open("core/agents/orchestrators/credit_memo_orchestrator.py", "r") as f:
    py = f.read()

replacement = """
            "system_two_critique": {
                "critique_points": [
                    "Valuation aligns with sector.",
                    "Legal review confirms standard protections.",
                ],
                "conviction_score": 0.85,
                "verification_status": "PASS",
                "author_agent": "System 2",
                "quantitative_analysis": {
                    "ratios_checked": ["Leverage", "DSCR", "Z-Score"],
                    "variance_analysis": "Consistent",
                    "dcf_validation": "WACC within range",
                },
            },
            "equity_data": data["market_data"],
            "financials": {
                "historicals": {
                    "revenue_2023": hist[1]["revenue"] if len(hist) > 1 else 0,
                    "revenue_2024": hist[0]["revenue"] if len(hist) > 0 else 0,
                    "ebitda_margin": (hist[0].get("ebitda", 0) / hist[0].get("revenue", 1)) if hist else 0,
                    "net_debt_to_ebitda": icat.credit_metrics.net_leverage,
                    "fcf_conversion": 0.85
                },
                "consensus_estimates": {
                    "revenue_2025": data["historical"]["revenue"][-1] * 1.05,
                    "revenue_2026": data["historical"]["revenue"][-1] * 1.10,
                    "eps_2025": 5.50,
                    "eps_2026": 6.10
                },
                "monte_carlo_forecasts": {
                    "iterations": 1000,
                    "metrics": {
                        "revenue_2025": {"p10": data["historical"]["revenue"][-1]*0.9, "p50": data["historical"]["revenue"][-1]*1.05, "p90": data["historical"]["revenue"][-1]*1.15},
                        "fcf_2025": {"p10": 1000, "p50": 1200, "p90": 1500}
                    }
                }
            },
            "valuation": {
                "baseCaseEV": icat.valuation_metrics.enterprise_value,
                "dcfSensitivityMatrix": [
                    {"wacc": data["forecast_assumptions"]["discount_rate"] - 0.01, "tgr": data["forecast_assumptions"]["terminal_growth_rate"], "implied_price": icat.valuation_metrics.dcf_value * 1.1},
                    {"wacc": data["forecast_assumptions"]["discount_rate"], "tgr": data["forecast_assumptions"]["terminal_growth_rate"], "implied_price": icat.valuation_metrics.dcf_value},
                    {"wacc": data["forecast_assumptions"]["discount_rate"] + 0.01, "tgr": data["forecast_assumptions"]["terminal_growth_rate"], "implied_price": icat.valuation_metrics.dcf_value * 0.9}
                ]
            },
            "regulatoryAnalysis": {
                "facilityRatings": [
                    {
                        "facility": "Mock Senior Debt",
                        "internalRating": "BBB",
                        "pd": risk["risk_quant_metrics"]["PD"],
                        "lgd": risk["risk_quant_metrics"]["LGD"],
                        "el": risk["risk_quant_metrics"]["PD"] * risk["risk_quant_metrics"]["LGD"],
                        "rr": "RR2"
                    }
                ],
                "basel_iii_rwa_impact": "Standard Mock Impact"
            },
            "peers": [f"{data['sector'][:3].upper()}-A", f"{data['sector'][:3].upper()}-B"]
        }
"""

py = re.sub(
    r'            "system_two_critique": \{\n                "critique_points": \[\n                    "Valuation aligns with sector.",\n                    "Legal review confirms standard protections.",\n                \],\n                "conviction_score": 0.85,\n                "verification_status": "PASS",\n                "author_agent": "System 2",\n                "quantitative_analysis": \{\n                    "ratios_checked": \["Leverage", "DSCR", "Z-Score"\],\n                    "variance_analysis": "Consistent",\n                    "dcf_validation": "WACC within range",\n                \},\n            \},\n            "equity_data": data\["market_data"\],\n        \}',
    replacement.strip("\n"),
    py
)

with open("core/agents/orchestrators/credit_memo_orchestrator.py", "w") as f:
    f.write(py)
