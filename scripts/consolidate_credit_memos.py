import json
import os
import random

OUTPUT_FILE = "showcase/data/unified_credit_memos.json"
LIBRARY_FILE = "showcase/data/credit_memo_library.json"
DATA_DIR = "showcase/data"


def generate_historical_financials(base_rev, base_ebitda):
    """Generate 5 years of historical financial data."""
    hist = []
    current_year = 2023
    for i in range(5):
        year = current_year - i
        growth = random.uniform(0.02, 0.15)
        margin = random.uniform(0.15, 0.35)

        rev = base_rev / ((1 + growth) ** i) if i > 0 else base_rev
        ebitda = rev * margin if i > 0 else base_ebitda

        hist.append(
            {
                "year": year,
                "revenue": round(rev, 1),
                "ebitda": round(ebitda, 1),
                "net_income": round(ebitda * 0.6, 1),
                "fcf": round(ebitda * 0.45, 1),
                "total_debt": round(ebitda * random.uniform(1.5, 3.5), 1),
                "cash": round(rev * random.uniform(0.05, 0.15), 1),
            }
        )
    return sorted(hist, key=lambda x: x["year"])


def generate_consensus_estimates(base_rev, base_ebitda):
    """Generate consensus estimates for next 3 years."""
    estimates = []
    current_year = 2024
    for i in range(3):
        year = current_year + i
        growth_mean = random.uniform(0.05, 0.12)

        rev_mean = base_rev * ((1 + growth_mean) ** (i + 1))
        ebitda_mean = rev_mean * random.uniform(0.2, 0.3)

        estimates.append(
            {
                "year": year,
                "revenue": {
                    "low": round(rev_mean * 0.9, 1),
                    "mean": round(rev_mean, 1),
                    "high": round(rev_mean * 1.1, 1),
                },
                "ebitda": {
                    "low": round(ebitda_mean * 0.85, 1),
                    "mean": round(ebitda_mean, 1),
                    "high": round(ebitda_mean * 1.15, 1),
                },
                "eps": {
                    "low": round((ebitda_mean * 0.5) / 1000 * 0.9, 2),
                    "mean": round((ebitda_mean * 0.5) / 1000, 2),
                    "high": round((ebitda_mean * 0.5) / 1000 * 1.1, 2),
                },
            }
        )
    return estimates


def generate_monte_carlo_forecasts(base_fcf, base_debt, base_ebitda):
    """Generate simulated FCF and Leverage forecasts."""
    scenarios = ["Base Case", "Bull Case", "Bear Case"]
    forecasts = []

    for scenario in scenarios:
        if scenario == "Bull Case":
            growth = 0.15
            margin_imp = 0.02
        elif scenario == "Bear Case":
            growth = -0.05
            margin_imp = -0.05
        else:
            growth = 0.05
            margin_imp = 0.0

        fcf_proj = []
        lev_proj = []
        current_fcf = base_fcf
        current_ebitda = base_ebitda
        current_debt = base_debt

        for year in range(2024, 2029):
            current_fcf = current_fcf * (1 + growth)
            current_ebitda = current_ebitda * (1 + growth + margin_imp)
            # Paydown debt with 50% of FCF
            current_debt = max(0, current_debt - (current_fcf * 0.5))

            fcf_proj.append({"year": year, "value": round(current_fcf, 1)})
            lev_proj.append(
                {
                    "year": year,
                    "value": round(
                        current_debt / current_ebitda if current_ebitda > 0 else 0, 2
                    ),
                }
            )

        forecasts.append(
            {
                "scenario": scenario,
                "fcf_trajectory": fcf_proj,
                "leverage_trajectory": lev_proj,
            }
        )
    return forecasts


def generate_dcf_sensitivity(base_share_price):
    """Generate a DCF sensitivity matrix."""
    wacc_range = [0.08, 0.09, 0.10, 0.11, 0.12]
    tgr_range = [0.01, 0.02, 0.03, 0.04, 0.05]
    matrix = []

    for wacc in wacc_range:
        row = {"wacc": wacc, "values": []}
        for tgr in tgr_range:
            # Simple synthetic sensitivity logic
            wacc_impact = (0.10 - wacc) * 10
            tgr_impact = (tgr - 0.02) * 20
            implied_price = base_share_price * (1 + wacc_impact + tgr_impact)
            row["values"].append(
                {"tgr": tgr, "implied_price": max(1.0, round(implied_price, 2))}
            )
        matrix.append(row)
    return {"wacc_range": wacc_range, "tgr_range": tgr_range, "matrix": matrix}


def enrich_memo(memo):
    """Deeply enrich the memo with high-fidelity synthetic data."""

    # Try to extract base metrics from existing data
    base_rev = 10000.0
    base_ebitda = 2500.0
    base_debt = 5000.0
    base_share_price = 100.0

    if "historical_financials" in memo and len(memo["historical_financials"]) > 0:
        latest = memo["historical_financials"][0]
        base_rev = latest.get("revenue", base_rev)
        base_ebitda = latest.get("ebitda", base_ebitda)
        base_debt = latest.get("gross_debt", latest.get("total_debt", base_debt))

    if "equity_data" in memo and "share_price" in memo["equity_data"]:
        base_share_price = memo["equity_data"]["share_price"]
    elif "dcf_analysis" in memo and "share_price" in memo["dcf_analysis"]:
        base_share_price = memo["dcf_analysis"]["share_price"]

    base_fcf = base_ebitda * 0.45

    # Inject enriched data
    memo["enriched_data"] = {
        "historical_series": generate_historical_financials(base_rev, base_ebitda),
        "consensus_estimates": generate_consensus_estimates(base_rev, base_ebitda),
        "monte_carlo_simulations": generate_monte_carlo_forecasts(
            base_fcf, base_debt, base_ebitda
        ),
        "dcf_sensitivity": generate_dcf_sensitivity(base_share_price),
    }

    # Ensure there's a strong System 2 Critique block specifically for the UI to parse
    if "system_two_critique" not in memo:
        memo["system_two_critique"] = {
            "critique_points": [
                "Thesis appears directionally correct but assumptions may be aggressive.",
                "Leverage trajectory heavily dependent on successful integration of recent M&A.",
            ],
            "conviction_score": 0.75,
            "quantitative_analysis": {
                "variance_analysis": "Historical margins volatile, forward projections assume stabilization."
            },
        }

    return memo


def consolidate():
    with open(LIBRARY_FILE, "r") as f:
        library = json.load(f)

    # Sort or pick top 10
    library.sort(key=lambda x: x.get("risk_score", 0), reverse=True)
    top_entries = library[:10]

    consolidated = []
    for entry in top_entries:
        file_path = os.path.join(DATA_DIR, entry["file"])
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                memo_data = json.load(f)

                # Enrich data
                memo_data = enrich_memo(memo_data)

                consolidated.append(memo_data)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(consolidated, f, indent=2)

    print(
        f"Consolidated and enriched {len(consolidated)} credit memos into {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    consolidate()
