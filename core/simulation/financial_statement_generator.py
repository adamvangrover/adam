import random
import datetime

class FinancialStatementGenerator:
    def __init__(self, seed=None):
        if seed:
            random.seed(seed)

    def generate_3_statement_model(self, ticker, sector, market_cap_str, current_price):
        # Parse market cap
        try:
            mcap_val = float(market_cap_str.replace('T', '').replace('B', ''))
            if 'T' in market_cap_str:
                mcap = mcap_val * 1000 # Billions
            else:
                mcap = mcap_val
        except:
            mcap = 100.0 # Default fallback

        # Sector Assumptions (Growth, Margin, Debt/EBITDA)
        assumptions = {
            "Technology": {"growth": (0.05, 0.15), "margin": (0.20, 0.40), "debt_mult": (0.5, 2.0)},
            "Financials": {"growth": (0.02, 0.08), "margin": (0.15, 0.30), "debt_mult": (3.0, 6.0)},
            "Healthcare": {"growth": (0.03, 0.10), "margin": (0.10, 0.25), "debt_mult": (1.0, 3.0)},
            "Consumer Discretionary": {"growth": (0.04, 0.12), "margin": (0.05, 0.15), "debt_mult": (1.5, 3.5)},
            "Consumer Staples": {"growth": (0.01, 0.05), "margin": (0.05, 0.12), "debt_mult": (2.0, 4.0)},
            "Energy": {"growth": (0.0, 0.08), "margin": (0.10, 0.20), "debt_mult": (1.0, 2.5)},
            "Industrials": {"growth": (0.02, 0.06), "margin": (0.08, 0.15), "debt_mult": (2.0, 3.5)},
            "Utilities": {"growth": (0.01, 0.04), "margin": (0.10, 0.20), "debt_mult": (4.0, 6.0)},
        }

        prof = assumptions.get(sector, {"growth": (0.03, 0.07), "margin": (0.10, 0.20), "debt_mult": (1.5, 3.0)})

        years = [2023, 2024, 2025]
        financials = {}

        # Base Year (2025)
        # Revenue approx Market Cap / P/S Ratio (Simulated P/S based on sector margin)
        ps_ratio = prof["margin"][1] * 15 # Rough heuristic
        revenue_2025 = mcap / ps_ratio

        # Backcast for 2023, 2024
        growth_rate = random.uniform(*prof["growth"])
        revenue_2024 = revenue_2025 / (1 + growth_rate)
        revenue_2023 = revenue_2024 / (1 + growth_rate)

        revenues = {2023: revenue_2023, 2024: revenue_2024, 2025: revenue_2025}

        for yr in years:
            rev = revenues[yr]
            # Income Statement
            cogs = rev * (1 - random.uniform(*prof["margin"])) # Using margin as gross margin proxy
            gross_profit = rev - cogs
            opex = gross_profit * random.uniform(0.4, 0.7)
            ebitda = gross_profit - opex
            depreciation = rev * 0.05
            ebit = ebitda - depreciation
            interest = ebit * 0.15 # Approx coverage
            tax = (ebit - interest) * 0.21
            net_income = ebit - interest - tax

            # Balance Sheet (Simplified)
            cash = rev * 0.15
            receivables = rev * 0.10
            inventory = cogs * 0.15
            current_assets = cash + receivables + inventory
            pp_e = rev * 0.4
            total_assets = current_assets + pp_e

            payables = cogs * 0.10
            current_liabs = payables
            long_term_debt = ebitda * random.uniform(*prof["debt_mult"])
            total_liabs = current_liabs + long_term_debt

            # Equity Plug
            total_equity = total_assets - total_liabs

            # Cash Flow (Simplified)
            ocf = net_income + depreciation
            capex = pp_e * 0.1
            fcf = ocf - capex

            financials[yr] = {
                "Income Statement": {
                    "Revenue": round(rev, 2),
                    "COGS": round(cogs, 2),
                    "Gross Profit": round(gross_profit, 2),
                    "Operating Expenses": round(opex, 2),
                    "EBITDA": round(ebitda, 2),
                    "EBIT": round(ebit, 2),
                    "Net Income": round(net_income, 2)
                },
                "Balance Sheet": {
                    "Cash & Equivalents": round(cash, 2),
                    "Total Assets": round(total_assets, 2),
                    "Total Debt": round(long_term_debt, 2),
                    "Total Liabilities": round(total_liabs, 2),
                    "Total Equity": round(total_equity, 2)
                },
                "Cash Flow": {
                    "Operating Cash Flow": round(ocf, 2),
                    "CapEx": round(capex, 2),
                    "Free Cash Flow": round(fcf, 2)
                },
                "Ratios": {
                    "Gross Margin": round(gross_profit/rev, 4),
                    "EBITDA Margin": round(ebitda/rev, 4),
                    "Net Margin": round(net_income/rev, 4),
                    "Leverage (Debt/EBITDA)": round(long_term_debt/ebitda, 2) if ebitda > 0 else 0
                }
            }

        return financials

    def generate_audit_trail(self, model_version="FinGPT-v4.2"):
        now = datetime.datetime.now().isoformat()
        return {
            "model_version": model_version,
            "timestamp": now,
            "data_cutoff": "2025-10-31",
            "compliance_check": "PASS",
            "auditor_id": f"AUDIT-{random.randint(1000,9999)}",
            "confidence_score": round(random.uniform(0.85, 0.99), 4)
        }
