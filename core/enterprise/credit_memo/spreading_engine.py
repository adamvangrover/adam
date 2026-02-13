from typing import Dict, List, Any
from .model import FinancialSpread, DCFAnalysis, CreditRating, DebtFacility, EquityMarketData
import random
from datetime import datetime, timedelta

class SpreadingEngine:
    """
    Automates financial spreading using simulated OCR and FIBO mapping.
    Protocol: Spreading Agent
    """
    def spread_financials(self, borrower_name: str, raw_data: str) -> FinancialSpread:
        """
        Parses raw text (simulating OCR output) and normalizes it.
        """
        name_lower = borrower_name.lower()

        if "apple" in name_lower:
            # Apple Inc. (FY2025 Q1 - Period Ending Dec 2024)
            assets = 379300.0
            liabilities = 291100.0
            equity = assets - liabilities
            revenue = 391000.0
            ebitda = 132000.0
            net_income = 102000.0
            interest = 3900.0

            return FinancialSpread(
                total_assets=assets,
                total_liabilities=liabilities,
                total_equity=equity,
                revenue=revenue,
                ebitda=ebitda,
                net_income=net_income,
                interest_expense=interest,
                dscr=ebitda / interest if interest > 0 else 999.0,
                leverage_ratio=liabilities / ebitda if ebitda > 0 else 0.0,
                current_ratio=0.98,
                period="FY2025 Q1"
            )

        elif "tesla" in name_lower:
            # Tesla Inc. (FY2024 Q3 TTM)
            assets = 119800.0
            liabilities = 45700.0
            equity = 74100.0
            revenue = 97000.0
            ebitda = 12500.0
            net_income = 8400.0
            interest = 150.0

            return FinancialSpread(
                total_assets=assets,
                total_liabilities=liabilities,
                total_equity=equity,
                revenue=revenue,
                ebitda=ebitda,
                net_income=net_income,
                interest_expense=interest,
                dscr=ebitda / interest if interest > 0 else 999.0,
                leverage_ratio=liabilities / ebitda if ebitda > 0 else 0.0,
                current_ratio=1.70,
                period="FY2024 Q3"
            )

        elif "jpmorgan" in name_lower or "chase" in name_lower:
             # JPMorgan Chase (FY2024 Q4)
            assets = 4000000.0
            liabilities = 3655000.0
            equity = 345000.0
            revenue = 170000.0
            ebitda = 90000.0
            net_income = 56000.0
            interest = 1.0

            return FinancialSpread(
                total_assets=assets,
                total_liabilities=liabilities,
                total_equity=equity,
                revenue=revenue,
                ebitda=ebitda,
                net_income=net_income,
                interest_expense=interest,
                dscr=999.0,
                leverage_ratio=liabilities / equity,
                current_ratio=1.1,
                period="FY2024 Q4"
            )

        elif "techcorp" in name_lower:
            # Mock TechCorp
            assets = 5000.0
            liabilities = 3000.0
            equity = 2000.0

            return FinancialSpread(
                total_assets=assets,
                total_liabilities=liabilities,
                total_equity=equity,
                revenue=1200.0,
                ebitda=350.0,
                net_income=150.0,
                interest_expense=50.0,
                dscr=350.0 / 50.0,
                leverage_ratio=3000.0 / 350.0,
                current_ratio=1.5,
                period="FY2025 Mock"
            )
        else:
            return FinancialSpread(
                total_assets=100.0,
                total_liabilities=50.0,
                total_equity=50.0,
                revenue=20.0,
                ebitda=5.0,
                net_income=2.0,
                interest_expense=1.0,
                dscr=5.0,
                leverage_ratio=10.0,
                current_ratio=2.0,
                period="FY2025"
            )

    def get_historicals(self, current_spread: FinancialSpread) -> List[FinancialSpread]:
        """
        Generates 2 prior years of mock historical data based on current spread.
        """
        historicals = [current_spread]

        # FY-1
        fy_minus_1 = current_spread.model_copy()
        fy_minus_1.period = f"FY{int(current_spread.period[2:6])-1}" if "FY" in current_spread.period else "Prior Year"
        # Simulate ~5-10% smaller previous year
        factor = 0.92
        fy_minus_1.total_assets *= factor
        fy_minus_1.total_liabilities *= factor
        fy_minus_1.total_equity *= factor
        fy_minus_1.revenue *= 0.90 # Slower growth
        fy_minus_1.ebitda *= 0.88 # Margin expansion in current year
        fy_minus_1.net_income *= 0.85
        historicals.append(fy_minus_1)

        # FY-2
        fy_minus_2 = fy_minus_1.model_copy()
        fy_minus_2.period = f"FY{int(current_spread.period[2:6])-2}" if "FY" in current_spread.period else "Prior Year - 1"
        factor = 0.95
        fy_minus_2.total_assets *= factor
        fy_minus_2.total_liabilities *= factor
        fy_minus_2.total_equity *= factor
        fy_minus_2.revenue *= 0.92
        fy_minus_2.ebitda *= 0.90
        fy_minus_2.net_income *= 0.88
        historicals.append(fy_minus_2)

        return historicals

    def calculate_dcf(self, current_spread: FinancialSpread) -> DCFAnalysis:
        """
        Performs a deterministic DCF analysis.
        """
        # Assumptions
        growth_rate = 0.03 # 3% terminal growth
        wacc = 0.09 # 9% WACC

        # Estimate Free Cash Flow (FCF) ~ EBITDA * (1-Tax) - CapEx (Proxy)
        # Simple Proxy: FCF = EBITDA * 0.65
        base_fcf = current_spread.ebitda * 0.65

        projected_fcf = []
        for i in range(1, 6):
            # Degrading growth rate for projection: 5% -> 3%
            g = 0.05 - (0.005 * (i-1))
            fcf = base_fcf * ((1 + g) ** i)
            projected_fcf.append(fcf)

        # Terminal Value = (Final FCF * (1+g)) / (WACC - g)
        terminal_val = (projected_fcf[-1] * (1 + growth_rate)) / (wacc - growth_rate)

        # Discount to PV
        pv_fcf = 0
        for i, fcf in enumerate(projected_fcf):
            pv_fcf += fcf / ((1 + wacc) ** (i + 1))

        pv_terminal = terminal_val / ((1 + wacc) ** 5)

        enterprise_value = pv_fcf + pv_terminal

        # Equity Value = EV - Debt + Cash (Simplified: EV - Liabilities)
        # Note: Liabilities is a proxy for Debt here
        equity_value = enterprise_value - current_spread.total_liabilities

        # Implied Share Price (Mock Share Count)
        # Assume share count such that price is realistic (e.g., ~$150-200)
        # Reverse engineer for display purposes or use mock count
        mock_share_count = equity_value / 185.0 if equity_value > 0 else 1.0
        share_price = equity_value / mock_share_count

        return DCFAnalysis(
            free_cash_flow=projected_fcf,
            growth_rate=growth_rate,
            wacc=wacc,
            terminal_value=terminal_val,
            enterprise_value=enterprise_value,
            equity_value=equity_value,
            share_price=share_price
        )

    def get_credit_ratings(self, borrower_name: str) -> List[CreditRating]:
        """
        Returns mock credit ratings.
        """
        name = borrower_name.lower()
        today = datetime.now().strftime("%Y-%m-%d")

        if "apple" in name:
            return [
                CreditRating(agency="Moody's", rating="Aaa", outlook="Stable", date=today),
                CreditRating(agency="S&P", rating="AA+", outlook="Stable", date=today),
                CreditRating(agency="Fitch", rating="AA+", outlook="Stable", date=today)
            ]
        elif "tesla" in name:
            return [
                CreditRating(agency="Moody's", rating="Baa3", outlook="Positive", date=today),
                CreditRating(agency="S&P", rating="BBB", outlook="Stable", date=today),
                CreditRating(agency="Fitch", rating="BBB", outlook="Stable", date=today)
            ]
        elif "jpmorgan" in name:
            return [
                CreditRating(agency="Moody's", rating="A1", outlook="Stable", date=today),
                CreditRating(agency="S&P", rating="A-", outlook="Stable", date=today),
                CreditRating(agency="Fitch", rating="AA-", outlook="Stable", date=today)
            ]
        elif "techcorp" in name:
            return [
                CreditRating(agency="Moody's", rating="B2", outlook="Negative", date=today),
                CreditRating(agency="S&P", rating="B", outlook="Watch", date=today)
            ]
        else:
            return [
                 CreditRating(agency="Moody's", rating="Ba2", outlook="Stable", date=today),
                 CreditRating(agency="S&P", rating="BB", outlook="Stable", date=today)
            ]

    def get_debt_facilities(self, borrower_name: str) -> List[DebtFacility]:
        """
        Returns mock debt facilities.
        """
        name = borrower_name.lower()

        if "apple" in name:
            return [
                DebtFacility(
                    facility_type="Revolving Credit Facility",
                    amount_committed=10000.0,
                    amount_drawn=0.0,
                    interest_rate="SOFR + 0.75%",
                    maturity_date="2028-09-15",
                    snc_rating="Pass",
                    drc=1.0, # Strong cash flow
                    ltv=0.0, # Unsecured
                    conviction_score=0.98
                ),
                DebtFacility(
                    facility_type="Senior Unsecured Notes (2030)",
                    amount_committed=2500.0,
                    amount_drawn=2500.0,
                    interest_rate="3.25%",
                    maturity_date="2030-05-11",
                    snc_rating="Pass",
                    drc=1.0,
                    ltv=0.1,
                    conviction_score=0.95
                ),
                DebtFacility(
                    facility_type="Senior Unsecured Notes (2040)",
                    amount_committed=1500.0,
                    amount_drawn=1500.0,
                    interest_rate="4.10%",
                    maturity_date="2040-02-28",
                    snc_rating="Pass",
                    drc=1.0,
                    ltv=0.1,
                    conviction_score=0.92
                ),
                DebtFacility(
                    facility_type="Term Loan A",
                    amount_committed=5000.0,
                    amount_drawn=5000.0,
                    interest_rate="SOFR + 1.10%",
                    maturity_date="2027-03-30",
                    snc_rating="Pass",
                    drc=1.0,
                    ltv=0.2,
                    conviction_score=0.90
                )
            ]
        elif "tesla" in name:
            return [
                DebtFacility(
                    facility_type="ABL Revolver",
                    amount_committed=5000.0,
                    amount_drawn=500.0,
                    interest_rate="SOFR + 1.50%",
                    maturity_date="2026-10-20",
                    snc_rating="Pass",
                    drc=0.9,
                    ltv=0.4, # Secured by inventory/AR
                    conviction_score=0.88
                ),
                DebtFacility(
                    facility_type="Convertible Senior Notes",
                    amount_committed=1800.0,
                    amount_drawn=1800.0,
                    interest_rate="2.00%",
                    maturity_date="2027-05-15",
                    snc_rating="Pass",
                    drc=0.85,
                    ltv=0.3,
                    conviction_score=0.85
                ),
                DebtFacility(
                    facility_type="Auto ABS Facilities",
                    amount_committed=3000.0,
                    amount_drawn=2200.0,
                    interest_rate="Variable",
                    maturity_date="Rolling",
                    snc_rating="Pass",
                    drc=0.95,
                    ltv=0.8,
                    conviction_score=0.92
                )
            ]
        elif "techcorp" in name:
             return [
                DebtFacility(
                    facility_type="Revolver",
                    amount_committed=500.0,
                    amount_drawn=350.0,
                    interest_rate="SOFR + 3.50%",
                    maturity_date="2025-12-31",
                    snc_rating="Special Mention",
                    drc=0.4,
                    ltv=0.7,
                    conviction_score=0.45
                ),
                DebtFacility(
                    facility_type="Term Loan B",
                    amount_committed=1200.0,
                    amount_drawn=1200.0,
                    interest_rate="SOFR + 4.75%",
                    maturity_date="2028-06-30",
                    snc_rating="Substandard",
                    drc=0.35,
                    ltv=0.85,
                    conviction_score=0.30
                ),
                DebtFacility(
                    facility_type="Mezzanine Debt",
                    amount_committed=300.0,
                    amount_drawn=300.0,
                    interest_rate="12.00% PIK",
                    maturity_date="2029-06-30",
                    snc_rating="Doubtful",
                    drc=0.1,
                    ltv=0.95,
                    conviction_score=0.15
                )
            ]
        else:
            return [
                 DebtFacility(
                    facility_type="Revolver",
                    amount_committed=100.0,
                    amount_drawn=20.0,
                    interest_rate="Prime + 1.0%",
                    maturity_date="2026-01-01",
                    snc_rating="Pass",
                    drc=0.8,
                    ltv=0.5,
                    conviction_score=0.7
                 ),
                 DebtFacility(
                    facility_type="Term Loan",
                    amount_committed=500.0,
                    amount_drawn=500.0,
                    interest_rate="5.50%",
                    maturity_date="2029-01-01",
                    snc_rating="Pass",
                    drc=0.75,
                    ltv=0.6,
                    conviction_score=0.75
                 )
            ]

    def get_equity_data(self, borrower_name: str) -> EquityMarketData:
        """
        Returns mock equity market data.
        """
        name = borrower_name.lower()

        if "apple" in name:
            return EquityMarketData(
                market_cap=3450000.0,
                share_price=225.50,
                volume_avg_30d=45000000.0,
                pe_ratio=31.5,
                dividend_yield=0.55,
                beta=1.15
            )
        elif "tesla" in name:
            return EquityMarketData(
                market_cap=850000.0,
                share_price=265.40,
                volume_avg_30d=98000000.0,
                pe_ratio=68.2,
                dividend_yield=0.0,
                beta=2.05
            )
        elif "jpmorgan" in name:
             return EquityMarketData(
                market_cap=580000.0,
                share_price=205.10,
                volume_avg_30d=9500000.0,
                pe_ratio=11.8,
                dividend_yield=2.30,
                beta=1.05
            )
        elif "techcorp" in name:
            return EquityMarketData(
                market_cap=1200.0,
                share_price=15.25,
                volume_avg_30d=150000.0,
                pe_ratio=18.5,
                dividend_yield=0.0,
                beta=1.85
            )
        else:
             return EquityMarketData(
                market_cap=100.0,
                share_price=10.00,
                volume_avg_30d=1000.0,
                pe_ratio=15.0,
                dividend_yield=1.0,
                beta=1.0
            )

# Global Instance
spreading_engine = SpreadingEngine()
