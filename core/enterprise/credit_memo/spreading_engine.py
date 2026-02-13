from typing import Dict, List, Any
from pydantic import BaseModel, Field

class FinancialSpread(BaseModel):
    """
    Standardized financial template (FIBO-aligned).
    """
    total_assets: float
    total_liabilities: float
    total_equity: float
    revenue: float
    ebitda: float
    net_income: float
    interest_expense: float
    dscr: float
    leverage_ratio: float
    current_ratio: float
    period: str

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
            # Mock Apple: Huge cash, low leverage
            assets = 380000.0
            liabilities = 300000.0
            equity = 80000.0
            revenue = 400000.0
            ebitda = 135000.0
            interest = 4000.0

            return FinancialSpread(
                total_assets=assets,
                total_liabilities=liabilities,
                total_equity=equity,
                revenue=revenue,
                ebitda=ebitda,
                net_income=105000.0,
                interest_expense=interest,
                dscr=ebitda / interest, # ~33.75x (Very High)
                leverage_ratio=liabilities / ebitda, # ~2.2x
                current_ratio=0.98,
                period="FY2025"
            )

        elif "tesla" in name_lower:
            # Mock Tesla: High CapEx, Variable Margins
            assets = 110000.0
            liabilities = 45000.0
            equity = 65000.0
            revenue = 98000.0
            ebitda = 16000.0
            interest = 800.0

            return FinancialSpread(
                total_assets=assets,
                total_liabilities=liabilities,
                total_equity=equity,
                revenue=revenue,
                ebitda=ebitda,
                net_income=9000.0,
                interest_expense=interest,
                dscr=ebitda / interest, # ~20x
                leverage_ratio=liabilities / ebitda, # ~2.8x
                current_ratio=1.7,
                period="FY2025"
            )

        elif "jpmorgan" in name_lower or "chase" in name_lower:
             # Mock Bank: High Leverage (Deposits are Liabilities), Low Margins
            assets = 4000000.0
            liabilities = 3700000.0
            equity = 300000.0
            revenue = 170000.0 # Net Revenue
            ebitda = 90000.0 # Operating Profit
            interest = 0.0 # Not applicable to banks in same way, set to 1 for div/0

            return FinancialSpread(
                total_assets=assets,
                total_liabilities=liabilities,
                total_equity=equity,
                revenue=revenue,
                ebitda=ebitda,
                net_income=55000.0,
                interest_expense=1.0, # N/A
                dscr=999.0,
                leverage_ratio=liabilities / equity, # ~12.3x (Standard for banks)
                current_ratio=1.1, # Liquidity Coverage Ratio proxy
                period="FY2025"
            )

        elif "techcorp" in name_lower:
            # Mock TechCorp: High Leverage, Mid-Cap
            assets = 5000.0
            liabilities = 3000.0
            equity = 2000.0

            # Integrity Check
            delta = abs(assets - (liabilities + equity))
            if delta > 1.0:
                raise ValueError(f"Balance Sheet Imbalance: {delta}")

            return FinancialSpread(
                total_assets=assets,
                total_liabilities=liabilities,
                total_equity=equity,
                revenue=1200.0,
                ebitda=350.0,
                net_income=150.0,
                interest_expense=50.0,
                dscr=350.0 / 50.0, # 7.0x
                leverage_ratio=3000.0 / 350.0, # ~8.57x
                current_ratio=1.5,
                period="FY2025"
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

# Global Instance
spreading_engine = SpreadingEngine()
