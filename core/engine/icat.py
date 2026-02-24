import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
try:
    import numpy_financial as npf
except ImportError:
    npf = None

from core.financial_data.icat_schema import (
    ICATOutput, CreditMetrics, ValuationMetrics, LBOResult, LBOParameters, DebtTranche, CarveOutParameters
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ICATEngine")

class ICATEngine:
    """
    ICAT Engine (Ingest, Clean, Analyze, Transform) for Institutional Credit Assessment.
    Supports LBO modeling, Credit Risk metrics (PD, LGD, LTV, DSCR), and Valuation (DCF, EV).
    """

    def __init__(self, mock_data_path: str = "showcase/data/icat_mock_data.json"):
        self.mock_data_path = mock_data_path
        self.mock_db = self._load_mock_data()
        self._setup_edgar()

    def _load_mock_data(self) -> Dict[str, Any]:
        if os.path.exists(self.mock_data_path):
            with open(self.mock_data_path, 'r') as f:
                return json.load(f)
        logger.warning(f"Mock data not found at {self.mock_data_path}")
        return {}

    def _setup_edgar(self):
        try:
            from edgartools import Edgartools
            # Placeholder identity if not set in env
            identity = os.environ.get("EDGAR_IDENTITY", "Adam Agent <adam@example.com>")
            # edgartools uses set_identity at module level usually or via simple config
            from edgartools import set_identity
            set_identity(identity)
            self.edgar_available = True
        except ImportError:
            logger.warning("edgartools not installed. SEC data ingestion disabled.")
            self.edgar_available = False
        except Exception as e:
            logger.warning(f"Failed to initialize edgartools: {e}")
            self.edgar_available = False

    def ingest(self, ticker: str, source: str = "mock") -> Dict[str, Any]:
        """
        Ingest financial data from a source (mock or edgar).
        """
        logger.info(f"Ingesting data for {ticker} from {source}")

        if source == "mock":
            data = self.mock_db.get(ticker)
            if not data:
                # Try to find by partial match if exact key fails
                for k, v in self.mock_db.items():
                    if v.get('ticker') == ticker:
                        return v
                raise ValueError(f"Ticker {ticker} not found in mock data.")
            return data

        elif source == "edgar":
            if not self.edgar_available:
                raise ImportError("edgartools is not available.")
            return self._fetch_from_edgar(ticker)

        else:
            raise ValueError(f"Unknown source: {source}")

    def _fetch_from_edgar(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch data from SEC EDGAR using edgartools.
        This is a simplified implementation to map to our internal structure.
        """
        from edgartools import Company
        company = Company(ticker)

        # This would need substantial mapping logic in a real prod env
        # For now, we return a structure that mimics our mock data but with empty/placeholder values
        # or simplified extractions if easy.

        # Real implementation would fetch 10-K, parse Income Statement/Balance Sheet.
        # This is a placeholder for the "Standard public name digestion" capability.
        return {
            "ticker": ticker,
            "name": company.name,
            "sector": "Public", # simplified
            "historical": {
                # In a full implementation, we would populate this from company.financials
                "revenue": [],
                "ebitda": [],
                "year": []
            },
            "source": "SEC EDGAR"
        }

    def clean(self, raw_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Clean and normalize data into a DataFrame.
        """
        hist = raw_data.get('historical', {})
        df = pd.DataFrame(hist)
        if 'year' in df.columns:
            df.set_index('year', inplace=True)
        return df.sort_index()

    def analyze(self,
                ticker: str,
                source: str = "mock",
                scenario_name: str = "Base Case",
                lbo_params: Optional[LBOParameters] = None,
                carve_out_params: Optional[CarveOutParameters] = None
                ) -> ICATOutput:

        raw_data = self.ingest(ticker, source)
        df_hist = self.clean(raw_data)

        # Use default LBO params from mock data if not provided
        if lbo_params is None and 'lbo_params' in raw_data:
            lbo_params = LBOParameters(**raw_data['lbo_params'])

        # 1. Credit Metrics (Current/Historical)
        credit_metrics = self._calculate_credit_metrics(df_hist)

        # 2. Valuation (Simple EV and DCF)
        valuation_metrics = self._calculate_valuation(df_hist, raw_data.get('forecast_assumptions', {}))

        # 3. LBO Analysis (if params exist)
        lbo_result = None
        if lbo_params:
            lbo_result = self._run_lbo(df_hist, lbo_params, raw_data.get('forecast_assumptions', {}), carve_out_params)

        # 4. Carve-out Impact
        carve_out_impact = 0.0
        if carve_out_params:
            # Simple impact: standalone cost adjustments reduce EBITDA -> reduce EV
            # Assuming EV is based on EBITDA multiple
            last_ebitda = df_hist['ebitda'].iloc[-1] if not df_hist.empty else 0
            multiple = lbo_params.entry_multiple if lbo_params else 8.0
            carve_out_impact = -carve_out_params.standalone_cost_adjustments * multiple

        return ICATOutput(
            ticker=ticker,
            scenario_name=scenario_name,
            credit_metrics=credit_metrics,
            valuation_metrics=valuation_metrics,
            lbo_analysis=lbo_result,
            carve_out_impact=carve_out_impact,
            generated_at=datetime.utcnow().isoformat()
        )

    def _calculate_credit_metrics(self, df: pd.DataFrame) -> CreditMetrics:
        if df.empty:
            return CreditMetrics(pd_1yr=0, lgd=0, ltv=0, dscr=0, interest_coverage=0, net_leverage=0)

        latest = df.iloc[-1]

        ebitda = latest.get('ebitda', 1.0)
        debt = latest.get('total_debt', 0.0)
        cash = latest.get('cash', 0.0)
        ebit = latest.get('ebit', ebitda * 0.8) # Proxy
        interest = latest.get('interest_expense', 1.0)
        assets = latest.get('total_assets', 1.0)
        liabilities = latest.get('total_liabilities', debt * 1.2) # Proxy

        net_debt = debt - cash
        net_leverage = net_debt / ebitda if ebitda else 0.0
        interest_coverage = ebit / interest if interest else 0.0

        # LTV (Loan to Value) - Value proxy via EV (using 8x EBITDA default)
        ev_proxy = ebitda * 8.0
        ltv = debt / ev_proxy if ev_proxy else 0.0

        # DSCR (Debt Service Coverage Ratio)
        # EBITDA - Capex / (Interest + Principal)
        # Assuming 5% principal amortization as proxy if not known
        principal = debt * 0.05
        capex = latest.get('capex', 0.0)
        dscr = (ebitda - capex) / (interest + principal) if (interest + principal) > 0 else 0.0

        # Z-Score (Simplified for private/public mix)
        # 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        # A = Working Capital / Total Assets
        # B = Retained Earnings / Total Assets (Proxy: Net Income / Assets)
        # C = EBIT / Total Assets
        # D = Market Value of Equity / Total Liabilities (Proxy: Equity Book Value / Liabilities)
        # E = Sales / Total Assets
        wc = (latest.get('current_assets', assets*0.2) - latest.get('current_liabilities', liabilities*0.5))
        re_proxy = latest.get('net_income', 0) # Weak proxy
        equity_proxy = assets - liabilities

        A = wc / assets
        B = re_proxy / assets
        C = ebit / assets
        D = equity_proxy / liabilities if liabilities else 0
        E = latest.get('revenue', 0) / assets

        z_score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

        # PD mapping (Simplified)
        # Z > 3.0 -> Safe (<0.1%)
        # 1.8 < Z < 3.0 -> Grey (1-5%)
        # Z < 1.8 -> Distress (>10%)
        if z_score > 3.0: pd_1yr = 0.001
        elif z_score > 1.8: pd_1yr = 0.02
        else: pd_1yr = 0.15

        # LGD Mapping based on LTV
        # If LTV < 0.4 -> LGD low (20%)
        # If LTV > 0.8 -> LGD high (60%)
        lgd = 0.2 + (0.6 * min(max(ltv - 0.2, 0), 1.0)) # Linear map roughly

        return CreditMetrics(
            pd_1yr=round(pd_1yr, 4),
            lgd=round(lgd, 4),
            ltv=round(ltv, 4),
            dscr=round(dscr, 4),
            interest_coverage=round(interest_coverage, 4),
            net_leverage=round(net_leverage, 4),
            z_score=round(z_score, 4)
        )

    def _calculate_valuation(self, df: pd.DataFrame, assumptions: Dict[str, Any]) -> ValuationMetrics:
        if df.empty:
            return ValuationMetrics(enterprise_value=0, equity_value=0)

        latest = df.iloc[-1]
        ebitda = latest.get('ebitda', 0)
        debt = latest.get('total_debt', 0)
        cash = latest.get('cash', 0)

        # Simple Multiple Valuation
        multiple = 8.0 # Default
        ev_multiple = ebitda * multiple
        equity_val_multiple = ev_multiple - debt + cash

        # DCF Valuation
        growth = assumptions.get('revenue_growth', 0.05)
        margin = assumptions.get('ebitda_margin', 0.25)
        wacc = assumptions.get('discount_rate', 0.10)
        tgr = assumptions.get('terminal_growth_rate', 0.02)
        years = 5

        current_revenue = latest.get('revenue', ebitda/margin if margin else 0)
        dcf_sum = 0

        for i in range(1, years + 1):
            current_revenue *= (1 + growth)
            proj_ebitda = current_revenue * margin
            # Simplified FCF: EBITDA * conversion (e.g. 60%) to account for Tax, Capex, WC
            fcf = proj_ebitda * 0.6
            dcf_sum += fcf / ((1 + wacc) ** i)

        # Terminal Value
        terminal_fcf = (current_revenue * (1 + growth)) * margin * 0.6
        tv = terminal_fcf / (wacc - tgr)
        tv_discounted = tv / ((1 + wacc) ** years)

        dcf_ev = dcf_sum + tv_discounted

        return ValuationMetrics(
            enterprise_value=ev_multiple,
            equity_value=equity_val_multiple,
            trading_comps_value=ev_multiple, # Proxy
            dcf_value=round(dcf_ev, 2)
        )

    def _run_lbo(self,
                 df: pd.DataFrame,
                 params: LBOParameters,
                 forecast_assumptions: Dict[str, Any],
                 carve_out_params: Optional[CarveOutParameters] = None) -> LBOResult:

        latest = df.iloc[-1]
        ebitda_entry = latest.get('ebitda', 0)

        if carve_out_params:
            ebitda_entry -= carve_out_params.standalone_cost_adjustments

        # Sources and Uses
        entry_ev = ebitda_entry * params.entry_multiple
        total_debt_raised = sum(t.amount for t in params.debt_structure)
        equity_check = entry_ev - total_debt_raised + params.transaction_fees

        # Projection
        years = params.forecast_years
        growth = forecast_assumptions.get('revenue_growth', 0.05)
        margin = forecast_assumptions.get('ebitda_margin', 0.25)
        capex_pct = params.capex_percent_revenue
        tax_rate = params.tax_rate

        current_revenue = latest.get('revenue', ebitda_entry/margin if margin else 0)
        current_debt_tranches = [t.model_copy() for t in params.debt_structure]

        cash_flows = []

        for i in range(years):
            # Operations
            current_revenue *= (1 + growth)
            current_ebitda = current_revenue * margin
            current_capex = current_revenue * capex_pct

            # Debt Service
            interest = sum(t.amount * t.interest_rate for t in current_debt_tranches)
            amortization = sum(t.amount * t.amortization_rate for t in current_debt_tranches)

            # Tax
            depreciation = current_capex # Proxy
            ebit = current_ebitda - depreciation
            taxes = max(0, (ebit - interest) * tax_rate)

            # Cash Flow Available for Debt Service (CFADS)
            # Free Cash Flow = EBITDA - Taxes - Capex - Change in WC
            # Simplified:
            fcf = current_ebitda - taxes - current_capex

            # Paydown Logic
            mandatory_paydown = amortization
            cash_available = fcf - mandatory_paydown

            # Sweep (assume 100% sweep of excess cash to pay down senior debt)
            sweep_paydown = max(0, cash_available)

            # Apply paydowns
            for t in current_debt_tranches:
                # Apply amortization
                pay_amt = min(t.amount, t.amount * t.amortization_rate)
                t.amount -= pay_amt

                # Apply sweep to first tranche (Senior) usually
                if sweep_paydown > 0 and t.name == current_debt_tranches[0].name:
                     sweep_amt = min(t.amount, sweep_paydown)
                     t.amount -= sweep_amt
                     sweep_paydown -= sweep_amt

        # Exit
        exit_ebitda = current_ebitda
        exit_ev = exit_ebitda * params.exit_multiple
        exit_debt = sum(t.amount for t in current_debt_tranches)
        exit_equity = exit_ev - exit_debt

        # Returns
        mom = exit_equity / equity_check if equity_check > 0 else 0

        # IRR
        if npf:
            flows = [-equity_check] + [0]*(years-1) + [exit_equity]
            irr = npf.irr(flows)
        else:
            # Simple approximation if numpy-financial missing
            irr = (mom ** (1/years)) - 1 if mom > 0 else -1

        debt_paydown = total_debt_raised - exit_debt

        return LBOResult(
            irr=round(irr, 4),
            mom_multiple=round(mom, 2),
            equity_value_entry=equity_check,
            equity_value_exit=exit_equity,
            debt_paydown=debt_paydown
        )
