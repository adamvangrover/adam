import logging
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
try:
    import numpy_financial as npf
except ImportError:
    npf = None

from core.financial_data.icat_schema import (
    ICATOutput, CreditMetrics, ValuationMetrics, LBOResult, LBOParameters,
    DebtTranche, CarveOutParameters, ForecastAssumptions, EnvironmentContext
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

    def ingest_from_repo_file(self, filepath: str) -> Dict[str, Any]:
        """
        Ingest financial data from a local JSON file in the repo.
        Useful for user-selected mock data simulations.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                logger.info(f"Loaded data from {filepath}")
                return data
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in file: {filepath}")
        except Exception as e:
            raise RuntimeError(f"Error reading file {filepath}: {e}")

    def ingest(self, ticker: str, source: str = "mock", filepath: Optional[str] = None) -> Dict[str, Any]:
        """
        Ingest financial data from a source (mock, edgar, or file).
        """
        logger.info(f"Ingesting data for {ticker} from {source}")

        if source == "file" and filepath:
            data = self.ingest_from_repo_file(filepath)
            # If the file is a collection, try to find the ticker
            if ticker in data:
                return data[ticker]
            # If the file itself is the data object
            if data.get('ticker') == ticker:
                return data
            # If not found
            raise ValueError(f"Ticker {ticker} not found in file {filepath}")

        elif source == "mock":
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
                logger.warning("Edgartools not available. Returning mock SEC structure.")
                return self._mock_edgar_fetch(ticker)
            return self._fetch_from_edgar(ticker)

        else:
            raise ValueError(f"Unknown source: {source}")

    def _mock_edgar_fetch(self, ticker: str) -> Dict[str, Any]:
        """Mock response simulating an SEC fetch for demonstration."""
        return {
            "ticker": ticker,
            "name": f"{ticker} Inc. (Simulated SEC Data)",
            "sector": "Simulated Public Sector",
            "historical": {
                "year": [2021, 2022, 2023],
                "revenue": [1000, 1100, 1210],
                "ebitda": [200, 220, 242],
                "total_debt": [500, 480, 460],
                "cash": [50, 60, 70]
            },
            "source": "SEC EDGAR (Simulated)"
        }

    def _fetch_from_edgar(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch data from SEC EDGAR using edgartools.
        """
        try:
            from edgartools import Company
            company = Company(ticker)
            return {
                "ticker": ticker,
                "name": company.name,
                "sector": "Public",
                "historical": {
                    "year": [2022, 2023],
                    "revenue": [0, 0],
                    "ebitda": [0, 0]
                },
                "source": "SEC EDGAR"
            }
        except Exception as e:
            logger.error(f"Error fetching from EDGAR: {e}")
            raise

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
                filepath: Optional[str] = None,
                scenario_name: str = "Base Case",
                lbo_params: Optional[LBOParameters] = None,
                carve_out_params: Optional[CarveOutParameters] = None,
                forecast_assumptions: Optional[ForecastAssumptions] = None,
                environment: Optional[EnvironmentContext] = None
                ) -> ICATOutput:

        raw_data = self.ingest(ticker, source, filepath)
        df_hist = self.clean(raw_data)

        # Environment Context
        if environment is None:
            environment = EnvironmentContext()

        # Forecast Assumptions
        if forecast_assumptions is None and 'forecast_assumptions' in raw_data:
            forecast_assumptions = ForecastAssumptions(**raw_data['forecast_assumptions'])
        elif forecast_assumptions is None:
            # Default fallback if missing
            forecast_assumptions = ForecastAssumptions(
                revenue_growth=[0.05]*5,
                ebitda_margin=[0.25]*5
            )

        # LBO Params
        if lbo_params is None and 'lbo_params' in raw_data:
            lbo_params = LBOParameters(**raw_data['lbo_params'])

        # Carve-out Params
        if carve_out_params is None and 'carve_out_params' in raw_data:
            carve_out_params = CarveOutParameters(**raw_data['carve_out_params'])

        # 1. Credit Metrics (Current/Historical)
        credit_metrics = self._calculate_credit_metrics(df_hist)

        # 2. Valuation (Simple EV and DCF)
        valuation_metrics = self._calculate_valuation(df_hist, forecast_assumptions, environment)

        # 3. LBO Analysis (if params exist)
        lbo_result = None
        if lbo_params:
            lbo_result = self._run_lbo(df_hist, lbo_params, forecast_assumptions, carve_out_params)

        # 4. Carve-out Impact
        carve_out_impact = 0.0
        if carve_out_params:
            last_ebitda = df_hist['ebitda'].iloc[-1] if not df_hist.empty else 0
            multiple = lbo_params.entry_multiple if lbo_params else 8.0
            carve_out_impact = -carve_out_params.standalone_cost_adjustments * multiple

        return ICATOutput(
            ticker=ticker,
            scenario_name=scenario_name,
            environment=environment,
            credit_metrics=credit_metrics,
            valuation_metrics=valuation_metrics,
            lbo_analysis=lbo_result,
            carve_out_impact=carve_out_impact,
            generated_at=datetime.utcnow().isoformat()
        )

    def _calculate_credit_metrics(self, df: pd.DataFrame) -> CreditMetrics:
        if df.empty:
            return CreditMetrics(
                pd_1yr=0, lgd=0, ltv=0, dscr=0, avg_dscr=0,
                interest_coverage=0, net_leverage=0, z_score=0, credit_rating="N/A"
            )

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

        A = wc / assets if assets else 0
        B = re_proxy / assets if assets else 0
        C = ebit / assets if assets else 0
        D = equity_proxy / liabilities if liabilities else 0
        E = latest.get('revenue', 0) / assets if assets else 0

        z_score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E

        # PD mapping (Simplified)
        # Z > 3.0 -> Safe (<0.1%)
        # 1.8 < Z < 3.0 -> Grey (1-5%)
        # Z < 1.8 -> Distress (>10%)
        if z_score > 2.99:
            pd_1yr = 0.001
            rating = "Investment Grade"
        elif z_score > 1.81:
            pd_1yr = 0.05
            rating = "High Yield"
        else:
            pd_1yr = 0.20
            rating = "Distressed"

        # LGD Mapping based on LTV
        # If LTV < 0.4 -> LGD low (20%)
        # If LTV > 0.8 -> LGD high (60%)
        lgd = 0.2 + (0.6 * min(max(ltv - 0.2, 0), 1.0)) # Linear map roughly

        return CreditMetrics(
            pd_1yr=round(pd_1yr, 4),
            lgd=round(lgd, 4),
            ltv=round(ltv, 4),
            dscr=round(dscr, 4),
            avg_dscr=round(dscr, 4), # Simplified: using current as proxy for avg
            interest_coverage=round(interest_coverage, 4),
            net_leverage=round(net_leverage, 4),
            z_score=round(z_score, 4),
            credit_rating=rating
        )

    def _calculate_valuation(self,
                             df: pd.DataFrame,
                             assumptions: ForecastAssumptions,
                             env: EnvironmentContext) -> ValuationMetrics:
        if df.empty:
            return ValuationMetrics(enterprise_value=0, equity_value=0)

        latest = df.iloc[-1]
        ebitda = latest.get('ebitda', 0)
        debt = latest.get('total_debt', 0)
        cash = latest.get('cash', 0)
        net_debt = debt - cash

        # Multiples
        # In a real system, we'd fetch peer multiples. Here we assume 8.0 as a baseline proxy.
        multiple = 8.0
        ev_multiple = ebitda * multiple
        equity_val_multiple = ev_multiple - net_debt

        # DCF
        wacc = assumptions.discount_rate if assumptions.discount_rate else (env.risk_free_rate + env.market_risk_premium) # simplified CAPM
        tgr = assumptions.terminal_growth_rate

        # Projection
        current_revenue = latest.get('revenue', ebitda/0.25 if ebitda else 100)

        dcf_sum = 0.0
        years = len(assumptions.revenue_growth)

        for i in range(years):
            g = assumptions.revenue_growth[i] if i < len(assumptions.revenue_growth) else assumptions.revenue_growth[-1]
            m = assumptions.ebitda_margin[i] if i < len(assumptions.ebitda_margin) else assumptions.ebitda_margin[-1]

            current_revenue *= (1 + g)
            proj_ebitda = current_revenue * m

            # Unlevered Free Cash Flow (UFCF)
            # UFCF = EBITDA * (1 - Tax) + D&A - Capex - Change in WC
            # Simplified: EBITDA * (1-Tax) - Capex - WC_Change
            # We assume D&A roughly equals Capex in steady state, but for growth companies Capex > D&A.
            # Here we follow the schema parameters.

            capex = current_revenue * assumptions.capex_percent_revenue
            # Change in WC is tricky without balance sheet. We assume WC is % of Revenue.
            # Change in WC = (Rev_t - Rev_t-1) * WC_pct
            prev_revenue = current_revenue / (1+g)
            wc_change = (current_revenue - prev_revenue) * assumptions.working_capital_percent_revenue

            taxes = proj_ebitda * assumptions.tax_rate # Simplified tax on EBITDA proxy

            ufcf = proj_ebitda - taxes - capex - wc_change

            dcf_sum += ufcf / ((1 + wacc) ** (i + 1))

            # Save terminal year metrics
            if i == years - 1:
                terminal_fcf = ufcf * (1 + tgr) # Grow one more year

        # Terminal Value (Gordon Growth)
        tv = terminal_fcf / (wacc - tgr)
        tv_discounted = tv / ((1 + wacc) ** years)

        dcf_ev = dcf_sum + tv_discounted
        dcf_equity = dcf_ev - net_debt

        return ValuationMetrics(
            enterprise_value=ev_multiple,
            equity_value=equity_val_multiple,
            trading_comps_value=ev_multiple,
            dcf_value=round(dcf_ev, 2),
            terminal_value_method="Gordon Growth"
        )

    def _run_lbo(self,
                 df: pd.DataFrame,
                 params: LBOParameters,
                 assumptions: ForecastAssumptions,
                 carve_out_params: Optional[CarveOutParameters] = None) -> LBOResult:

        latest = df.iloc[-1]
        ebitda_entry = latest.get('ebitda', 0)

        if carve_out_params:
            ebitda_entry -= carve_out_params.standalone_cost_adjustments

        # Sources and Uses
        entry_ev = ebitda_entry * params.entry_multiple
        total_debt_raised = sum(t.amount for t in params.debt_structure)

        # If equity contribution is fixed %, we adjust debt or fees?
        # Usually Sources = Uses.
        # Uses = EV + Fees. Sources = Debt + Equity.
        # Equity = Uses - Debt.
        uses = entry_ev + params.transaction_fees
        equity_check = uses - total_debt_raised

        # Projection
        years = params.forecast_years
        current_revenue = latest.get('revenue', ebitda_entry/0.25 if ebitda_entry else 100)

        # Create stateful debt tranches
        tranches = [t.model_copy() for t in params.debt_structure]
        # Sort by seniority (lower number = more senior)
        tranches.sort(key=lambda x: x.seniority)

        # Simulation
        for i in range(years):
            g = assumptions.revenue_growth[i] if i < len(assumptions.revenue_growth) else assumptions.revenue_growth[-1]
            m = assumptions.ebitda_margin[i] if i < len(assumptions.ebitda_margin) else assumptions.ebitda_margin[-1]

            current_revenue *= (1 + g)
            current_ebitda = current_revenue * m

            # Tax
            # Interest is tax deductible (simplified)
            total_interest_cash = sum(t.amount * t.interest_rate * (1 - t.pik_interest) for t in tranches)
            total_interest_pik = sum(t.amount * t.interest_rate * t.pik_interest for t in tranches)

            depreciation = current_revenue * assumptions.capex_percent_revenue # Proxy
            ebit = current_ebitda - depreciation

            taxable_income = ebit - (total_interest_cash + total_interest_pik)
            taxes = max(0, taxable_income * assumptions.tax_rate) # Assuming no NOLs for now

            # Cash Flow
            capex = current_revenue * assumptions.capex_percent_revenue
            # WC Change
            prev_rev = current_revenue / (1+g)
            wc_change = (current_revenue - prev_rev) * assumptions.working_capital_percent_revenue

            # CFADS (Cash Flow Available for Debt Service)
            # EBITDA - Taxes - Capex - WC_Change
            cfads = current_ebitda - taxes - capex - wc_change

            # Pay Interest (Cash)
            cf_after_interest = cfads - total_interest_cash

            # Pay Mandatory Amortization
            mandatory_amort = 0
            for t in tranches:
                amt = t.amount * t.amortization_rate
                # Check if enough cash
                paid = min(amt, max(0, cf_after_interest))
                t.amount -= paid
                cf_after_interest -= paid
                mandatory_amort += paid

            # Cash Sweep
            # Sweep remaining cash based on seniority and sweep share
            excess_cash = max(0, cf_after_interest)

            for t in tranches:
                if excess_cash <= 0: break
                if t.cash_sweep_share > 0:
                    sweep_potential = excess_cash * t.cash_sweep_share
                    paid = min(t.amount, sweep_potential)
                    t.amount -= paid
                    excess_cash -= paid # We assume swept cash is gone

            # PIK Accrual
            for t in tranches:
                pik_amt = t.amount * t.interest_rate * t.pik_interest
                t.amount += pik_amt

        # Exit
        # Assume last year EBITDA is the exit EBITDA
        exit_ebitda = current_ebitda
        exit_ev = exit_ebitda * params.exit_multiple
        exit_debt = sum(t.amount for t in tranches)
        exit_equity = exit_ev - exit_debt

        # Returns
        mom = exit_equity / equity_check if equity_check > 0 else 0

        if npf:
            flows = [-equity_check] + [0]*(years-1) + [exit_equity]
            irr = npf.irr(flows)
        else:
             irr = (mom ** (1/years)) - 1 if mom > 0 else -1

        debt_paydown = total_debt_raised - exit_debt # Note: can be negative if PIK > paydown

        return LBOResult(
            irr=round(irr, 4),
            mom_multiple=round(mom, 2),
            equity_value_entry=round(equity_check, 2),
            equity_value_exit=round(exit_equity, 2),
            debt_paydown=round(debt_paydown, 2),
            exit_enterprise_value=round(exit_ev, 2),
            exit_equity_value=round(exit_equity, 2)
        )
