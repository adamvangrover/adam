# core/agents/financial_modeling_agent.py

import numpy as np
import numpy_financial as npf
import pandas as pd
import logging
from typing import Dict, Any, Tuple, List, Optional
from core.agents.agent_base import AgentBase
from core.financial_data.modeling_schema import FinancialAssumptions, ValuationResult, ValuationMethod, FinancialGlossary, DiscountedCashFlowModel, LBOAssumptions, LBOResult, LBOModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FinancialModelingAgent(AgentBase):
    """
    Agent for performing comprehensive financial modeling, including DCF valuation, sensitivity analysis,
    stress testing, Monte Carlo simulations, and ratio analysis.
    """

    def __init__(self, initial_cash_flow=None, discount_rate=None, growth_rate=None, terminal_growth_rate=None, config: Dict[str, Any] = None, constitution: Dict[str, Any] = None, kernel: Any = None):
        """
        Initializes the financial modeling agent with key parameters.
        Supports both config dict and direct arguments for backward compatibility.
        """
        if config is None:
            config = {}

        # Merge direct args into config if provided
        if initial_cash_flow is not None: config['initial_cash_flow'] = initial_cash_flow
        if discount_rate is not None: config['discount_rate'] = discount_rate
        if growth_rate is not None: config['growth_rate'] = growth_rate
        if terminal_growth_rate is not None: config['terminal_growth_rate'] = terminal_growth_rate

        super().__init__(config, constitution, kernel)

        # Initialize internal state using schema if possible, fallback to config
        self.assumptions = FinancialAssumptions(
            initial_cash_flow=self.config.get('initial_cash_flow', 1000000),
            discount_rate=self.config.get('discount_rate', 0.1),
            growth_rate=self.config.get('growth_rate', 0.05),
            terminal_growth_rate=self.config.get('terminal_growth_rate', 0.02),
            forecast_years=self.config.get('forecast_years', 10)
        )

        # Compatibility properties mapping to assumptions
        self.initial_cash_flow = self.assumptions.initial_cash_flow
        self.discount_rate = self.assumptions.discount_rate
        self.growth_rate = self.assumptions.growth_rate
        self.terminal_growth_rate = self.assumptions.terminal_growth_rate
        self.forecast_years = self.assumptions.forecast_years

        self.cash_flows = None
        self.discounted_cash_flows = None
        self.terminal_value = None
        self.npv = None
        self.industry_multiples = self.config.get('industry_multiples', {'EBITDA': 10.0, 'Revenue': 2.0})
        self.terminal_valuation_method = self.config.get('terminal_valuation_method', 'Gordon Growth')
        self.company_name = None

        self.glossary = FinancialGlossary()

    async def execute(self, *args, **kwargs):
        """
        Executes the main logic of the agent.
        Tasks:
        - "dcf": Standard Discounted Cash Flow analysis (default).
        - "monte_carlo": Monte Carlo simulation for valuation.
        - "ratios": Financial ratio analysis.
        """
        task = kwargs.get('task', 'dcf')
        company_id = kwargs.get('company_id', 'unknown')
        company_name = kwargs.get('company_name', 'Unknown Company')
        sentiment_score = kwargs.get('sentiment_score', 0.0) # -1.0 to 1.0

        logging.info(f"Executing FinancialModelingAgent task '{task}' for {company_name} ({company_id})")

        # Apply sentiment adjustment if provided
        if sentiment_score != 0.0:
            self.apply_sentiment_adjustment(sentiment_score)

        try:
            if task == 'monte_carlo':
                num_simulations = kwargs.get('num_simulations', 1000)
                results = self.run_monte_carlo_simulation(num_simulations)
                return {
                    "company_id": company_id,
                    "task": "monte_carlo",
                    "results": results,
                    "status": "success",
                    "assumptions": self.assumptions.model_dump()
                }

            elif task == 'ratios':
                financial_data = kwargs.get('financial_data')
                if not financial_data:
                    # Try fetch if not provided
                    financial_data = self._fetch_financial_data(company_id)['historical']

                ratios = self.calculate_financial_ratios(financial_data)
                return {
                    "company_id": company_id,
                    "task": "ratios",
                    "ratios": ratios,
                    "status": "success"
                }

            elif task == 'lbo':
                lbo_assumptions = kwargs.get('lbo_assumptions')
                if lbo_assumptions:
                    if isinstance(lbo_assumptions, dict):
                         # If dict passed, convert to Pydantic
                        self.lbo_assumptions = LBOAssumptions(**lbo_assumptions)
                    else:
                        self.lbo_assumptions = lbo_assumptions
                else:
                     # Use default or error? Let's use mock defaults for now if not provided
                    self.lbo_assumptions = LBOAssumptions(
                        entry_multiple=10.0,
                        exit_multiple=10.0,
                        initial_ebitda=kwargs.get('initial_ebitda', 100.0),
                        debt_amount=kwargs.get('debt_amount', 500.0),
                        interest_rate=kwargs.get('interest_rate', 0.08),
                        equity_contribution=kwargs.get('equity_contribution', 500.0)
                    )

                lbo_result = self.calculate_lbo()

                return {
                    "company_id": company_id,
                    "task": "lbo",
                    "result": lbo_result.model_dump(),
                    "assumptions": self.lbo_assumptions.model_dump(),
                    "status": "success"
                }

            else: # Default: DCF
                intrinsic_value, dcf_details, report_data = self.fetch_and_calculate_dcf(company_id, company_name)

                # Construct Schema-based Result
                valuation_result = ValuationResult(
                    intrinsic_value=intrinsic_value,
                    terminal_value=dcf_details['terminal_value'],
                    present_value_of_cash_flows=sum(dcf_details['discounted_fcf']),
                    assumptions_used=self.assumptions,
                    method=ValuationMethod.GORDON_GROWTH if self.terminal_valuation_method == 'Gordon Growth' else ValuationMethod.EXIT_MULTIPLE
                )

                result = {
                    "company_id": company_id,
                    "intrinsic_value": intrinsic_value,
                    "dcf_details": dcf_details,
                    "valuation_model": valuation_result.model_dump(),
                    "glossary": self.glossary.model_dump(),
                    "status": "success"
                }
                return result

        except Exception as e:
            logging.error(f"Error executing FinancialModelingAgent: {e}")
            return {"status": "error", "message": str(e)}

    def apply_sentiment_adjustment(self, sentiment_score: float):
        """
        Adjusts financial assumptions based on a sentiment score.
        Score range: -1.0 (Very Bearish) to 1.0 (Very Bullish)
        """
        # Logic:
        # Bullish (>0): Increases growth rate slightly, might lower discount rate (lower risk premium).
        # Bearish (<0): Decreases growth rate, increases discount rate.

        # Adjustment Factors
        growth_impact = 0.02 * sentiment_score # +/- 2%
        discount_impact = -0.01 * sentiment_score # Inverse: High sentiment -> Lower risk premium

        self.assumptions.growth_rate += growth_impact
        self.assumptions.discount_rate += discount_impact
        self.assumptions.sentiment_adjustment_factor = 1.0 + (0.1 * sentiment_score)

        # Sync back to instance vars
        self.growth_rate = self.assumptions.growth_rate
        self.discount_rate = self.assumptions.discount_rate

        logging.info(f"Applied sentiment adjustment (score: {sentiment_score}). New Growth: {self.growth_rate:.4f}, New Discount: {self.discount_rate:.4f}")

    def generate_cash_flows(self, years=None, cash_flow_input=None):
        """
        Generates a forecast of cash flows over a number of years.
        """
        years = years or self.forecast_years
        if cash_flow_input is not None:
            if not isinstance(cash_flow_input, (list, np.ndarray)):
                raise ValueError("cash_flow_input must be a list or numpy array.")
            self.cash_flows = np.array(cash_flow_input)
        else:
            cash_flows = []
            for year in range(1, years + 1):
                cash_flows.append(self.initial_cash_flow * (1 + self.growth_rate) ** year)
            self.cash_flows = np.array(cash_flows)
        return self.cash_flows

    def calculate_discounted_cash_flows(self):
        """
        Calculates the discounted cash flows (DCF) using the provided discount rate.
        """
        if self.cash_flows is None:
            raise ValueError("Cash flows have not been generated.")

        discounted_cash_flows = self.cash_flows / (1 + self.discount_rate) ** np.arange(1, len(self.cash_flows) + 1)
        self.discounted_cash_flows = discounted_cash_flows
        return self.discounted_cash_flows

    def calculate_terminal_value(self):
        """
        Calculates the terminal value based on the final year's cash flow and the terminal growth rate.
        """
        if self.cash_flows is None:
            raise ValueError("Cash flows have not been generated.")

        if self.terminal_valuation_method == 'Gordon Growth':
            terminal_cash_flow = self.cash_flows[-1] * (1 + self.terminal_growth_rate)
            terminal_value = terminal_cash_flow / (self.discount_rate - self.terminal_growth_rate)
        elif self.terminal_valuation_method == 'Exit Multiple':
            terminal_value = self.cash_flows[-1] * self.industry_multiples.get("EBITDA", 10)
        else:
            raise ValueError("Invalid terminal valuation method.")
        self.terminal_value = terminal_value
        return self.terminal_value

    def calculate_npv(self):
        """
        Calculates the net present value (NPV) of the investment based on discounted cash flows and terminal value.
        """
        if self.discounted_cash_flows is None:
            self.calculate_discounted_cash_flows()

        if self.terminal_value is None:
            self.calculate_terminal_value()

        npv = np.sum(self.discounted_cash_flows) + self.terminal_value / \
            (1 + self.discount_rate) ** len(self.discounted_cash_flows)
        self.npv = npv
        return self.npv

    def perform_sensitivity_analysis(self, sensitivity_range, variable='growth_rate'):
        """
        Performs sensitivity analysis on a given variable (e.g., growth rate, discount rate).
        """
        results = {}
        original_variable = getattr(self, variable)
        for value in sensitivity_range:
            setattr(self, variable, value)
            # Re-generate cash flows if growth rate changes
            if variable == 'growth_rate':
                self.generate_cash_flows()
            self.calculate_discounted_cash_flows()
            self.calculate_terminal_value()
            npv = self.calculate_npv()
            results[value] = npv

        # Reset to original state
        setattr(self, variable, original_variable)
        if variable == 'growth_rate':
            self.generate_cash_flows()
        self.calculate_discounted_cash_flows()
        self.calculate_terminal_value()

        return results

    def fetch_and_calculate_dcf(self, company_identifier: str, company_name: str = "Unknown Company") -> Tuple[float, Dict[str, Any], Dict[str, pd.DataFrame]]:
        """
        Fetches financial data, calculates DCF, and generates a comprehensive report.
        """
        self.company_name = company_name

        financial_data = self._fetch_financial_data(company_identifier)
        # Calculate Free Cash Flow for forecast
        fcf_forecast = self._calculate_fcf_forecast(financial_data)

        # Calculate DCF
        intrinsic_value, dcf_details = self.calculate_dcf_from_projections(fcf_forecast)

        report_data = {
            'historical': pd.DataFrame(financial_data['historical']),
            'forecast': pd.DataFrame(financial_data['forecast']),
            'dcf_details': dcf_details
        }
        return intrinsic_value, dcf_details, report_data

    def _fetch_financial_data(self, company_identifier: str) -> Dict[str, Any]:
        """
        Placeholder method to fetch financial data from data sources.
        """
        # Mock data
        historical_data = {
            'revenue': [100, 110, 120, 130, 140],
            'ebitda': [20, 22, 25, 28, 30],
            'ebit': [15, 17, 20, 23, 25],
            'interest_expense': [2, 2, 2, 2, 2],
            'total_debt': [50, 48, 46, 44, 42],
            'cash_and_equivalents': [10, 12, 15, 18, 20],
            'total_assets': [200, 210, 220, 230, 240],
            'current_assets': [50, 55, 60, 65, 70],
            'current_liabilities': [30, 32, 34, 36, 38],
            'ebitda_margin': [0.20, 0.21, 0.22, 0.23, 0.24],
            'capex': [10, 11, 12, 13, 14],
        }
        forecast_data = {
            'revenue_growth': [0.10] * 7,
            'ebitda_margin': [0.25] * 7,
            'capex_percent_revenue': [0.10] * 7,
            'working_capital_percent_revenue': [0.05] * 7,
        }
        return {'historical': historical_data, 'forecast': forecast_data}

    def _calculate_fcf_forecast(self, financial_data):
        """
        Helper to calculate FCF from forecast assumptions.
        """
        historical_df = pd.DataFrame(financial_data['historical'])
        forecast_assumptions = financial_data['forecast']

        last_revenue = historical_df['revenue'].iloc[-1]

        fcf_list = []
        current_revenue = last_revenue

        for i in range(len(forecast_assumptions['revenue_growth'])):
            growth = forecast_assumptions['revenue_growth'][i]
            margin = forecast_assumptions['ebitda_margin'][i]
            capex_pct = forecast_assumptions['capex_percent_revenue'][i]
            wc_pct = forecast_assumptions['working_capital_percent_revenue'][i]

            # Apply growth rates from assumptions
            # Overwrite mock forecast data if assumptions differ significantly?
            # Ideally, use self.growth_rate for revenue growth if standardizing.
            # But let's stick to the granular mock forecast for now.

            current_revenue *= (1 + growth)
            ebitda = current_revenue * margin
            capex = current_revenue * capex_pct
            # Simplified WC change
            wc_change = current_revenue * wc_pct

            # Simplified tax/interest for FCF
            tax_rate = self.assumptions.tax_rate
            depreciation = capex # steady state assumption
            ebit = ebitda - depreciation
            nopat = ebit * (1 - tax_rate)
            fcf = nopat + depreciation - capex - wc_change
            fcf_list.append(fcf)

        return fcf_list

    def calculate_dcf_from_projections(self, fcf_projections: List[float]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates the DCF value based on FCF projections.
        """
        if not fcf_projections:
            raise ValueError("Free cash flow projections are missing.")

        forecast_len = len(fcf_projections)

        discounted_fcf = []
        for year, fcf in enumerate(fcf_projections):
            discount_factor = 1 / (1 + self.discount_rate) ** (year + 1)
            discounted_fcf.append(fcf * discount_factor)

        # Terminal Value
        last_fcf = fcf_projections[-1]
        if self.terminal_valuation_method == 'Gordon Growth':
            terminal_value = last_fcf * (1 + self.terminal_growth_rate) / (self.discount_rate - self.terminal_growth_rate)
        elif self.terminal_valuation_method == 'Exit Multiple':
            terminal_value = last_fcf * self.industry_multiples.get("EBITDA", 10) # Using FCF as proxy for now
        else:
            terminal_value = 0

        terminal_value_discounted = terminal_value / (1 + self.discount_rate) ** forecast_len

        intrinsic_value = sum(discounted_fcf) + terminal_value_discounted

        detailed_calculations = {
            'discounted_fcf': discounted_fcf,
            'terminal_value': terminal_value,
            'terminal_value_discounted': terminal_value_discounted,
            'discount_rate': self.discount_rate,
            'terminal_growth_rate': self.terminal_growth_rate,
            'forecast_years': forecast_len,
            'terminal_valuation_method': self.terminal_valuation_method
        }
        return intrinsic_value, detailed_calculations

    def run_monte_carlo_simulation(self, num_simulations: int = 1000) -> Dict[str, Any]:
        """
        Performs Monte Carlo simulation by varying discount rate and growth rate.
        Assumes normal distribution for input variables.
        """
        npvs = []
        original_discount_rate = self.discount_rate
        original_growth_rate = self.growth_rate

        # Simulation parameters (standard deviation assumptions)
        discount_std = 0.01  # 1% standard deviation
        growth_std = 0.01    # 1% standard deviation

        for _ in range(num_simulations):
            # Sample parameters
            sim_discount = np.random.normal(original_discount_rate, discount_std)
            sim_growth = np.random.normal(original_growth_rate, growth_std)

            # Apply constraints (e.g., discount rate shouldn't be negative)
            sim_discount = max(0.01, sim_discount)

            # Update state
            self.discount_rate = sim_discount
            self.growth_rate = sim_growth
            self.generate_cash_flows()
            self.calculate_discounted_cash_flows()
            self.calculate_terminal_value()
            npvs.append(self.calculate_npv())

        # Restore state
        self.discount_rate = original_discount_rate
        self.growth_rate = original_growth_rate
        self.generate_cash_flows() # Restore cash flows based on original growth
        self.calculate_discounted_cash_flows()
        self.calculate_terminal_value()

        npvs = np.array(npvs)

        return {
            "mean_npv": float(np.mean(npvs)),
            "median_npv": float(np.median(npvs)),
            "std_dev": float(np.std(npvs)),
            "min_npv": float(np.min(npvs)),
            "max_npv": float(np.max(npvs)),
            "percentile_5": float(np.percentile(npvs, 5)),
            "percentile_95": float(np.percentile(npvs, 95)),
            "num_simulations": num_simulations
        }

    def calculate_financial_ratios(self, financial_data: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculates key financial ratios from historical data.
        Expected keys in financial_data: revenue, ebitda, ebit, interest_expense,
        total_debt, cash_and_equivalents, total_assets, current_assets, current_liabilities
        """
        # Use the most recent year (last element)
        idx = -1

        def get_val(key, default=0.0):
            val = financial_data.get(key)
            if isinstance(val, list) and len(val) > 0:
                return val[idx]
            return default

        revenue = get_val('revenue')
        ebitda = get_val('ebitda')
        ebit = get_val('ebit')
        interest = get_val('interest_expense')
        debt = get_val('total_debt')
        cash = get_val('cash_and_equivalents')
        assets = get_val('total_assets')
        curr_assets = get_val('current_assets')
        curr_liab = get_val('current_liabilities')

        ratios = {}

        # Profitability
        if revenue:
            ratios['EBITDA_Margin'] = ebitda / revenue
            ratios['Operating_Margin'] = ebit / revenue

        # Leverage
        if ebitda:
            ratios['Net_Debt_to_EBITDA'] = (debt - cash) / ebitda

        # Solvency
        if interest:
            ratios['Interest_Coverage'] = ebit / interest

        # Liquidity
        if curr_liab:
            ratios['Current_Ratio'] = curr_assets / curr_liab

        # Efficiency
        if assets:
            ratios['ROA'] = ebit / assets # Simplified (using EBIT)

        return {k: round(v, 4) for k, v in ratios.items()}

    def calculate_lbo(self) -> LBOResult:
        """
        Calculates LBO returns (IRR, MoM) based on assumptions.
        Simplified model:
        - EBITDA grows annually.
        - CapEx and WC are subtracted.
        - Tax is applied to EBIT.
        - FCF pays down debt (cash sweep).
        - Exit at end of holding period.
        """
        assumptions = self.lbo_assumptions
        years = assumptions.holding_period

        # Initialize State
        current_debt = assumptions.debt_amount
        current_ebitda = assumptions.initial_ebitda
        current_revenue = current_ebitda / assumptions.ebitda_margin # inferred revenue

        cash_flows_to_equity = []

        # Year 0: Investment
        cash_flows_to_equity.append(-assumptions.equity_contribution)

        # Forecast Years
        for year in range(1, years + 1):
            # Grow Operations
            current_revenue *= (1 + assumptions.revenue_growth)
            current_ebitda = current_revenue * assumptions.ebitda_margin

            # Expenses
            depreciation = current_revenue * assumptions.capex_percent_revenue # Steady state
            ebit = current_ebitda - depreciation
            interest = current_debt * assumptions.interest_rate

            # Tax
            taxable_income = max(0, ebit - interest)
            taxes = taxable_income * assumptions.tax_rate

            # Free Cash Flow
            # FCF = EBITDA - Interest - Taxes - CapEx - Change in WC
            capex = current_revenue * assumptions.capex_percent_revenue
            wc_change = current_revenue * assumptions.working_capital_percent_revenue

            fcf = current_ebitda - interest - taxes - capex - wc_change

            # Debt Paydown
            paydown = max(0, fcf * assumptions.debt_paydown_percent)
            current_debt = max(0, current_debt - paydown)

            # Cash flow to equity holders (dividends) - typically 0 in LBO until exit,
            # but let's assume remaining FCF is distributed or accumulated (value accretive)
            # For IRR calc, we usually only care about entry and exit unless there are dividends.
            # We will treat this as 0 for intermediate years for simplicity of typical LBO.
            cash_flows_to_equity.append(0.0)

        # Exit Year
        exit_enterprise_value = current_ebitda * assumptions.exit_multiple
        exit_equity_value = exit_enterprise_value - current_debt

        # Add exit proceeds to last year
        cash_flows_to_equity[-1] += exit_equity_value

        # Calculate MoM
        total_in = assumptions.equity_contribution
        total_out = exit_equity_value # + sum(dividends) if any
        mom = total_out / total_in if total_in > 0 else 0.0

        # Calculate IRR
        try:
            irr = npf.irr(cash_flows_to_equity)
            if np.isnan(irr): irr = 0.0
        except:
            irr = 0.0

        return LBOResult(
            irr=irr,
            mom_multiple=mom,
            exit_equity_value=exit_equity_value,
            final_debt=current_debt,
            cash_flows=cash_flows_to_equity
        )
