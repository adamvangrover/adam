# core/agents/financial_modeling_agent.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import logging
from typing import Dict, Any, Tuple, List
import openpyxl

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class FinancialModelingAgent:
    """
    Agent for performing comprehensive financial modeling, including DCF valuation, sensitivity analysis,
    stress testing, and detailed reporting. This agent determines the minimum complexity required to best model the company.
    """

    def __init__(self, initial_cash_flow=1000000, discount_rate=0.1, growth_rate=0.05, terminal_growth_rate=0.02, config: Dict[str, Any] = None):
        """
        Initializes the financial modeling agent with key parameters.

        Args:
            initial_cash_flow (float): The initial cash flow used in the DCF model.
            discount_rate (float): The discount rate used for the DCF model.
            growth_rate (float): The annual growth rate of cash flows.
            terminal_growth_rate (float): The perpetual growth rate for terminal value calculation.
            config (Dict[str, Any], optional): Configuration parameters for advanced modeling.
        """
        self.initial_cash_flow = initial_cash_flow
        self.discount_rate = discount_rate
        self.growth_rate = growth_rate
        self.terminal_growth_rate = terminal_growth_rate
        self.cash_flows = None
        self.discounted_cash_flows = None
        self.terminal_value = None
        self.npv = None
        self.config = config or {}
        self.forecast_years = self.config.get('forecast_years', 10)
        self.industry_multiples = self.config.get('industry_multiples', {'EBITDA': 10.0, 'Revenue': 2.0})
        self.terminal_valuation_method = self.config.get('terminal_valuation_method', 'Gordon Growth')
        self.data_sources = self.config.get('data_sources', {})
        self.company_name = None

    def generate_cash_flows(self, years=None, cash_flow_input=None):
        """
        Generates a forecast of cash flows over a number of years.

        Args:
            years (int, optional): The number of years for which the cash flows are to be forecasted. Defaults to forecast_years from config or 10.
            cash_flow_input (list or np.array, optional): Predefined cash flow list.

        Returns:
            np.array: An array containing the forecasted cash flows.
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

        Returns:
            np.array: An array of discounted cash flows.
        """
        if self.cash_flows is None:
            raise ValueError("Cash flows have not been generated.")

        discounted_cash_flows = self.cash_flows / (1 + self.discount_rate) ** np.arange(1, len(self.cash_flows) + 1)
        self.discounted_cash_flows = discounted_cash_flows
        return self.discounted_cash_flows

    def calculate_terminal_value(self):
        """
        Calculates the terminal value based on the final year's cash flow and the terminal growth rate.

        Returns:
            float: The calculated terminal value.
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

        Returns:
            float: The net present value (NPV).
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

        Args:
            sensitivity_range (list): A list of values for the sensitivity analysis.
            variable (str): The variable to be analyzed ('growth_rate', 'discount_rate').

        Returns:
            dict: Sensitivity results with npv for each variable value in the range.
        """
        results = {}
        original_variable = getattr(self, variable)
        for value in sensitivity_range:
            setattr(self, variable, value)
            npv = self.calculate_npv()
            results[value] = npv
        setattr(self, variable, original_variable)  # Reset to original value
        return results

    def perform_stress_testing(self, stress_factor=0.2):
        """
        Performs stress testing by applying a stress factor to key assumptions like cash flow, discount rate, and growth rate.

        Args:
            stress_factor (float): The factor by which to stress key assumptions (e.g., a 20% stress is 0.2).

        Returns:
            dict: Stress test results for NPV under stressed conditions.
        """
        stress_results = {}

        # Stress the cash flows (reduce by stress_factor)
        original_cash_flows = self.cash_flows.copy()
        stressed_cash_flows = self.cash_flows * (1 - stress_factor)
        self.cash_flows = stressed_cash_flows
        stressed_npv = self.calculate_npv()
        stress_results['cash_flows'] = stressed_npv
        self.cash_flows = original_cash_flows  # Reset cash flows

        # Stress the discount rate (increase by stress_factor)
        original_discount_rate = self.discount_rate
        stressed_discount_rate = self.discount_rate * (1 + stress_factor)
        self.discount_rate = stressed_discount_rate
        stressed_npv_discount_rate = self.calculate_npv()
        stress_results['discount_rate'] = stressed_npv_discount_rate
        self.discount_rate = original_discount_rate  # Reset discount rate

        # Stress the growth rate (reduce by stress_factor)
        original_growth_rate = self.growth_rate
        stressed_growth_rate = self.growth_rate * (1 - stress_factor)
        self.growth_rate = stressed_growth_rate
        stressed_npv_growth_rate = self.calculate_npv()
        stress_results['growth_rate'] = stressed_npv_growth_rate
        self.growth_rate = original_growth_rate  # Reset growth rate

        return stress_results

    def plot_sensitivity_analysis(self, sensitivity_range, variable='growth_rate'):
        """
        Plots the results of the sensitivity analysis.

        Args:
            sensitivity_range (list): A list of values for the sensitivity analysis.
            variable (str): The variable to be analyzed ('growth_rate', 'discount_rate').
        """
        sensitivity_results = self.perform_sensitivity_analysis(sensitivity_range, variable)

        plt.figure(figsize=(10, 6))
        plt.plot(sensitivity_results.keys(), sensitivity_results.values(), label=f'Sensitivity to {variable}')
        plt.title(f'Sensitivity Analysis of {variable}')
        plt.xlabel(f'{variable}')
        plt.ylabel('NPV')
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_stress_test_results(self, stress_results):
        """
        Plots the results of the stress testing.

        Args:
            stress_results (dict): The results of the stress test containing NPV under different conditions.
        """
        plt.figure(figsize=(10, 6))
        categories = ['Cash Flows', 'Discount Rate', 'Growth Rate']
        npvs = [stress_results['cash_flows'], stress_results['discount_rate'], stress_results['growth_rate']]

        plt.bar(categories, npvs, color='red')
        plt.title('Stress Testing Results')
        plt.ylabel('NPV')
        plt.show()

    def fetch_and_calculate_dcf(self, company_identifier: str, company_name: str = "Unknown Company") -> Tuple[float, Dict[str, Any], Dict[str, pd.DataFrame]]:
        """
        Fetches financial data, calculates DCF, and generates a comprehensive report.
        """
        self.company_name = company_name

        try:
            financial_data = self._fetch_financial_data(company_identifier)
            intrinsic_value, dcf_details = self.calculate_dcf(financial_data['forecast'])
            report_data = self._generate_comprehensive_report(financial_data, dcf_details, company_identifier)
            return intrinsic_value, dcf_details, report_data
        except Exception as e:
            logging.error(f"Error calculating DCF for {company_identifier}: {e}")
            return None, {}, {}

    def _fetch_financial_data(self, company_identifier: str) -> Dict[str, Any]:
        """
        Placeholder method to fetch financial data from data sources.
        """
        historical_data = {
            'revenue': [100, 110, 120, 130, 140],
            'ebitda_margin': [0.20, 0.21, 0.22, 0.23, 0.24],
            'capex': [10, 11, 12, 13, 14],
            'working_capital_change': [5, 6, 7, 8, 9],
            'revolver_debt': [10, 10, 10, 10, 10],
            'revolver_interest_rate': 0.04,
            'term_loan_a_debt': [20, 20, 20, 20, 20],
            'term_loan_a_interest_rate': 0.05,
            'term_loan_b_debt': [20, 20, 20, 20, 20],
            'term_loan_b_interest_rate': 0.06,
            'secured_notes_debt': [0, 0, 0, 0, 0],
            'secured_notes_interest_rate': 0.07,
            'unsecured_notes_debt': [0, 0, 0, 0, 0],
            'unsecured_notes_interest_rate': 0.08,
            'hybrid_debt': [0, 0, 0, 0, 0],
            'hybrid_debt_interest_rate': 0.09,
            'preferred_equity': [0, 0, 0, 0, 0],
            'common_equity': [100, 100, 100, 100, 100]
        }
        forecast_data = {
            'revenue_growth': [0.10] * 7,
            'ebitda_margin': [0.25] * 7,
            'capex_percent_revenue': [0.10] * 7,
            'working_capital_percent_revenue': [0.05] * 7,
            'revolver_repayment': [0] * 7,
            'term_loan_a_repayment': [2] * 7,
            'term_loan_b_repayment': [3] * 7,
            'secured_notes_repayment': [0] * 7,
            'unsecured_notes_repayment': [0] * 7,
            'hybrid_debt_repayment': [0] * 7,
            'revolver_spread': 0.02,
            'term_loan_a_spread': 0.03,
            'term_loan_b_spread': 0.04,
        }
        return {'historical': historical_data, 'forecast': forecast_data}

    def _generate_comprehensive_report(self, financial_data: Dict[str, Any], dcf_details: Dict[str, Any], company_identifier: str) -> Dict[str, pd.DataFrame]:
        """
        Generates a comprehensive financial report.
        """
        historical_df = pd.DataFrame(financial_data['historical'])
        forecast_df = self._generate_forecast_statements(financial_data['historical'], financial_data['forecast'])
        return {'historical': historical_df, 'forecast': forecast_df, 'dcf_details': pd.DataFrame([dcf_details])}

    def _generate_forecast_statements(self, historical_data: Dict[str, List[float]], forecast_data: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Generates forecast financial statements.
        """
        historical_df = pd.DataFrame(historical_data)
        forecast_df = pd.DataFrame()
        forecast_df['revenue'] = historical_df['revenue'].iloc[-1] * \
            (1 + np.array(forecast_data['revenue_growth']).cumprod())
        forecast_df['ebitda'] = forecast_df['revenue'] * np.array(forecast_data['ebitda_margin'])
        forecast_df['capex'] = forecast_df['revenue'] * np.array(forecast_data['capex_percent_revenue'])
        forecast_df['working_capital_change'] = forecast_df['revenue'] * \
            np.array(forecast_data['working_capital_percent_revenue'])
        forecast_df['free_cash_flow'] = forecast_df['ebitda'] - \
            forecast_df['capex'] - forecast_df['working_capital_change']
        return forecast_df


# Example usage:
if __name__ == "__main__":
    # Initialize the agent with some initial values
    config = {'forecast_years': 10, 'industry_multiples': {'EBITDA': 8.0,
                                                           'Revenue': 1.5}, 'terminal_valuation_method': 'Exit Multiple'}
    agent = FinancialModelingAgent(initial_cash_flow=1500000, discount_rate=0.08, growth_rate=0.04, config=config)

    # Generate cash flows and calculate NPV
    agent.generate_cash_flows()
    agent.calculate_discounted_cash_flows()
    agent.calculate_terminal_value()
    npv = agent.calculate_npv()

    print(f"NPV: {npv:.2f}")

    # Sensitivity analysis for growth rate
    sensitivity_range = np.linspace(0.02, 0.1, 10)
    sensitivity_results = agent.perform_sensitivity_analysis(sensitivity_range, variable='growth_rate')
    print(f"Sensitivity Analysis Results: {sensitivity_results}")

    # Plot sensitivity analysis
    agent.plot_sensitivity_analysis(sensitivity_range, variable='growth_rate')

    # Stress testing the model
    stress_results = agent.perform_stress_testing(stress_factor=0.2)
    print(f"Stress Test Results: {stress_results}")

    # Plot stress test results
    agent.plot_stress_test_results(stress_results)

    # Example fetch and calculate DCF
    company_id = "test_company"
    intrinsic_value, dcf_details, report_data = agent.fetch_and_calculate_dcf(company_id, company_name="Test Company")

    if intrinsic_value is not None:
        print(f"Intrinsic Value for {company_id}: {intrinsic_value:.2f}")
        print(f"DCF Details: {dcf_details}")
        print(f"Report Data Keys: {report_data.keys()}")
        if 'historical' in report_data:
            print(f"Historical Data:\n{report_data['historical'].head()}")
        if 'forecast' in report_data:
            print(f"Forecast Data:\n{report_data['forecast'].head()}")
        if 'dcf_details' in report_data:
            print(f"DCF Details in Report:\n{report_data['dcf_details']}")

        # Example: Exporting the report data to an Excel file
        with pd.ExcelWriter('financial_report.xlsx') as writer:
            report_data['historical'].to_excel(writer, sheet_name='Historical Data')
            report_data['forecast'].to_excel(writer, sheet_name='Forecast Data')
            report_data['dcf_details'].to_excel(writer, sheet_name='DCF Details')

        print("Report data exported to financial_report.xlsx")

    def calculate_dcf(self, financial_data: Dict[str, List[float]]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculates the DCF value based on the provided financial data.
        """
        fcf_projections = financial_data.get('free_cash_flow')
        if fcf_projections is None:
            raise ValueError("Free cash flow projections are missing.")

        if len(fcf_projections) < self.forecast_years:
            self.forecast_years = len(fcf_projections)

        discounted_fcf = []
        for year, fcf in enumerate(fcf_projections[:self.forecast_years]):
            discount_factor = 1 / (1 + self.discount_rate) ** (year + 1)
            discounted_fcf.append(fcf * discount_factor)

        if self.terminal_valuation_method == 'Gordon Growth':
            terminal_value = fcf_projections[self.forecast_years - 1] * \
                (1 + self.terminal_growth_rate) / (self.discount_rate - self.terminal_growth_rate)
        elif self.terminal_valuation_method == 'Exit Multiple':
            terminal_value = fcf_projections[self.forecast_years - 1] * self.industry_multiples.get("EBITDA", 10)
        else:
            raise ValueError("Invalid terminal valuation method.")

        terminal_value_discounted = terminal_value / (1 + self.discount_rate) ** self.forecast_years

        intrinsic_value = sum(discounted_fcf) + terminal_value_discounted

        detailed_calculations = {
            'discounted_fcf': discounted_fcf,
            'terminal_value': terminal_value,
            'terminal_value_discounted': terminal_value_discounted,
            'discount_rate': self.discount_rate,
            'terminal_growth_rate': self.terminal_growth_rate,
            'forecast_years': self.forecast_years,
            'terminal_valuation_method': self.terminal_valuation_method
        }
        return intrinsic_value, detailed_calculations

    def calculate_wacc(self, equity_market_value, debt_market_value, cost_of_equity, cost_of_debt, tax_rate):
        """Calculates the Weighted Average Cost of Capital (WACC)."""
        total_value = equity_market_value + debt_market_value
        equity_weight = equity_market_value / total_value
        debt_weight = debt_market_value / total_value
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        return wacc

    def _generate_forecast_statements(self, historical_data: Dict[str, List[float]], forecast_data: Dict[str, List[float]]) -> pd.DataFrame:
        """Generates forecast financial statements."""
        historical_df = pd.DataFrame(historical_data)
        forecast_df = pd.DataFrame()
        forecast_df['revenue'] = historical_df['revenue'].iloc[-1] * \
            (1 + np.array(forecast_data['revenue_growth']).cumprod())
        forecast_df['ebitda'] = forecast_df['revenue'] * np.array(forecast_data['ebitda_margin'])
        forecast_df['depreciation'] = forecast_df['ebitda'] * 0.1  # example depreciation
        forecast_df['ebit'] = forecast_df['ebitda'] - forecast_df['depreciation']
        forecast_df['interest_expense'] = 10  # example interest expense
        forecast_df['pretax_income'] = forecast_df['ebit'] - forecast_df['interest_expense']
        forecast_df['tax_expense'] = forecast_df['pretax_income'] * 0.25  # example tax rate
        forecast_df['net_income'] = forecast_df['pretax_income'] - forecast_df['tax_expense']
        forecast_df['capex'] = forecast_df['revenue'] * np.array(forecast_data['capex_percent_revenue'])
        forecast_df['working_capital_change'] = forecast_df['revenue'] * \
            np.array(forecast_data['working_capital_percent_revenue'])
        forecast_df['free_cash_flow'] = forecast_df['net_income'] + forecast_df['depreciation'] - \
            forecast_df['capex'] - forecast_df['working_capital_change']
        return forecast_df

    # ... (other methods)

# Example usage:
if __name__ == "__main__":
    # ... (previous example usage)
    agent = FinancialModelingAgent(initial_cash_flow=1500000, discount_rate=0.08, growth_rate=0.04, config=config)
    # ... (the rest of the example)
    wacc = agent.calculate_wacc(1000, 500, 0.1, 0.06, 0.25)
    print(f"WACC: {wacc:.2f}")
