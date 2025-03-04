#core/agents/financial_modeling_agent.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class FinancialModelingAgent:
    def __init__(self, initial_cash_flow=1000000, discount_rate=0.1, growth_rate=0.05, terminal_growth_rate=0.02):
        """
        Initializes the financial modeling agent with key parameters.

        Args:
            initial_cash_flow (float): The initial cash flow used in the DCF model.
            discount_rate (float): The discount rate used for the DCF model.
            growth_rate (float): The annual growth rate of cash flows.
            terminal_growth_rate (float): The perpetual growth rate for terminal value calculation.
        """
        self.initial_cash_flow = initial_cash_flow
        self.discount_rate = discount_rate
        self.growth_rate = growth_rate
        self.terminal_growth_rate = terminal_growth_rate
        self.cash_flows = None
        self.discounted_cash_flows = None
        self.terminal_value = None
        self.npv = None

    def generate_cash_flows(self, years=10):
        """
        Generates a forecast of cash flows over a number of years.

        Args:
            years (int): The number of years for which the cash flows are to be forecasted.

        Returns:
            np.array: An array containing the forecasted cash flows.
        """
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
        terminal_cash_flow = self.cash_flows[-1] * (1 + self.terminal_growth_rate)
        terminal_value = terminal_cash_flow / (self.discount_rate - self.terminal_growth_rate)
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

        npv = np.sum(self.discounted_cash_flows) + self.terminal_value / (1 + self.discount_rate) ** len(self.discounted_cash_flows)
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
        for value in sensitivity_range:
            if variable == 'growth_rate':
                self.growth_rate = value
            elif variable == 'discount_rate':
                self.discount_rate = value
            else:
                raise ValueError("Invalid variable for sensitivity analysis.")
            
            npv = self.calculate_npv()
            results[value] = npv

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
        stressed_cash_flows = self.cash_flows * (1 - stress_factor)
        self.cash_flows = stressed_cash_flows
        stressed_npv = self.calculate_npv()
        stress_results['cash_flows'] = stressed_npv

        # Stress the discount rate (increase by stress_factor)
        stressed_discount_rate = self.discount_rate * (1 + stress_factor)
        self.discount_rate = stressed_discount_rate
        stressed_npv_discount_rate = self.calculate_npv()
        stress_results['discount_rate'] = stressed_npv_discount_rate

        # Stress the growth rate (reduce by stress_factor)
        stressed_growth_rate = self.growth_rate * (1 - stress_factor)
        self.growth_rate = stressed_growth_rate
        stressed_npv_growth_rate = self.calculate_npv()
        stress_results['growth_rate'] = stressed_npv_growth_rate

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

# Example usage:
if __name__ == "__main__":
    # Initialize the agent with some initial values
    agent = FinancialModelingAgent(initial_cash_flow=1500000, discount_rate=0.08, growth_rate=0.04)

    # Generate cash flows and calculate NPV
    agent.generate_cash_flows(years=10)
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
