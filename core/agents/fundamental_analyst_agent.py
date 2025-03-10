# core/agents/fundamental_analyst_agent.py

import csv
import os
from core.utils.data_utils import send_message

import logging
import pandas as pd
import numpy as np
from scipy import stats  # For statistical calculations (e.g., for DCF)
from typing import Dict, Any, Optional, Union
from core.agents.agent_base import AgentBase
from core.utils.config_utils import load_config
# Placeholder for message queue interaction (replace with real implementation later)
# from core.system.message_queue import MessageQueue

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FundamentalAnalystAgent(AgentBase):
    """
    Agent for performing fundamental analysis of companies.

    This agent analyzes financial statements, calculates key financial ratios,
    performs valuation modeling (DCF and comparables), and assesses financial health.
    """

    def __init__(self, config_path: str = "config/agents.yaml"):
        super().__init__()
        self.config = load_config(config_path)
        agent_config = self.config.get('agents', {}).get('FundamentalAnalystAgent', {})

        if not agent_config:
            logging.error("FundamentalAnalystAgent configuration not found.")
            raise ValueError("FundamentalAnalystAgent configuration not found.")

        self.persona = agent_config.get('persona', "Financial Analyst")
        self.description = agent_config.get('description', "Performs fundamental company analysis.")
        self.data_sources_config = load_config("config/data_sources.yaml")
        if not self.data_sources_config:
             logging.error("Failed to load data sources configuration.")
             raise FileNotFoundError("data_sources.yaml could not be loaded")


    def execute(self, company_id: str) -> Dict[str, Any]:
        """
        Performs fundamental analysis on a given company.

        Args:
            company_id: The ID of the company to analyze.

        Returns:
            A dictionary containing the analysis results.  This includes:
            - financial_ratios: Key financial ratios.
            - dcf_valuation:  DCF valuation (if possible).
            - comps_valuation: Valuation based on comparables (if possible).
            - financial_health:  An assessment of financial health (e.g., "Strong", "Moderate", "Weak").
            - analysis_summary: A textual summary of the analysis.
            - error: An error message, if any.

        """

        # Placeholder for retrieving company data (replace with actual data retrieval)
        # Assume this returns a dictionary like:
        # {
        #   "income_statement": { "revenue": [100, 110, 120], "net_income": [10, 12, 15] ... },
        #   "balance_sheet": { "assets": [200, 220, 240], "liabilities": [50, 55, 60], ... },
        #   "cash_flow_statement": { "operating_cash_flow": [15, 18, 22], ... },
        #   "historical_prices": [10, 11, 12, 11.5, 12.5],
        #   "industry": "Technology",
        #    "competitors": ["COMP1", "COMP2"]
        # }
        #  Error handling if not all this info is present.
        try:
            company_data = self.retrieve_company_data(company_id)  # You would implement this
            if company_data is None:
                return {"error": f"Could not retrieve data for company {company_id}"}

            financial_ratios = self.calculate_financial_ratios(company_data)
            dcf_valuation = self.calculate_dcf_valuation(company_data)
            comps_valuation = self.calculate_comps_valuation(company_data)
            financial_health = self.assess_financial_health(financial_ratios)
            analysis_summary = self.generate_analysis_summary(company_id, financial_ratios, dcf_valuation, comps_valuation, financial_health)

            return {
                "company_id": company_id,
                "financial_ratios": financial_ratios,
                "dcf_valuation": dcf_valuation,
                "comps_valuation": comps_valuation,
                "financial_health": financial_health,
                "analysis_summary": analysis_summary,
                "error": None  # No error
            }

        except Exception as e:
            logging.exception(f"Error during fundamental analysis of {company_id}: {e}")
            return {"error": f"An error occurred during analysis: {e}"}

    def retrieve_company_data(self, company_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves company data. Placeholder for actual data retrieval."""
        # TODO: Implement data retrieval from your data sources (files, APIs, etc.)
        # This is just placeholder data for demonstration.
        logging.warning("Using placeholder data for company analysis.")

        if company_id == "ABC":
             return {
                "name": "ABC Corp",
                "industry": "Technology",
                "financial_statements": {
                    "income_statement": {
                        "revenue": [1000, 1100, 1250],  # Example: 3 years of revenue
                        "net_income": [100, 120, 150],
                        "ebitda": [150, 170, 200]
                    },
                    "balance_sheet": {
                        "total_assets": [2000, 2100, 2200],
                        "total_liabilities": [800, 850, 900],
                        "shareholders_equity": [1200, 1250, 1300],
                        "cash_and_equivalents": [200, 250, 300],
                        "long_term_debt": [500,450, 400]
                    },
                     "cash_flow_statement": {
                        "operating_cash_flow": [180, 200, 230],
                        "investing_cash_flow": [-50, -60, -70],
                        "financing_cash_flow": [-30, -40, -50],
                        "free_cash_flow": [130, 140, 160]
                    }
                },
                "historical_prices": [50, 52, 55, 53, 58, 60],  # Example: last 6 periods
                "competitors": ["XYZ", "LMN"],
                "growth_rate": .05,
                "discount_rate": .10,
                "tax_rate": 0.25,
                "terminal_growth_rate": 0.03
            }
        else:
            return None

    def calculate_financial_ratios(self, company_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates key financial ratios.

        Args:
            company_data: Dictionary of company financial data.

        Returns:
            Dictionary of financial ratios.  Returns empty dict on error.
        """
        try:
            income_statement = company_data["financial_statements"]["income_statement"]
            balance_sheet = company_data["financial_statements"]["balance_sheet"]
            #cash_flow = company_data['financial_statements']['cash_flow_statement']
            # Ensure data availability and prevent index errors:
            if not income_statement or not balance_sheet:  # Check for None and empty
                logging.error("Missing financial statements for ratio calculation.")
                return {}
            if not income_statement["revenue"] or not income_statement["net_income"]:
                logging.error("Missing revenue or net_income in income statement")
                return {}
            if not balance_sheet["total_assets"] or not balance_sheet["total_liabilities"] or not balance_sheet["shareholders_equity"]:
                logging.error("Missing data in balance sheet")
                return {}
            if len(income_statement["revenue"]) == 0 or len(income_statement["net_income"]) == 0 :
                logging.error("Empty List")
                return {}

            # Use the *last* reported values for calculations (most recent).
            revenue = income_statement["revenue"][-1]
            net_income = income_statement["net_income"][-1]
            total_assets = balance_sheet["total_assets"][-1]
            total_liabilities = balance_sheet["total_liabilities"][-1]
            shareholders_equity = balance_sheet["shareholders_equity"][-1]
            #Add ebitda
            ebitda = 0
            if 'ebitda' in income_statement:
                if income_statement["ebitda"][-1]:
                    ebitda = income_statement["ebitda"][-1]
            # Calculate ratios (handle potential division by zero)
            ratios = {}

            ratios["revenue_growth"] = (revenue / income_statement["revenue"][-2] - 1) if len(income_statement["revenue"]) > 1 and income_statement["revenue"][-2] !=0 else None
            ratios["net_profit_margin"] = net_income / revenue if revenue else None
            ratios["return_on_equity"] = net_income / shareholders_equity if shareholders_equity else None
            ratios["debt_to_equity"] = total_liabilities / shareholders_equity if shareholders_equity else None
            ratios["current_ratio"] = balance_sheet["total_assets"][-1] / total_liabilities if total_liabilities else None
            if ebitda:
                ratios["ebitda_margin"] = ebitda / revenue if revenue else None
            else:
                ratios["ebitda_margin"] = None
            return ratios
        except (KeyError, IndexError, TypeError) as e:
            logging.exception(f"Error calculating financial ratios: {e}")
            return {} #Return empty in an exception

    def calculate_dcf_valuation(self, company_data: Dict[str, Any]) -> Optional[float]:
        """Calculates DCF valuation (simplified example)."""
        # Use get to safely extract values, and provide defaults for missing data
        try:
            cash_flows = company_data.get('financial_statements',{}).get("cash_flow_statement",{}).get("free_cash_flow", [])
            growth_rate = company_data.get("growth_rate", 0.05) # Assume some growth rate
            discount_rate = company_data.get("discount_rate", 0.10)
            terminal_growth_rate = company_data.get("terminal_growth_rate", 0.03)
            num_periods = len(cash_flows)

            if num_periods == 0:
                return None # No data for calculation
            # Calculate present value of projected cash flows
            present_values = []
            for i, cf in enumerate(cash_flows):
                pv = cf / ((1 + discount_rate) ** (i + 1))
                present_values.append(pv)

            # Calculate terminal value (using Gordon Growth Model) and its present value
            if num_periods > 0:
                terminal_value = (cash_flows[-1] * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)
                terminal_pv = terminal_value / ((1 + discount_rate) ** num_periods)
            else:
                terminal_pv = 0


            # Sum present values to get DCF valuation
            dcf_valuation = sum(present_values) + terminal_pv
            return dcf_valuation

        except (KeyError, TypeError, ZeroDivisionError) as e:
            logging.exception(f"Error during DCF calculation: {e}")
            return None # Indicate failure


    def calculate_comps_valuation(self, company_data: Dict[str, Any]) -> Optional[float]:
        """Calculates valuation based on comparables (simplified example)."""
        # Placeholder: Implement comparable company analysis
        # This is a simplified example.  In a real implementation, you'd:
        # 1.  Identify comparable companies.
        # 2.  Gather their valuation multiples (P/E, EV/EBITDA, etc.).
        # 3.  Calculate an average or median multiple.
        # 4.  Apply the multiple to the target company's metrics.
        logging.warning("Comps valuation not yet implemented.")
        return None

    def assess_financial_health(self, financial_ratios: Dict[str, float]) -> str:
        """
        Assesses the financial health of a company based on its ratios
        (simplified example).
        """
        # Placeholder: Implement a more sophisticated assessment.
        if not financial_ratios:
            return "Unknown"

        score = 0
        if financial_ratios.get("return_on_equity", 0) > 0.15:
            score += 2
        if financial_ratios.get("debt_to_equity", 1) < 1:
            score += 1
        if financial_ratios.get("current_ratio", 0) > 1.5:
            score += 1
        if financial_ratios.get("net_profit_margin", 0) > .1:
            score +=1
        if financial_ratios.get("revenue_growth", 0) != None:
            score += 1

        if score >= 4:
            return "Strong"
        elif score >= 2:
            return "Moderate"
        else:
            return "Weak"


    def generate_analysis_summary(self, company_id: str, financial_ratios: Dict[str, float],
                                  dcf_valuation: Optional[float], comps_valuation: Optional[float],
                                  financial_health: str) -> str:
        """
        Generates a textual summary of the fundamental analysis.

        Args:
            company_id: ID
            financial_ratios: Ratios
            dcf_valuation: DCF
            comps_valuation: Comps
            financial_health: Assessment

        Returns:
            A summary string.
        """
        summary = f"Fundamental Analysis for {company_id}:\n\n"

        if financial_ratios:
            summary += "Key Financial Ratios:\n"
            for ratio, value in financial_ratios.items():
                summary += f"  - {ratio}: {value:.2f}\n" if value is not None else f" -{ratio}: N/A\n"
        else:
            summary += "Financial ratios could not be calculated.\n"

        summary += f"Financial Health: {financial_health}\n"

        if dcf_valuation is not None:
            summary += f"DCF Valuation: {dcf_valuation:.2f}\n"
        else:
            summary += "DCF Valuation: Not available\n"

        if comps_valuation is not None:
            summary += f"Comps Valuation: {comps_valuation:.2f}\n"
        else:
            summary += "Comps Valuation: Not available\n"
        # Add additional summary details and context as needed.

        return summary


# Add additional summary details and context as needed.

        return summary

    def export_to_csv(self, data: Dict[str, List[Union[str, float]]], filename: str):
        """
        Exports financial statement data to a CSV file.

        Args:
            data: A dictionary where keys are column headers and values are lists of data.
            filename: The name of the CSV file to create.
        """
        filepath = os.path.join("data", filename) #Better for storage
        try:
            # Convert dictionary to list of lists for CSV writing
            headers = list(data.keys())
            rows = [headers]  # Start with headers
            # Transpose the data to create rows
            for row_values in zip(*data.values()):
                rows.append(list(row_values))

            with open(filepath, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(rows)
            logging.info(f"Successfully exported data to {filepath}")

        except Exception as e:
            logging.exception(f"Error exporting data to CSV: {e}")
            # Consider raising a custom exception here if you want calling code to handle this

    def calculate_growth_rate(self, financial_statements: Dict[str, Any], metric: str) -> Optional[float]:
        """
        Calculates the growth rate of a given metric over time.

        Args:
            financial_statements:  Dictionary containing financial statement data.
            metric: The name of the metric to calculate the growth rate for (e.g., "revenue").

        Returns:
            The growth rate as a float, or None if the growth rate cannot be calculated.
        """
        try:
            if metric not in financial_statements["income_statement"]:
                return None # Metric not present
            values = financial_statements["income_statement"][metric]
            if len(values) < 2:
                return None  # Not enough data to calculate growth

            # Calculate year-over-year growth. ((Current - Previous)/Previous)
            #  This is a simplification. More complex CAGR could be done.
            growth_rate = (values[-1] - values[-2]) / abs(values[-2]) if values[-2] != 0 else None
            return growth_rate

        except (KeyError, TypeError, IndexError) as e:
            logging.exception(f"Error calculating growth rate for {metric}: {e}")
            return None  # Indicate failure


    def calculate_ebitda_margin(self, financial_statements: Dict[str, Any]) -> Optional[float]:
        """Calculates EBITDA margin."""
        try:
            if "ebitda" not in financial_statements["income_statement"] or "revenue" not in financial_statements["income_statement"]:
                return None

            ebitda = financial_statements["income_statement"]["ebitda"][-1]
            revenue = financial_statements["income_statement"]["revenue"][-1]
            if revenue == 0:
                return None
            return ebitda / revenue
        except (KeyError, TypeError, IndexError) as e:
            logging.exception(f"Error during EBITDA margin: {e}")
            return None

    def calculate_dcf_valuation(self, company_data: Dict[str, Any]) -> Optional[float]:
        """
        Calculates the Discounted Cash Flow (DCF) valuation of the company.

        This is a simplified DCF model for demonstration purposes. A real-world
        DCF model would be significantly more complex.
        """
        try:
            # --- Input Data & Assumptions ---
            cash_flows = company_data['financial_statements']['cash_flow_statement']['free_cash_flow']
            if not cash_flows: #Empty list
                return None
            growth_rate = company_data.get('growth_rate', 0.05)  # Default growth rate
            discount_rate = company_data.get('discount_rate', 0.10)  # WACC (Weighted Average Cost of Capital)
            terminal_growth_rate = company_data.get('terminal_growth_rate', 0.03)
            num_periods = len(cash_flows)

            # --- Projection Period ---
            projected_cash_flows = []
            last_fcf = cash_flows[-1]
            for i in range(5): #Project for a fixed number
                next_fcf = last_fcf * (1 + growth_rate)
                projected_cash_flows.append(next_fcf)
                last_fcf = next_fcf #Update



            # --- Terminal Value ---
            # Gordon Growth Model:  TV = (FCF * (1 + g)) / (r - g)
            terminal_value = (projected_cash_flows[-1] * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)

            # --- Discounting ---
            present_values = []
            for i, fcf in enumerate(projected_cash_flows):
                pv = fcf / ((1 + discount_rate) ** (i + 1))
                present_values.append(pv)

            terminal_pv = terminal_value / ((1 + discount_rate) ** (len(projected_cash_flows)))

            # --- DCF Valuation ---
            dcf_valuation = sum(present_values) + terminal_pv
            return dcf_valuation

        except (KeyError, TypeError, ZeroDivisionError) as e:
            logging.exception(f"Error calculating DCF valuation: {e}")
            return None


    def calculate_enterprise_value(self, financial_statements: Dict[str, Any]) -> Optional[float]:
        """Calculates Enterprise Value (EV).  EV = Market Cap + Debt - Cash"""
        try:
            # Placeholder: Get Market Cap (you'd need stock price and shares outstanding)
            # For this example, we'll *simulate* a market cap calculation
            if "net_income" not in financial_statements["income_statement"] or not financial_statements["income_statement"]["net_income"]:
                return None
            net_income = financial_statements["income_statement"]["net_income"][-1]

            if "cash_and_equivalents" not in financial_statements["balance_sheet"] or not financial_statements["balance_sheet"]["cash_and_equivalents"]:
                return None
            cash = financial_statements["balance_sheet"]["cash_and_equivalents"][-1]

            if "long_term_debt" not in financial_statements["balance_sheet"] or not financial_statements["balance_sheet"]["long_term_debt"]:
                return None
            debt = financial_statements["balance_sheet"]["long_term_debt"][-1]

            #Simulate
            assumed_pe_ratio = 15
            market_cap = net_income * assumed_pe_ratio

            enterprise_value = market_cap + debt - cash
            return enterprise_value

        except (KeyError, TypeError) as e:
            logging.exception(f"Error calculating enterprise value: {e}")
            return None
    # --- Default Likelihood (Simulated, Placeholder) ---

    def estimate_default_likelihood(self, financial_ratios: Dict[str, float]) -> Optional[float]:
        """
        Estimates the likelihood of default (very simplified placeholder).

        In a real system, you would use a credit scoring model (like Altman Z-score),
        or integrate with a credit rating agency API.
        """
        logging.warning("Default likelihood estimation is a placeholder.  Returning a simulated value.")
        # Very basic example based on debt-to-equity
        debt_to_equity = financial_ratios.get("debt_to_equity")
        if debt_to_equity is None:
            return None
        if debt_to_equity > 2:
            return 0.2  # High risk
        elif debt_to_equity > 1:
            return 0.05 # Moderate Risk
        else:
            return 0.01  # Low risk


    # --- Distressed Metrics and Recovery (Simulated, Placeholder) ---
    def calculate_distressed_metrics(self, financial_statements: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculates distressed metrics (placeholder - very simplified).

        In a real system, you would calculate metrics like:
            - Interest Coverage Ratio
            - Cash Burn Rate
            - Liquidity Ratios
        """
        logging.warning("Distressed metrics calculation is a placeholder. Returning simulated values.")
        return {
            "interest_coverage_ratio": 2.5,  # Simulated
            "cash_burn_rate": -10,  # Simulated (negative = burning cash)
        }


    def estimate_recovery_rate(self, financial_statements, default_likelihood):
        """Placeholder, simulate recovery rate"""
        logging.warning("Recovery rate calculation is a placeholder. Returning simulated.")
        return .40 #Assume 40 cents on the dollar.

    def generate_analysis_summary(self, company_id: str, financial_ratios: Dict[str, float],
                                  dcf_valuation: Optional[float], comps_valuation: Optional[float],
                                  financial_health: str) -> str:
        """
        Generates a textual summary of the fundamental analysis.

        Args:
            company_id: The ID of the company.
            financial_ratios: A dictionary of calculated financial ratios.
            dcf_valuation: The DCF valuation (or None).
            comps_valuation: The comps valuation (or None).
            financial_health: The assessed financial health (e.g., "Strong").

        Returns:
            A summary string.
        """

        summary = f"Fundamental Analysis for {company_id}:\n\n"

        if financial_ratios:
            summary += "Key Financial Ratios:\n"
            for ratio, value in financial_ratios.items():
                summary += f"  - {ratio}: {value:.2f}\n" if value is not None else f"  - {ratio}: N/A\n"
        else:
            summary += "  Financial ratios could not be calculated.\n"

        summary += f"Financial Health: {financial_health}\n"

        if dcf_valuation is not None:
            summary += f"DCF Valuation: {dcf_valuation:.2f}\n"
        else:
            summary += "DCF Valuation: Not available\n"

        if comps_valuation is not None:
            summary += f"Comps Valuation: {comps_valuation:.2f}\n"
        else:
            summary += "Comps Valuation: Not available\n"

        # Add default likelihood, distressed metrics, and recovery
        default_likelihood = self.estimate_default_likelihood(financial_ratios)
        if default_likelihood is not None:
             summary += f"Estimated Default Likelihood: {default_likelihood:.2%}\n" #Percentage
        else:
             summary += f"Estimated Default Likelihood: N/A\n"

        distressed_metrics = self.calculate_distressed_metrics(financial_ratios)
        if distressed_metrics:
            summary += "Distressed Metrics:\n"
            for metric, value in distressed_metrics.items():
                summary += f"  - {metric}: {value:.2f}\n"

        recovery_rate = self.estimate_recovery_rate(financial_ratios,default_likelihood)
        if recovery_rate is not None:
            summary += f"Estimated Recovery Rate (in default): {recovery_rate:.2%}\n" #Percentage


        # TODO: Add more sophisticated narrative generation here (using an LLM in the future)
        return summary

    def send_message(self, message: Dict[str, Any]):
        """Placeholder to represent messages sent to Queue."""
        print("Sending message: ", message)

# Example Usage (for testing):
if __name__ == '__main__':

    #Example Agents.yaml
    example_config = """
    agents:
      FundamentalAnalystAgent:
        persona: "test persona"

    """
    with open("config/agents.yaml", "w") as file:
        yaml.dump(example_config, file)

    #Create dummy data_sources.yaml
    example_data_sources = """
    risk_ratings:
      type: json
      path: data/risk_rating_mapping.json
    market_baseline:
      type: json
      path: data/adam_market_baseline.json
    knowledge_base:
        type: json
        path: data/knowledge_base.json
    company_data_source:
        type: json
        path: data/company_data.json
    """
    with open("config/data_sources.yaml", "w") as file:
        yaml.dump(example_data_sources, file)

    agent = FundamentalAnalystAgent()
    analysis_result = agent.execute("ABC")
    print(analysis_result)
    os.remove("config/agents.yaml") #Remove agent file
    os.remove("config/data_sources.yaml") #Remove Data file
