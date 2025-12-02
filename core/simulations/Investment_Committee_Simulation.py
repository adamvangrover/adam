# core/simulations/Investment_Committee_Simulation.py

import json
from datetime import datetime
from utils.api_communication import APICommunication
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.agents.technical_analyst_agent import TechnicalAnalystAgent
from core.agents.risk_assessment_agent import RiskAssessmentAgent
from core.agents.prediction_market_agent import PredictionMarketAgent
from core.agents.alternative_data_agent import AlternativeDataAgent
from core.agents.crypto_agent import CryptoAgent
from core.agents.discussion_chair_agent import DiscussionChairAgent  # Import the Discussion Chair Agent

class InvestmentCommitteeSimulation:
    def __init__(self, knowledge_base_path="knowledge_base/Knowledge_Graph.json", wsm_model_path="world_simulation_model/WSM_v7.1.json"):
        """
        Initializes the Investment Committee Simulation.

        Args:
            knowledge_base_path (str): Path to the knowledge base file.
            wsm_model_path (str): Path to the World Simulation Model file.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_json(knowledge_base_path)
        self.wsm_model_path = wsm_model_path
        self.wsm_model = self._load_json(wsm_model_path)
        self.api_communication = APICommunication()

        # Initialize agents
        self.fundamental_analyst = FundamentalAnalystAgent(knowledge_base_path)
        self.technical_analyst = TechnicalAnalystAgent(knowledge_base_path)
        self.risk_assessment_agent = RiskAssessmentAgent(knowledge_base_path)
        self.prediction_market_agent = PredictionMarketAgent(knowledge_base_path)
        self.alternative_data_agent = AlternativeDataAgent(knowledge_base_path)
        self.crypto_agent = CryptoAgent(knowledge_base_path)
        self.discussion_chair = DiscussionChairAgent(knowledge_base_path)  # Initialize the Discussion Chair Agent

    def _load_json(self, file_path):
        """
        Loads a JSON file.

        Args:
            file_path (str): Path to the JSON file.

        Returns:
            dict: The loaded JSON data.
        """
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {file_path}")
            return {}

    def run_simulation(self, company_name, investment_amount, investment_horizon):
        """
        Runs the investment committee simulation for a given company.

        Args:
            company_name (str): The name of the company.
            investment_amount (float): The investment amount.
            investment_horizon (str): The investment horizon (e.g., "1 year", "5 years").
        """
        # 1. Gather Data
        financial_data = self.api_communication.get_financial_data(company_name)
        market_data = self.api_communication.get_market_data(company_name)

        # 2. Agent Analysis
        fundamental_analysis = self.fundamental_analyst.analyze_company(company_name, financial_data)
        technical_analysis = self.technical_analyst.analyze_market_data(market_data)
        risk_assessment = self.risk_assessment_agent.assess_investment_risk(company_name, financial_data, market_data)
        prediction_market_data = self.prediction_market_agent.gather_prediction_market_data(company_name)
        alternative_data = self.alternative_data_agent.gather_alternative_data(company_name)
        crypto_exposure = self.crypto_agent.analyze_crypto_market(company_name)

        # 3. Committee Discussion and Decision
        # Pass all analysis results to the Discussion Chair Agent for final decision
        decision, rationale = self.discussion_chair.make_final_decision(
            company_name, investment_amount, investment_horizon,
            fundamental_analysis, technical_analysis, risk_assessment,
            prediction_market_data, alternative_data, crypto_exposure
        )

        # 4. Generate Report
        report = self.generate_report(
            company_name, investment_amount, investment_horizon,
            decision, rationale,
            fundamental_analysis, technical_analysis, risk_assessment,
            prediction_market_data, alternative_data, crypto_exposure
        )

        # 5. Save Results
        self.save_results(company_name, decision, report)

    def generate_report(
            self, company_name, investment_amount, investment_horizon,
            decision, rationale,
            fundamental_analysis, technical_analysis, risk_assessment,
            prediction_market_data, alternative_data, crypto_exposure
    ):
        """
        Generates an investment committee report.

        Args:
            company_name (str): The name of the company.
            investment_amount (float): The investment amount.
            investment_horizon (str): The investment horizon.
            decision (str): The investment decision.
            rationale (str): The rationale for the decision.
            fundamental_analysis (dict): Fundamental analysis results.
            technical_analysis (dict): Technical analysis results.
            risk_assessment (dict): Risk assessment results.
            prediction_market_data (dict): Prediction market data.
            alternative_data (dict): Alternative data.
            crypto_exposure (dict): Cryptocurrency exposure analysis.

        Returns:
            str: The generated report.
        """
        # Placeholder for report generation logic
        # This should involve formatting the data and analysis results
        # into a human-readable report.
        # ...

        report = f"""
        Investment Committee Report
        Company: {company_name}
        Investment Amount: {investment_amount}
        Investment Horizon: {investment_horizon}

        Decision: {decision}
        Rationale: {rationale}

        Fundamental Analysis:
        {fundamental_analysis}

        Technical Analysis:
        {technical_analysis}

        Risk Assessment:
        {risk_assessment}

        Prediction Market Data:
        {prediction_market_data}

        Alternative Data:
        {alternative_data}

        Cryptocurrency Exposure:
        {crypto_exposure}
        """

        return report

    def save_results(self, company_name, decision, report):
        """
        Saves the simulation results to the knowledge base and a report file.

        Args:
            company_name (str): The name of the company.
            decision (str): The investment decision.
            report (str): The generated report.
        """
        # Placeholder for saving results logic
        # This should involve updating the knowledge base and saving the report
        # to a file.
        # ...

        # Example: Save results to knowledge base
        if "investment_committee_simulations" not in self.knowledge_base:
            self.knowledge_base["investment_committee_simulations"] = {}
        self.knowledge_base["investment_committee_simulations"][company_name] = {
            "decision": decision,
            "report": report,
            "timestamp": datetime.now().isoformat()
        }

        # Example: Save report to file
        with open(f"libraries_and_archives/simulation_results/{company_name}_investment_committee_report.txt", 'w') as f:
            f.write(report)
