# core/agents/Discussion_Chair_Agent.py

import json

class DiscussionChairAgent:
    def __init__(self, knowledge_base_path="knowledge_base/Knowledge_Graph.json"):
        """
        Initializes the Discussion Chair Agent.

        Args:
            knowledge_base_path (str): Path to the knowledge base file.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        Loads the knowledge base from the JSON file.

        Returns:
            dict: The knowledge base data.
        """
        try:
            with open(self.knowledge_base_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Knowledge base file not found: {self.knowledge_base_path}")
            return {}
        except json.JSONDecodeError:
            print(f"Error decoding knowledge base JSON: {self.knowledge_base_path}")
            return {}

    def make_final_decision(self, simulation_type, **kwargs):
        """
        Makes the final decision for either the Credit Rating Assessment
        Simulation or the Investment Committee Simulation, based on the
        provided simulation type and relevant data.

        Args:
            simulation_type (str): The type of simulation ("credit_rating" or "investment_committee").
            **kwargs: Keyword arguments containing the data and analysis results
                      for the specific simulation.

        Returns:
            tuple: The final decision and a rationale for the decision,
                   formatted appropriately for the simulation type.
        """

        if simulation_type == "credit_rating":
            return self._make_credit_rating_decision(**kwargs)
        elif simulation_type == "investment_committee":
            return self._make_investment_decision(**kwargs)
        else:
            raise ValueError("Invalid simulation type.")

    def _make_credit_rating_decision(self, shared_knowledge_graph):
        """
        Makes the final decision on the PD rating and regulatory rating
        after considering all inputs and recommendations from the
        Credit Rating Assessment Simulation.

        Args:
            shared_knowledge_graph (dict): The shared knowledge graph containing
                                         all the data and analysis from the simulation.

        Returns:
            tuple: (str, str, str): The final PD rating, final regulatory rating,
                                   and a justification for the decision.
        """

        # 1. Review all data and analysis in the shared_knowledge_graph
        company_name = shared_knowledge_graph["company_name"]
        financial_data = shared_knowledge_graph["financial_data"]
        pd_to_regulatory_rating_mapping = shared_knowledge_graph["pd_to_regulatory_rating_mapping"]
        credit_metrics = shared_knowledge_graph["credit_metrics"]
        dcf_forecast = shared_knowledge_graph["dcf_forecast"]
        industry_analysis = shared_knowledge_graph["industry_analysis"]
        company_narrative = shared_knowledge_graph["company_narrative"]
        initial_pd_ratings = shared_knowledge_graph["initial_pd_ratings"]
        # ... (Access other relevant data from shared_knowledge_graph)

        # 2. Consider potential disagreements and conflicting information
        #    - Identify any discrepancies between different analyses
        #    - Evaluate the reliability and relevance of each data source
        #    ... (Implementation for considering disagreements)

        # 3. Weigh the quantitative and qualitative factors
        #    - Balance the quantitative data (financial metrics, risk scores) with
        #      qualitative factors (management quality, industry outlook)
        #    ... (Implementation for weighing factors)

        # 4. Make the final decision on PD rating and regulatory rating
        #    ... (Implementation for making final decision)

        # 5. Provide a clear and concise justification for the decision,
        #    referencing relevant data and analysis
        #    ... (Implementation for generating justification)

        # Placeholder for decision-making logic
        # ... (Implement the actual decision-making logic here)

        final_pd_rating = "BBB"  # Example final PD rating
        final_regulatory_rating = "Pass"  # Example final regulatory rating
        justification = "Based on the comprehensive analysis and discussion, the company's strong financials, positive industry outlook, and experienced management team support a 'Pass' regulatory rating and a 'BBB' PD rating."

        return final_pd_rating, final_regulatory_rating, justification

    def _make_investment_decision(self, company_name, investment_amount, investment_horizon, fundamental_analysis, technical_analysis, risk_assessment, prediction_market_data, alternative_data, crypto_exposure):
        """
        Makes the final decision on the investment proposal after considering
        all inputs and recommendations from the Investment Committee Simulation.

        Args:
            company_name (str): The name of the company.
            investment_amount (float): The investment amount.
            investment_horizon (str): The investment horizon.
            fundamental_analysis (dict): Fundamental analysis results.
            technical_analysis (dict): Technical analysis results.
            risk_assessment (dict): Risk assessment results.
            prediction_market_data (dict): Prediction market data.
            alternative_data (dict): Alternative data.
            crypto_exposure (dict): Cryptocurrency exposure analysis.

        Returns:
            tuple: (str, str): The investment decision ("Approve" or "Reject") and
                               a rationale for the decision.
        """

        # 1. Review all data and analysis
        #    - Fundamental analysis: valuation, growth prospects, financial health
        #    - Technical analysis: price trends, momentum, support/resistance levels
        #    - Risk assessment: overall risk score, risk factors
        #    - Prediction market data: market sentiment, probability of events
        #    - Alternative data: social media sentiment, news sentiment
        #    - Cryptocurrency exposure: risk assessment, potential impact on portfolio

        # 2. Consider potential disagreements and conflicting information
        #    - Identify any discrepancies between different analyses
        #    - Evaluate the reliability and relevance of each data source

        # 3. Weigh the quantitative and qualitative factors
        #    - Balance the quantitative data (financial metrics, risk scores) with
        #      qualitative factors (management quality, industry outlook)

        # 4. Make the final decision on investment (Approve or Reject)
        #    - Consider investment objectives, risk tolerance, and investment horizon
        #    - Evaluate the potential return on investment compared to the risk

        # 5. Provide a clear and concise justification for the decision,
        #    referencing relevant data and analysis

        # Placeholder for decision-making logic
        # ... (Implement the actual decision-making logic here)

        # Example decision based on simplified criteria
        if fundamental_analysis["valuation"] == "Overvalued" and risk_assessment["overall_risk"] == "High":
            decision = "Reject"
            rationale = "The company appears overvalued with high risk, not meeting investment criteria."
        elif fundamental_analysis["valuation"] == "Undervalued" and risk_assessment["overall_risk"] == "Low":
            decision = "Approve"
            rationale = "The company appears undervalued with low risk, presenting a favorable investment opportunity."
        else:
            decision = "Hold"  # Introduce a "Hold" decision option
            rationale = "The investment decision requires further analysis and deliberation."

        return decision, rationale
