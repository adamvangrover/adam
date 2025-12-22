# core/agents/discussion_chair_agent.py

import logging
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


# WIP /////////////////////////////////////////////////////


class DiscussionChairAgent:
    def __init__(self, knowledge_base_path="knowledge_base/Knowledge_Graph.json"):
        """
        Initializes the Discussion Chair Agent.

        Args:
            knowledge_base_path (str): Path to the knowledge base file, which contains the foundational data.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()

    def _load_knowledge_base(self):
        """
        Loads the knowledge base from the specified JSON file.

        Returns:
            dict: The knowledge base data loaded from the JSON file.

        Raises:
            FileNotFoundError: If the knowledge base file is not found.
            json.JSONDecodeError: If there is an error decoding the JSON data.
        """
        try:
            with open(self.knowledge_base_path, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logging.error(f"Knowledge base file not found: {self.knowledge_base_path}")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Error decoding knowledge base JSON: {self.knowledge_base_path}")
            return {}

    def make_final_decision(self, simulation_type, **kwargs):
        """
        Makes the final decision based on the simulation type and relevant data.

        Args:
            simulation_type (str): The type of simulation ("credit_rating" or "investment_committee").
            **kwargs: Additional arguments for the specific simulation type.

        Returns:
            tuple: A tuple containing the decision and rationale.

        Raises:
            ValueError: If the simulation type is invalid.
        """
        if simulation_type == "credit_rating":
            return self._make_credit_rating_decision(**kwargs)
        elif simulation_type == "investment_committee":
            return self._make_investment_decision(**kwargs)
        else:
            raise ValueError("Invalid simulation type. Please choose 'credit_rating' or 'investment_committee'.")

    def _make_credit_rating_decision(self, shared_knowledge_graph):
        """
        Makes the final decision on the PD (Probability of Default) rating and regulatory rating.

        Args:
            shared_knowledge_graph (dict): The shared knowledge graph containing all data and analysis for the credit rating.

        Returns:
            tuple: (str, str, str) - The final PD rating, regulatory rating, and the justification.
        """
        company_name = shared_knowledge_graph["company_name"]
        financial_data = shared_knowledge_graph["financial_data"]
        pd_to_regulatory_rating_mapping = shared_knowledge_graph["pd_to_regulatory_rating_mapping"]
        credit_metrics = shared_knowledge_graph["credit_metrics"]
        dcf_forecast = shared_knowledge_graph["dcf_forecast"]
        industry_analysis = shared_knowledge_graph["industry_analysis"]
        company_narrative = shared_knowledge_graph["company_narrative"]
        initial_pd_ratings = shared_knowledge_graph["initial_pd_ratings"]

        # Placeholder for decision-making logic
        # Implement more comprehensive checks and calculations here, based on the data.

        final_pd_rating = "BBB"  # Example decision based on simplified criteria
        final_regulatory_rating = "Pass"
        justification = (
            f"Based on the comprehensive analysis of {company_name}'s financials, "
            "positive industry outlook, and the company's experienced management team, "
            "the regulatory rating is 'Pass', and the PD rating is 'BBB'."
        )

        return final_pd_rating, final_regulatory_rating, justification

    def _make_investment_decision(self, company_name, investment_amount, investment_horizon, fundamental_analysis,
                                  technical_analysis, risk_assessment, prediction_market_data, alternative_data,
                                  crypto_exposure):
        """
        Makes the final decision on an investment proposal.

        Args:
            company_name (str): The name of the company.
            investment_amount (float): The amount proposed for investment.
            investment_horizon (str): The investment horizon (e.g., short-term, long-term).
            fundamental_analysis (dict): The results of the fundamental analysis (valuation, growth, etc.).
            technical_analysis (dict): The results of the technical analysis (price trends, momentum).
            risk_assessment (dict): The overall risk assessment of the company.
            prediction_market_data (dict): Market sentiment data and event probabilities.
            alternative_data (dict): Insights from alternative data (social media, sentiment analysis).
            crypto_exposure (dict): Analysis of the company's exposure to cryptocurrency.

        Returns:
            tuple: (str, str) - The investment decision ("Approve", "Reject", or "Hold") and the rationale.
        """
        # Placeholder for more advanced decision logic
        valuation = fundamental_analysis["valuation"]
        overall_risk = risk_assessment["overall_risk"]

        # Analyzing the relationship between fundamental analysis and risk
        if valuation == "Overvalued" and overall_risk == "High":
            decision = "Reject"
            rationale = (
                f"The company is overvalued and presents high risk, which does not meet our investment criteria."
            )
        elif valuation == "Undervalued" and overall_risk == "Low":
            decision = "Approve"
            rationale = (
                f"The company is undervalued and presents low risk, making it a promising investment opportunity."
            )
        elif valuation == "Fair" and overall_risk == "Medium":
            decision = "Hold"
            rationale = (
                "The company presents a balanced risk and reward profile; further analysis is required."
            )
        else:
            decision = "Hold"
            rationale = "Decision pending further analysis or updated data."

        return decision, rationale

    def log_decision(self, decision_type, decision, rationale):
        """
        Logs the final decision for traceability and audit purposes.

        Args:
            decision_type (str): Type of decision ("credit_rating" or "investment_committee").
            decision (str): The final decision ("Approve", "Reject", etc.).
            rationale (str): The rationale behind the decision.
        """
        logging.info(f"Decision Type: {decision_type}")
        logging.info(f"Decision: {decision}")
        logging.info(f"Rationale: {rationale}")


# WIP /////////////////////////////////////////////


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

        # Review all data and analysis in the shared_knowledge_graph
        company_name = shared_knowledge_graph["company_name"]
        financial_data = shared_knowledge_graph["financial_data"]
        pd_to_regulatory_rating_mapping = shared_knowledge_graph["pd_to_regulatory_rating_mapping"]
        credit_metrics = shared_knowledge_graph["credit_metrics"]
        dcf_forecast = shared_knowledge_graph["dcf_forecast"]
        industry_analysis = shared_knowledge_graph["industry_analysis"]
        company_narrative = shared_knowledge_graph["company_narrative"]
        initial_pd_ratings = shared_knowledge_graph["initial_pd_ratings"]

        # Conflict detection (example: PD ratings vs. Financial data)
        discrepancies = self._detect_conflicts(financial_data, initial_pd_ratings, industry_analysis)

        # Weigh quantitative vs qualitative factors
        final_pd_rating = self._weigh_quantitative_and_qualitative(financial_data, credit_metrics, industry_analysis)

        # Determine regulatory rating based on mapped PD
        final_regulatory_rating = pd_to_regulatory_rating_mapping.get(final_pd_rating, "Unknown")

        # Justification
        justification = f"Final PD rating: {final_pd_rating}, Regulatory rating: {final_regulatory_rating}. " \
            f"Discrepancies identified: {discrepancies}. Analysis points to the company's strong financials, but " \
            "conflicting industry outlook slightly impacts the rating."

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

        # Review all data and analysis
        all_data = {
            "fundamental_analysis": fundamental_analysis,
            "technical_analysis": technical_analysis,
            "risk_assessment": risk_assessment,
            "prediction_market_data": prediction_market_data,
            "alternative_data": alternative_data,
            "crypto_exposure": crypto_exposure
        }

        # Conflict detection (e.g., fundamental analysis vs. technical analysis)
        conflicts = self._detect_conflicts(fundamental_analysis, technical_analysis, risk_assessment)

        # Weigh quantitative vs qualitative factors
        decision = self._weigh_quantitative_and_qualitative_for_investment(fundamental_analysis, risk_assessment)

        # Justification
        rationale = f"Investment Decision: {decision}. Conflicts detected: {conflicts}. Further analysis suggests that the investment " \
            "is suitable based on fundamental analysis, but technical analysis and risk assessment show some caution."

        return decision, rationale

    def _detect_conflicts(self, *data_sources):
        """
        Detects conflicts between various data sources.

        Args:
            data_sources (tuple): Multiple data sources to be checked for conflicts.

        Returns:
            list: A list of detected conflicts or empty if no conflicts found.
        """
        conflicts = []
        # Simple example of conflict detection
        if "valuation" in data_sources[0] and "valuation" in data_sources[1]:
            if data_sources[0]["valuation"] != data_sources[1]["valuation"]:
                conflicts.append("Valuation mismatch between analyses.")

        return conflicts

    def _weigh_quantitative_and_qualitative(self, financial_data, credit_metrics, industry_analysis):
        """
        Weighs quantitative and qualitative factors in the credit rating decision.

        Args:
            financial_data (dict): The financial data.
            credit_metrics (dict): The credit metrics.
            industry_analysis (dict): The industry analysis.

        Returns:
            str: The final PD rating.
        """
        # Example: simplistic weighing logic
        if credit_metrics["debt_to_equity"] < 1.5 and financial_data["revenue_growth"] > 5:
            return "BBB"
        elif industry_analysis["sector_outlook"] == "Positive":
            return "A"
        else:
            return "BB"

    def _weigh_quantitative_and_qualitative_for_investment(self, fundamental_analysis, risk_assessment):
        """
        Weighs quantitative and qualitative factors in the investment decision.

        Args:
            fundamental_analysis (dict): The fundamental analysis results.
            risk_assessment (dict): The risk assessment results.

        Returns:
            str: The investment decision.
        """
        if fundamental_analysis["valuation"] == "Undervalued" and risk_assessment["overall_risk"] == "Low":
            return "Approve"
        else:
            return "Reject"
