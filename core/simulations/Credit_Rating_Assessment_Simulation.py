# core/simulations/Credit_Rating_Assessment_Simulation.py

import json
from core.agents.snc_analyst_agent import SNCAnalystAgent
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.agents.industry_specialist_agent import IndustrySpecialistAgent
from core.agents.discussion_chair_agent import DiscussionChairAgent  # Import the Discussion Chair Agent


class CreditRatingAssessmentSimulation:
    def __init__(self, knowledge_base_path="knowledge_base/Knowledge_Graph.json"):
        """
        Initializes the Credit Rating Assessment Simulation.

        Args:
            knowledge_base_path (str): Path to the knowledge base file.
        """
        self.knowledge_base_path = knowledge_base_path
        self.knowledge_base = self._load_knowledge_base()

        # Initialize agents with their respective roles
        self.credit_analyst_1 = FundamentalAnalystAgent(knowledge_base_path)
        self.credit_analyst_2 = IndustrySpecialistAgent(knowledge_base_path)
        self.team_lead = SNCAnalystAgent(knowledge_base_path)
        self.discussion_chair = DiscussionChairAgent(knowledge_base_path)  # Initialize the Discussion Chair Agent

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

    def run_simulation(self, company_name, financial_data):
        """
        Runs the credit rating assessment simulation for a given company.

        Args:
            company_name (str): The name of the company.
            financial_data (dict): Financial data of the company.
        """

        # Initialize shared knowledge graph
        shared_knowledge_graph = {
            "company_name": company_name,
            "financial_data": financial_data,
            "pd_to_regulatory_rating_mapping": {
                "S&P_Rating": ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-", "BB+", "BB", "BB-", "B+", "B", "B-", "CCC+", "CCC", "CCC-", "CC", "C", "D"],
                "Regulatory_Rating": ["Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Pass", "Special Mention", "Special Mention", "Substandard", "Substandard", "Substandard", "Doubtful", "Doubtful", "Doubtful", "Loss", "Loss", "Loss", "Loss", "Loss"]
            },
            "credit_metrics": None,
            "dcf_forecast": None,
            "industry_analysis": None,
            "company_narrative": None,
            "initial_pd_ratings": {},
            "initial_regulatory_ratings": {},
            "final_pd_rating_recommendation": None,
            "final_regulatory_rating_recommendation": None,
            "final_pd_rating_decision": None,
            "final_regulatory_rating_decision": None,
            "justification": None,
            "discussion_transcript": ""
        }

        # 1. Credit Analyst 1 Analysis
        credit_metrics = self.credit_analyst_1.analyze_financial_statements(financial_data)
        dcf_forecast = self.credit_analyst_1.perform_dcf_valuation(financial_data)
        initial_pd_rating_1, initial_regulatory_rating_1, justification_1 = self.credit_analyst_1.propose_initial_ratings(
            credit_metrics, dcf_forecast, shared_knowledge_graph["pd_to_regulatory_rating_mapping"])

        shared_knowledge_graph["credit_metrics"] = credit_metrics
        shared_knowledge_graph["dcf_forecast"] = dcf_forecast
        shared_knowledge_graph["initial_pd_ratings"]["Credit Analyst 1"] = initial_pd_rating_1
        shared_knowledge_graph["initial_regulatory_ratings"]["Credit Analyst 1"] = initial_regulatory_rating_1
        shared_knowledge_graph[
            "discussion_transcript"] += f"## Credit Analyst 1:\n\n* **Initial PD Rating:** {initial_pd_rating_1}\n* **Initial Regulatory Rating:** {initial_regulatory_rating_1}\n* **Justification:** {justification_1}\n\n"

        # 2. Credit Analyst 2 Analysis
        industry_analysis = self.credit_analyst_2.analyze_industry(company_name)
        company_narrative = self.credit_analyst_2.evaluate_company_narrative(company_name)
        initial_pd_rating_2, initial_regulatory_rating_2, justification_2 = self.credit_analyst_2.propose_initial_ratings(
            industry_analysis, company_narrative, shared_knowledge_graph["pd_to_regulatory_rating_mapping"])

        shared_knowledge_graph["industry_analysis"] = industry_analysis
        shared_knowledge_graph["company_narrative"] = company_narrative
        shared_knowledge_graph["initial_pd_ratings"]["Credit Analyst 2"] = initial_pd_rating_2
        shared_knowledge_graph["initial_regulatory_ratings"]["Credit Analyst 2"] = initial_regulatory_rating_2
        shared_knowledge_graph[
            "discussion_transcript"] += f"## Credit Analyst 2:\n\n* **Initial PD Rating:** {initial_pd_rating_2}\n* **Initial Regulatory Rating:** {initial_regulatory_rating_2}\n* **Justification:** {justification_2}\n\n"

        # 3. Team Lead Moderation and Recommendation
        final_pd_rating_recommendation, final_regulatory_rating_recommendation, team_lead_justification = self.team_lead.moderate_discussion(
            shared_knowledge_graph)

        shared_knowledge_graph["final_pd_rating_recommendation"] = final_pd_rating_recommendation
        shared_knowledge_graph["final_regulatory_rating_recommendation"] = final_regulatory_rating_recommendation
        shared_knowledge_graph[
            "discussion_transcript"] += f"## Team Lead:\n\n* **Final PD Rating Recommendation:** {final_pd_rating_recommendation}\n* **Final Regulatory Rating Recommendation:** {final_regulatory_rating_recommendation}\n* **Justification:** {team_lead_justification}\n\n"

        # 4. Discussion Chair Decision
        final_pd_rating_decision, final_regulatory_rating_decision, chair_justification = self.discussion_chair.make_final_decision(
            shared_knowledge_graph)

        shared_knowledge_graph["final_pd_rating_decision"] = final_pd_rating_decision
        shared_knowledge_graph["final_regulatory_rating_decision"] = final_regulatory_rating_decision
        shared_knowledge_graph["justification"] = chair_justification
        shared_knowledge_graph[
            "discussion_transcript"] += f"## Discussion Chair:\n\n* **Final PD Rating Decision:** {final_pd_rating_decision}\n* **Final Regulatory Rating Decision:** {final_regulatory_rating_decision}\n* **Justification:** {chair_justification}\n\n"

        # 5. Generate Output
        output = {
            "company_name": shared_knowledge_graph["company_name"],
            "final_pd_rating": shared_knowledge_graph["final_pd_rating_decision"],
            "final_regulatory_rating": shared_knowledge_graph["final_regulatory_rating_decision"],
            "justification": shared_knowledge_graph["justification"],
            "discussion_transcript": shared_knowledge_graph["discussion_transcript"]
        }

        # 6. Save Results
        self.save_results(output)

    def save_results(self, output):
        """
        Saves the simulation results to a JSON file.

        Args:
            output (dict): The output of the simulation.
        """
        # Placeholder for saving results logic
        # ...
        pass
