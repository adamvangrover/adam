# core/agents/industry_specialist_agent.py

import importlib
from core.utils.data_utils import send_message

class IndustrySpecialistAgent:
    def __init__(self, config):
        self.sector = config.get('sector', 'technology')  # Default to technology sector
        self.specialist = self.load_specialist(self.sector)

    def load_specialist(self, sector):
        """
        Dynamically loads the industry specialist module for the given sector.

        Args:
            sector (str): The name of the sector.

        Returns:
            object: An instance of the industry specialist class.
        """
        try:
            module = importlib.import_module(f"core.agents.industry_specialists.{sector}")
            class_name = sector.capitalize() + "Specialist"  # Example: TechnologySpecialist
            specialist_class = getattr(module, class_name)
            return specialist_class(self.config)  # Instantiate the specialist class
        except ImportError:
            print(f"No specialist found for sector: {sector}")
            return None

    def analyze_industry(self):
        """
        Analyzes the industry trends for the specified sector.

        Returns:
            dict: A dictionary containing industry trends and insights.
        """
        if self.specialist:
            trends = self.specialist.analyze_industry_trends()
            # Send trends to message queue
            message = {'agent': 'industry_specialist_agent', 'sector': self.sector, 'trends': trends}
            send_message(message)
            return trends
        else:
            print(f"No specialist loaded for sector: {self.sector}")
            return None

    def analyze_company(self, company_data):
        """
        Analyzes a company within the specified sector.

        Args:
            company_data (dict): Data about the company.

        Returns:
            dict: A dictionary containing company analysis results.
        """
        if self.specialist:
            analysis_results = self.specialist.analyze_company(company_data)
            # Send analysis results to message queue
            message = {
                'agent': 'industry_specialist_agent',
                'sector': self.sector,
                'company_analysis': analysis_results
            }
            send_message(message)
            return analysis_results
        else:
            print(f"No specialist loaded for sector: {self.sector}")
            return None
