# core/agents/industry_specialist_agent.py

import importlib
import logging
import asyncio
from typing import Dict, Any, Optional
from core.agents.agent_base import AgentBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IndustrySpecialistAgent(AgentBase):
    """
    Agent specializing in industry analysis by dynamically loading sector specialists.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        """
        Initializes the Industry Specialist Agent.

        Args:
            config (dict): Configuration dictionary.
            kernel (Optional[Any]): Semantic Kernel instance.
        """
        super().__init__(config, kernel=kernel)
        self.sector = self.config.get('sector', 'technology')  # Default to technology sector
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
            logger.warning(f"No specialist found for sector: {sector}. Using generic fallback.")
            return None
        except Exception as e:
            logger.error(f"Error loading specialist for {sector}: {e}")
            return None

    async def execute(self, *args, **kwargs):
        """
        Executes industry analysis tasks.

        Tasks:
        - "analyze_industry": Analyzes overall industry trends.
        - "analyze_company": Analyzes a specific company in the sector.
        """
        task = kwargs.get('task')
        logger.info(f"IndustrySpecialistAgent ({self.sector}) executing task: {task}")

        if task == "analyze_industry":
            return self.analyze_industry()

        elif task == "analyze_company":
            company_data = kwargs.get("company_data", {})
            return self.analyze_company(company_data)

        else:
            logger.warning(f"Unknown task: {task}")
            return {"error": f"Unknown task: {task}"}

    def analyze_industry(self):
        """
        Analyzes the industry trends for the specified sector.

        Returns:
            dict: A dictionary containing industry trends and insights.
        """
        if self.specialist:
            trends = self.specialist.analyze_industry_trends()
            # Send trends to message queue (Legacy support)
            try:
                message = {'agent': 'industry_specialist_agent', 'sector': self.sector, 'trends': trends}
            except Exception as e:
                logger.warning(f"Failed to send legacy message: {e}")
            return trends
        else:
            logger.warning(f"No specialist loaded for sector: {self.sector}")
            return {"status": "No specialist loaded", "sector": self.sector}

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
            # Send analysis results to message queue (Legacy support)
            try:
                message = {
                    'agent': 'industry_specialist_agent',
                    'sector': self.sector,
                    'company_analysis': analysis_results
                }
            except Exception as e:
                logger.warning(f"Failed to send legacy message: {e}")
            return analysis_results
        else:
            logger.warning(f"No specialist loaded for sector: {self.sector}")
            return {"status": "No specialist loaded", "sector": self.sector}

if __name__ == "__main__":
    # Mock config
    config = {'sector': 'technology'}
    agent = IndustrySpecialistAgent(config)
    async def main():
        res = await agent.execute(task="analyze_industry")
        print(res)
    asyncio.run(main())
