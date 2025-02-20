# core/agents/industry_specialist_agent.py

class IndustrySpecialistAgent:
    def __init__(self, config):
        self.sector = config['sector']
        self.specialist = self.load_specialist(self.sector)

    def load_specialist(self, sector):
        # Dynamically import the appropriate sub-module
        try:
            module = __import__(f"core.agents.industry_specialists.{sector}", 
                               fromlist=[f"{sector.capitalize()}Specialist"])
            return getattr(module, f"{sector.capitalize()}Specialist")()
        except ImportError:
            print(f"No specialist found for sector: {sector}")
            return None

    def analyze_industry(self):
        if self.specialist:
            return self.specialist.analyze_industry_trends()
        else:
            return None

    def analyze_company(self, company_data):
        if self.specialist:
            return self.specialist.analyze_company(company_data)
        else:
            return None
