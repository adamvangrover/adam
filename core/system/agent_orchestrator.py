# core/system/agent_orchestrator.py

from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.agents.macroeconomic_analysis_agent import MacroeconomicAnalysisAgent
from core.agents.geopolitical_risk_agent import GeopoliticalRiskAgent
from core.agents.industry_specialist_agent import IndustrySpecialistAgent
from core.agents.fundamental_analyst_agent import FundamentalAnalyst
from core.agents.technical_analyst_agent import TechnicalAnalyst
from core.agents.risk_assessment_agent import RiskAssessor
from core.agents.newsletter_layout_specialist_agent import NewsletterLayoutSpecialist
from core.agents.data_verification_agent import DataVerificationAgent
from core.agents.lexica_agent import LexicaAgent
from core.agents.archive_manager_agent import ArchiveManagerAgent

class AgentOrchestrator:
    def __init__(self, agents_config):
        self.agents = {}
        for agent_name, config in agents_config.items():
            if agent_name == "market_sentiment_agent":
                self.agents[agent_name] = MarketSentimentAgent(config)
            elif agent_name == "macroeconomic_analysis_agent":
                self.agents[agent_name] = MacroeconomicAnalysisAgent(config)
            elif agent_name == "geopolitical_risk_agent":
                self.agents[agent_name] = GeopoliticalRiskAgent(config)
            elif agent_name == "industry_specialist_agent":
                self.agents[agent_name] = IndustrySpecialistAgent(config)
            elif agent_name == "fundamental_analyst_agent":
                self.agents[agent_name] = FundamentalAnalyst(config)
            elif agent_name == "technical_analyst_agent":
                self.agents[agent_name] = TechnicalAnalyst(config)
            elif agent_name == "risk_assessment_agent":
                self.agents[agent_name] = RiskAssessor(config)
            elif agent_name == "newsletter_layout_specialist_agent":
                self.agents[agent_name] = NewsletterLayoutSpecialist(config)
            elif agent_name == "data_verification_agent":
                self.agents[agent_name] = DataVerificationAgent(config)
            elif agent_name == "lexica_agent":
                self.agents[agent_name] = LexicaAgent(config)
            elif agent_name == "archive_manager_agent":
                self.agents[agent_name] = ArchiveManagerAgent(config)

        self.workflows = {
            "generate_newsletter": {
                "agents": [
                    "market_sentiment_agent",
                    "macroeconomic_analysis_agent",
                    "geopolitical_risk_agent",
                    "industry_specialist_agent",
                    "newsletter_layout_specialist_agent"
                ],
                "dependencies": {}  # No dependencies for this workflow
            },
            "perform_company_analysis": {
                "agents": [
                    "fundamental_analyst_agent",
                    "technical_analyst_agent",
                    "risk_assessment_agent"
                ],
                "dependencies": {
                    "risk_assessment_agent": ["fundamental_analyst_agent", "technical_analyst_agent"]
                }
            }
        }

    def execute_workflow(self, task, **kwargs):
        if task in self.workflows:
            workflow = self.workflows[task]
            completed_agents =
            for agent_name in workflow["agents"]:
                # Check for dependencies
                if agent_name in workflow["dependencies"]:
                    dependencies = workflow["dependencies"][agent_name]
                    # Ensure all dependencies have run
                    if all(dep in completed_agents for dep in dependencies):
                        # Gather outputs from dependencies
                        dependency_outputs = {dep: self.agents[dep].outputs for dep in dependencies}
                        try:
                            self.agents[agent_name].run(dependency_outputs=dependency_outputs, **kwargs)
                            completed_agents.append(agent_name)
                        except Exception as e:
                            print(f"Error running agent {agent_name}: {e}")
                    else:
                        print(f"Agent {agent_name} is waiting for dependencies: {dependencies}")
                else:
                    try:
                        self.agents[agent_name].run(**kwargs)
                        completed_agents.append(agent_name)
                    except Exception as e:
                        print(f"Error running agent {agent_name}: {e}")
        else:
            print(f"Unknown task: {task}")

# Example usage (would be called by a main script)
if __name__ == "__main__":
    import yaml
    with open("../../config/agents.yaml", "r") as f:
        agents_config = yaml.safe_load(f)

    orchestrator = AgentOrchestrator(agents_config)
    orchestrator.execute_workflow("generate_newsletter")
    orchestrator.execute_workflow("perform_company_analysis")
