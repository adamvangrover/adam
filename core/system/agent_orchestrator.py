# core/system/agent_orchestrator.py

from core.agents.market_sentiment_agent import MarketSentimentAgent
from core.agents.macroeconomic_analysis_agent import MacroeconomicAnalysisAgent
from core.agents.geopolitical_risk_agent import GeopoliticalRiskAgent
from core.agents.industry_specialist_agent import IndustrySpecialistAgent
from core.agents.fundamental_analyst_agent import FundamentalAnalystAgent
from core.agents.technical_analyst_agent import TechnicalAnalyst
from core.agents.risk_assessment_agent import RiskAssessor
from core.agents.newsletter_layout_specialist_agent import NewsletterLayoutSpecialist
from core.agents.data_verification_agent import DataVerificationAgent
from core.agents.lexica_agent import LexicaAgent
from core.agents.archive_manager_agent import ArchiveManagerAgent
from core.agents.agent_forge import AgentForge
from core.agents.prompt_tuner import PromptTuner
from core.agents.code_alchemist import CodeAlchemist
from core.agents.lingua_maestro import LinguaMaestro
from core.agents.sense_weaver import SenseWeaver

#... (add imports for other agents as needed)

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
                self.agents[agent_name] = FundamentalAnalystAgent(config)
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
            elif agent_name == "agent_forge":
                self.agents[agent_name] = AgentForge(config, self)  # Pass orchestrator reference
            elif agent_name == "prompt_tuner":
                self.agents[agent_name] = PromptTuner(config, self)  # Pass orchestrator reference
            elif agent_name == "code_alchemist":
                self.agents[agent_name] = CodeAlchemist(config)
            elif agent_name == "lingua_maestro":
                self.agents[agent_name] = LinguaMaestro(config)
            elif agent_name == "sense_weaver":
                self.agents[agent_name] = SenseWeaver(config)
            #... (add instantiation for other agents)

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
            },
            #... (add more workflows)
        }

    def execute_workflow(self, task, **kwargs):
        # Dynamic Workflow Selection (example)
        if task == "analyze_investment":
            if kwargs.get('investment_type') == "stock":
                workflow = self.workflows["perform_company_analysis"]
            elif kwargs.get('investment_type') == "portfolio":
                workflow = self.workflows.get("analyze_portfolio")  # Get the workflow if it exists
                if not workflow:
                    print(f"Workflow not found for portfolio analysis.")
                    return
            #... (add more conditions)
        else:
            workflow = self.workflows.get(task)
            if not workflow:
                print(f"Unknown task: {task}")
                return

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
                        # Check if the agent uses message queue communication
                        if hasattr(self.agents[agent_name], 'use_message_queue') and \
                           self.agents[agent_name].use_message_queue:
                            self.agents[agent_name].run(dependency_outputs=dependency_outputs, **kwargs)
                        else:
                            # Execute the agent's run method directly
                            self.agents[agent_name].run(**kwargs)
                        completed_agents.append(agent_name)
                    except Exception as e:
                        print(f"Error running agent {agent_name}: {e}")
                else:
                    print(f"Agent {agent_name} is waiting for dependencies: {dependencies}")
            else:
                try:
                    # Check if the agent uses message queue communication
                    if hasattr(self.agents[agent_name], 'use_message_queue') and \
                       self.agents[agent_name].use_message_queue:
                        self.agents[agent_name].run(**kwargs)
                    else:
                        # Execute the agent's run method directly
                        self.agents[agent_name].run(**kwargs)
                    completed_agents.append(agent_name)
                except Exception as e:
                    print(f"Error running agent {agent_name}: {e}")

    def add_agent(self, agent_name, agent_type, **kwargs):
        """
        Adds a new agent to the orchestrator.
        """

        #... (instantiate the new agent and add it to self.agents)
        pass  # Placeholder for actual implementation

    def update_agent_prompt(self, agent_name, **kwargs):
        """
        Updates the prompt of an existing agent.
        """

        #... (fetch the agent and update its prompt)
        pass  # Placeholder for actual implementation

# Example usage (would be called by a main script)
if __name__ == "__main__":
    import yaml
    with open("../../config/agents.yaml", "r") as f:
        agents_config = yaml.safe_load(f)

    orchestrator = AgentOrchestrator(agents_config)
    orchestrator.execute_workflow("generate_newsletter")
    orchestrator.execute_workflow("perform_company_analysis")
    #... (add execution of other workflows)
