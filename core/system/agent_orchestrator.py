# core/system/agent_orchestrator.py

class AgentOrchestrator:
    def __init__(self, agents):
        self.agents = agents

    def run_agents(self):
        print("Running agents...")
        for agent_name, agent in self.agents.items():
            # Call each agent's relevant methods
            if agent_name == "market_sentiment_agent":
              sentiment = agent.analyze_sentiment()
              print(f"Market Sentiment: {sentiment}")
            elif agent_name == "macroeconomic_analysis_agent":
              # Add method call for macroeconomic analysis
              pass
            # Add more agent calls here
