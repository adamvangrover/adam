from core.agents.agent_base import AgentBase

class CreditRiskOrchestrator(AgentBase):
    def __init__(self, config):
        super().__init__(config)
        self.sub_agents = self.config.get("sub_agents", {})
        self.meta_agents = self.config.get("meta_agents", {})

    def execute(self, query):
        # 1. Decompose the query into a plan
        plan = self._create_plan(query)

        # 2. Execute the plan
        results = {}
        for step in plan:
            agent_name = step.get("agent")
            agent_input = results.get(step.get("input"))
            if agent_name in self.sub_agents:
                agent = self.sub_agents[agent_name]
                results[agent_name] = agent.execute(agent_input)
            elif agent_name in self.meta_agents:
                agent = self.meta_agents[agent_name]
                results[agent_name] = agent.execute(agent_input)

        # 3. Synthesize the results
        return self._synthesize_results(results)

    def _create_plan(self, query):
        # In a real implementation, this would use an LLM to generate the plan
        return [
            {
                "agent": "financial_news_sub_agent",
                "input": query,
            },
            {
                "agent": "sentiment_analysis_meta_agent",
                "input": "financial_news_sub_agent",
            },
        ]

    def _synthesize_results(self, results):
        # In a real implementation, this would use an LLM to synthesize the results
        return {
            "source_agent": self.__class__.__name__,
            "confidence_score": 0.9,
            "data": results,
        }
