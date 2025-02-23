# scripts/run_adam.py

import yaml
from core.system.agent_orchestrator import AgentOrchestrator
from core.system.echo import Echo
from core.utils.api_utils import (
    get_knowledge_graph_data,
    update_knowledge_graph_node,
)

def main():
    """
    Main execution script for Adam v17.0.
    """
    # Load configuration
    with open("../config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Initialize Agent Orchestrator and Echo System
    agent_orchestrator = AgentOrchestrator(config.get("agents", {}))
    echo_system = Echo(config.get("echo_system", {}))

    # Example usage:
    # 1. Run a specific analysis module
    market_sentiment = agent_orchestrator.run_analysis("market_sentiment")
    print("Market Sentiment Analysis:", market_sentiment)

    # 2. Access and update the knowledge graph
    dcf_data = get_knowledge_graph_data("Valuation", "DCF")
    print("DCF Data:", dcf_data)

    update_status = update_knowledge_graph_node("Valuation", "DCF", "discount_rate", 0.12)
    print("Update Status:", update_status)

    # 3. Get insights from the Echo system
    insights = echo_system.get_insights(query="What are the latest market trends?")
    print("Echo Insights:", insights)

    # 4. Execute a workflow
    agent_orchestrator.execute_workflow("generate_newsletter")

    #... (Add more examples and use cases)

if __name__ == "__main__":
    main()
