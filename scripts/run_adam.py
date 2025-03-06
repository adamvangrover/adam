import sys
import os
from core.system.agent_orchestrator import AgentOrchestrator
from core.system.knowledge_base import KnowledgeBase
from core.system.data_manager import DataManager
from core.system.echo import Echo
from core.utils.config_utils import load_config
from core.utils.logging_utils import setup_logging
from core.utils.api_utils import get_knowledge_graph_data, update_knowledge_graph_node
from core.llm_plugin import LLMPlugin  # Import LLM plugin

def main():
    """
    Main execution script for Adam v18.0.
    """
    try:
        # Load configuration
        config = load_config()

        # Set up logging
        setup_logging(config)

        # Initialize knowledge base
        knowledge_base = KnowledgeBase(config)

        # Initialize data manager
        data_manager = DataManager(config)

        # Initialize agent orchestrator
        agent_orchestrator = AgentOrchestrator(config, knowledge_base, data_manager)

        # Initialize Echo System
        echo_system = Echo(config.get("echo_system", {}))

        # Initialize LLM plugin
        llm_plugin = LLMPlugin(config.get("llm_plugin", {}))

        # Start the agent orchestrator
        agent_orchestrator.run()

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

        # 4. Use the LLM plugin for content generation
        generated_content = llm_plugin.generate_content("financial trends", "summarize the latest market news")
        print("Generated Content:", generated_content)

        # 5. Execute a workflow
        agent_orchestrator.execute_workflow("generate_newsletter")

        # ... (Add more examples and use cases)

    except Exception as e:
        print(f"Error running Adam v18.0: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
