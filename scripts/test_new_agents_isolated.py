# scripts/test_new_agents_isolated.py
import asyncio
import os
import sys

import yaml

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.agents.behavioral_economics_agent import BehavioralEconomicsAgent
from core.agents.meta_cognitive_agent import MetaCognitiveAgent
from core.utils.config_utils import load_config


async def main():
    """
    Main function to run the test script.
    """
    # 1. Load agent configurations
    config_path = 'config/agents.yaml'
    agent_configs = load_config(config_path)

    if not agent_configs or 'agents' not in agent_configs:
        print("Error: Could not load agent configurations or 'agents' key not found.")
        return

    # 2. Instantiate the agents
    try:
        behavioral_agent_config = agent_configs['agents']['behavioral_economics_agent']
        meta_cognitive_agent_config = agent_configs['agents']['meta_cognitive_agent']

        behavioral_agent = BehavioralEconomicsAgent(behavioral_agent_config)
        meta_cognitive_agent = MetaCognitiveAgent(meta_cognitive_agent_config)
        
        print("Agents instantiated successfully.")
    except KeyError as e:
        print(f"Error: Agent configuration not found for {e}. Make sure they are defined in {config_path}")
        return

    # 3. Create sample data for testing
    print("\n--- Testing BehavioralEconomicsAgent ---")
    market_content = "There is a lot of FOMO in the market right now, everyone is buying tech stocks based on recent performance."
    user_history = ["Should I buy stock X?", "Can you find evidence for my view that stock Y will go up?"]
    
    print(f"Analyzing market content: '{market_content}'")
    print(f"Analyzing user history: {user_history}")

    # 4. Execute the BehavioralEconomicsAgent
    behavioral_results = await behavioral_agent.execute(market_content, user_history)
    print("\nBehavioralEconomicsAgent Results:")
    print(yaml.dump(behavioral_results, indent=2))

    # 5. Create sample data for MetaCognitiveAgent
    print("\n--- Testing MetaCognitiveAgent ---")
    analysis_chain = [
        {"agent": "FundamentalAnalyst", "output": "This stock is a strong buy, despite some expert says it is risky."},
        {"agent": "RiskAssessment", "output": "The risk for this asset is very high."}
    ]
    
    print("Analyzing agent chain:")
    print(yaml.dump(analysis_chain, indent=2))

    # 6. Execute the MetaCognitiveAgent
    meta_cognitive_results = await meta_cognitive_agent.execute(analysis_chain)
    print("\nMetaCognitiveAgent Results:")
    print(yaml.dump(meta_cognitive_results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
