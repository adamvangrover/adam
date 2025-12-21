# scripts/test_new_agents.py
import asyncio
import os
import sys

import yaml

# Add the root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.system.agent_orchestrator import AgentOrchestrator


async def main():
    """
    Main function to run the test script.
    """
    # 1. Instantiate the AgentOrchestrator
    # This will load all agents and their configurations
    try:
        orchestrator = AgentOrchestrator()
        print("Orchestrator instantiated successfully.")
    except Exception as e:
        print(f"Error instantiating orchestrator: {e}")
        return

    # 2. Create sample data for testing
    print("\n--- Testing BehavioralEconomicsAgent via Orchestrator ---")
    market_content = "There is a lot of FOMO in the market right now, everyone is buying tech stocks based on recent performance."
    user_history = ["Should I buy stock X?", "Can you find evidence for my view that stock Y will go up?"]
    
    print(f"Analyzing market content: '{market_content}'")
    print(f"Analyzing user history: {user_history}")

    # 3. Execute the BehavioralEconomicsAgent via the orchestrator
    behavioral_results = orchestrator.run_analysis(
        "behavioral_economics",
        analysis_content=market_content,
        user_query_history=user_history
    )
    print("\nBehavioralEconomicsAgent Results:")
    # The result from execute is a coroutine, so we need to await it
    if behavioral_results:
        result_data = await behavioral_results
        print(yaml.dump(result_data, indent=2))
    else:
        print("No result from behavioral economics agent.")


    # 4. Create sample data for MetaCognitiveAgent
    print("\n--- Testing MetaCognitiveAgent via Orchestrator ---")
    analysis_chain = [
        {"agent": "FundamentalAnalyst", "output": "This stock is a strong buy, despite some expert says it is risky."},
        {"agent": "RiskAssessment", "output": "The risk for this asset is very high."}
    ]
    
    print("Analyzing agent chain:")
    print(yaml.dump(analysis_chain, indent=2))

    # 5. Execute the MetaCognitiveAgent via the orchestrator
    meta_cognitive_results = orchestrator.run_analysis(
        "meta_cognitive",
        analysis_chain=analysis_chain
    )
    print("\nMetaCognitiveAgent Results:")
    if meta_cognitive_results:
        result_data = await meta_cognitive_results
        print(yaml.dump(result_data, indent=2))
    else:
        print("No result from meta cognitive agent.")


if __name__ == "__main__":
    # The main function is now async, so we run it with asyncio.run()
    asyncio.run(main())
