import asyncio
import logging
import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.agents.meta_cognitive_agent import MetaCognitiveAgent
from core.agents.agent_base import AgentInput

logging.basicConfig(level=logging.INFO, format='%(message)s')

async def run_meta_test():
    """
    Test harness to verify the MetaCognitiveAgent's logical fallacy 
    and contradiction detection logic.
    """
    agent = MetaCognitiveAgent()
    
    print("\n" + "="*80)
    print("TEST 1: LOGICAL FALLACY DETECTION (HASTY GENERALIZATION)")
    print("="*80)
    
    # Intentionally trigger the "hasty_generalization" regular expression
    bad_input_1 = AgentInput(
        query="The market is crashing because everyone is selling. No one is buying tech stocks.",
        context={"agent_name": "PanicAlgo"}
    )
    
    result1 = await agent.execute(bad_input_1)
    print(f"\n[Final Decision]: {result1.answer}")
    print(f"[Detected Fallacies]: {json.dumps(result1.metadata['detected_fallacies'], indent=2)}")
    
    
    print("\n" + "="*80)
    print("TEST 2: CONTRADICTION DETECTION (EXTREME POLARITY WITHOUT NUANCE)")
    print("="*80)
    
    # Intentionally trigger both positive and negative keywords without nuance words
    bad_input_2 = AgentInput(
        query="This is a strong buy opportunity. It is a high risk and bearish asset.",
        context={"agent_name": "ConfusedAnalyst"}
    )
    
    result2 = await agent.execute(bad_input_2)
    print(f"\n[Final Decision]: {result2.answer}")
    print(f"[Detected Contradictions]: {json.dumps(result2.metadata['contradictions_detected'], indent=2)}")


    print("\n" + "="*80)
    print("TEST 3: NUANCED HEALTHY OUTPUT")
    print("="*80)
    
    good_input = AgentInput(
        query="This is a strong buy opportunity. However, it remains a high risk asset in the short term.",
        context={"agent_name": "NuancedAnalyst"}
    )
    
    result3 = await agent.execute(good_input)
    print(f"\n[Final Decision]: {result3.answer}")
    print(f"[Detected Contradictions]: {result3.metadata['contradictions_detected']}")
    
    print("\nTEST HARNESS COMPLETE.")


if __name__ == "__main__":
    asyncio.run(run_meta_test())
