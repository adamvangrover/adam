import asyncio
import logging
import json
import os
import sys

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine.system2_state import System2State
from core.engine.system2_graph import system2_app

# Configure logging to see the Graph's execution steps easily
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

async def run_test_harness():
    """
    Test harness for the System 2 LangGraph Reflexion Loop.
    """
    print("\n" + "="*60)
    print("INITIALIZING SYSTEM 2: NEURO-SYMBOLIC GRAPH TEST HARNESS")
    print("="*60 + "\n")
    
    # Initialize the starting state
    initial_state: System2State = {
        "company_ticker": "AAPL",
        "historical_data": {"revenue": [365e9, 394e9, 383e9]},
        "iteration_count": 0,
        "max_iterations": 3,
        "generated_dcf": None,
        "validation_feedback": [],
        "is_valid": False,
        "final_report": ""
    }
    
    print("Starting State Injection into LangGraph...")
    
    # Invoke the Graph
    final_state = await system2_app.ainvoke(initial_state)
    
    print("\n" + "="*60)
    print("GRAPH EXECUTION COMPLETE")
    print("="*60 + "\n")
    
    # Evaluate Results
    print(f"Final Iteration Count: {final_state['iteration_count']}")
    print(f"Is Financial Model Valid?: {final_state['is_valid']}")
    
    if final_state['validation_feedback']:
        print("\nFeedback Trace History:")
        for fb in final_state['validation_feedback']:
            print(f"  -> {fb}")
            
    print("\nFinal Generated DCF Output Snippet:")
    dcf = final_state.get('generated_dcf', {})
    print(json.dumps(
        {k: v for k, v in dcf.items() if k in ['company_ticker', 'wacc', 'terminal_growth_rate', 'enterprise_value']}, 
        indent=4
    ))
    
    print("\n--- Final System Report ---")
    print(final_state.get('final_report', "NO REPORT GENERATED"))
    print("\nTEST HARNESS FINISHED.")

if __name__ == "__main__":
    asyncio.run(run_test_harness())
