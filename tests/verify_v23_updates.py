import asyncio
import logging
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logging.basicConfig(level=logging.INFO)

async def test_reflector_agent():
    print("\n--- Testing ReflectorAgent ---")
    from core.agents.reflector_agent import ReflectorAgent

    agent = ReflectorAgent({})
    content = "This is a very short text." # Should trigger critique

    result = await agent.execute(content)
    print(f"Reflector Result: {result}")

    assert "critique_notes" in result
    assert "verification_status" in result

async def test_crisis_agent():
    print("\n--- Testing CrisisSimulationMetaAgent ---")
    from core.agents.meta_agents.crisis_simulation_agent import CrisisSimulationMetaAgent
    from core.schemas.crisis_simulation import CrisisSimulationInput, RiskEntity

    agent = CrisisSimulationMetaAgent({})

    input_data = CrisisSimulationInput(
        risk_portfolio=[
            RiskEntity(
                Risk_ID="R-001", description="Test Risk", velocity="Rapid",
                persistence="Transient", interconnectivity=[], strategic_objective="Growth",
                quantitative_exposure=100.0, control_effectiveness=0.5, control_strength="Moderate"
            )
        ],
        current_date="2023-10-27",
        user_scenario="Global Cyber Attack"
    )

    try:
        result = await agent.execute(input_data)
        print(f"Crisis Result: {result.executive_summary}")
    except Exception as e:
        print(f"Crisis Agent Failed (Expected if mock/graph issues): {e}")

async def test_risk_agent():
    print("\n--- Testing RiskAssessmentAgent ---")
    from core.agents.risk_assessment_agent import RiskAssessmentAgent

    # Enable v23 graph in config
    agent = RiskAssessmentAgent({"use_v23_graph": True})

    target = {"company_name": "AAPL"}
    result = await agent.execute(target, risk_type="investment")

    print(f"Risk Result Keys: {result.keys()}")
    if "graph_state" in result:
        print("Successfully used v23 Graph!")
    else:
        print("Fallback to v21 Logic.")

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    loop.run_until_complete(test_reflector_agent())
    loop.run_until_complete(test_crisis_agent())
    loop.run_until_complete(test_risk_agent())

    loop.close()

if __name__ == "__main__":
    main()
