import pytest
import json
import os
import sys
# Ensure core can be imported
sys.path.append(os.getcwd())

from core.agents.black_swan_agent import BlackSwanAgent

GOLDEN_SET_PATH = os.path.join(os.path.dirname(__file__), 'data/golden_set.json')

def load_golden_set():
    with open(GOLDEN_SET_PATH, 'r') as f:
        return json.load(f)

@pytest.mark.asyncio
async def test_prompt_regression():
    golden_set = load_golden_set()
    agent = BlackSwanAgent(config={"name": "TestBlackSwan", "persona": "Test"})

    for case in golden_set:
        print(f"Running Case: {case['scenario_id']}")
        input_data = case['input']

        # Execute Agent
        result = await agent.execute(**input_data)

        # Validate against Golden Set
        for check in case['expected_checks']:
            field = check['field']
            operator = check['operator']
            value = check['value']

            actual_value = result.get(field)

            if operator == "lt":
                assert actual_value < value, f"Failed {field}: {actual_value} is not < {value}"
            elif operator == "contains_scenario":
                scenarios = [r['scenario_name'] for r in result.get('sensitivity_analysis', [])]
                assert value in scenarios, f"Failed: {value} not found in scenarios {scenarios}"

        # "Judge Agent" Logic (Mocked)
        # Verify structure adheres to schema (Pydantic does this internally but we check output)
        assert "sensitivity_table_markdown" in result
        assert "| Scenario |" in result["sensitivity_table_markdown"]
