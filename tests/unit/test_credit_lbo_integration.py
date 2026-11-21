import pytest
import asyncio
from unittest.mock import MagicMock, patch
from core.agents.credit.quant import QuantAgent
from core.agents.credit.writer import WriterAgent

@pytest.mark.asyncio
async def test_quant_agent_lbo_extraction():
    agent = QuantAgent({})
    inputs = {
        "documents": [], # Mocked
        "financial_data": {} # Not used directly in this mock implementation but usually present
    }

    # We need to pass data that QuantAgent expects or mock its internal logic if it relies on parsing.
    # The current implementation of QuantAgent in the diff relies on 'financial_data' dict for mocked extraction.
    # Wait, the code I modified in QuantAgent uses `financial_data.get(...)` on the dictionary BUILT from documents.
    # So I need to mock the document chunk content.

    inputs = {
        "documents": [
            {
                "chunks": [
                    {
                        "type": "financial_table",
                        "content_json": {
                            "ebitda": 100.0,
                            "total_debt": 500.0,
                            "interest_expense": 50.0,
                            "capex": 10.0,
                            "senior_secured_debt": 300.0
                        }
                    }
                ]
            }
        ]
    }

    result = await agent.execute(inputs)

    assert "ratios" in result
    assert result["ratios"]["leverage"] == 5.0 # 500/100
    assert result["ratios"]["fccr"] == (100 - 10) / 50.0 # 1.8

    assert "debt_structure" in result
    assert result["debt_structure"]["senior_secured"] == 300.0

@pytest.mark.asyncio
async def test_writer_agent_prompt_integration():
    # We need to mock PromptLoader where it is used in writer.py
    with patch("core.agents.credit.writer.PromptLoader") as MockLoaderCls:
        mock_loader = MockLoaderCls.return_value
        mock_config = MagicMock()
        mock_config.system_template = "System"
        mock_config.user_template = "User {{lbo_analysis}}" # Verify placeholder
        mock_config.input_variables = [{"name": "lbo_analysis"}]
        mock_config.version = "1.0.0" # Needed for metadata
        mock_loader.load_prompt.return_value = mock_config
        mock_loader.render_messages.return_value = []

        agent = WriterAgent({})

        inputs = {
            "borrower_name": "TestCo",
            "lbo_analysis": "IRR: 25%",
            "distressed_scenarios": "Liquidation: 50c"
        }

        # We need to mock _mock_llm_response because it's hardcoded and creates the output
        # But wait, I updated _mock_llm_response to use the context!
        # So I can test the output of execute directly.

        result = await agent.execute(inputs)

        output = result["output"]
        assert "Leveraged Finance / LBO" in output
        assert "IRR: 25%" in output
        assert "Distressed Scenarios" in output
        assert "Liquidation: 50c" in output
