import pytest
from pydantic import BaseModel
from adam_v3.kernel.kernel import Kernel

class MockLLMOutput(BaseModel):
    filing_data: dict
    risk_metrics: dict
    evaluation: dict

def test_kernel_jsonlogic_evaluation():
    kernel = Kernel(logic_rules_path="kernel/logic_rules.json")

    # Test Rule 1: Covenant failure trigger
    output = MockLLMOutput(
        filing_data={"form_type": "10-K", "covenants_detected": False},
        risk_metrics={"base_pd": 0.05},
        evaluation={"agent_confidence": 0.9, "evidence_quality_score": 0.9}
    )
    result = kernel.evaluate_output(output)
    assert result.get("action") == "FAIL_AND_TERMINATE"

    # Test Rule 2: PD boundary constraint
    output = MockLLMOutput(
        filing_data={"form_type": "10-K", "covenants_detected": True},
        risk_metrics={"base_pd": 0.20},
        evaluation={"agent_confidence": 0.9, "evidence_quality_score": 0.9}
    )
    result = kernel.evaluate_output(output)
    assert result.get("action") == "ROUTE_TO_QUANT_SOLVER"
