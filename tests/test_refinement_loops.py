import pytest
from core.evaluation.system_judge import SystemJudge, EvaluationRubric
from core.evaluation.iterative_llm_judge import IterativeLLMJudge
from core.learning.prompt_refinery import PromptRefinery
from core.monitoring.drift_monitor import ModelDriftMonitor
from core.evaluation.rubric_logger import EvaluationMarkdownLogger
import os

@pytest.fixture
def sample_rubrics():
    return [
        EvaluationRubric(criteria="Accuracy", max_score=10, weight=0.5, description="Factual correctness."),
        EvaluationRubric(criteria="Style", max_score=5, weight=0.5, description="Tone and formatting.")
    ]

@pytest.fixture
def mock_llm_plugin():
    class MockPlugin:
        def generate_structured(self, prompt, schema):
            # simulate fallback logic in iterative judge
            raise Exception("Mock error to trigger fallback")

        async def generate_text_async(self, prompt):
            return "This is a mock generated output."

        def generate_text(self, prompt):
            return "Refined prompt text."

    return MockPlugin()

@pytest.mark.asyncio
async def test_prompt_refinery_loop(mock_llm_plugin, sample_rubrics):
    system_judge = SystemJudge()
    llm_judge = IterativeLLMJudge(rubrics=sample_rubrics, llm_plugin=mock_llm_plugin)
    refinery = PromptRefinery(llm_plugin=mock_llm_plugin, judge=llm_judge, system_judge=system_judge)

    result = await refinery.refine_prompt_loop(
        initial_prompt="Tell me about financial risk.",
        max_iterations=2,
        target_score=95.0
    )

    assert "final_prompt" in result
    assert result["iterations"] > 0
    assert len(result["history"]) > 0
    assert "system_metrics" in result["history"][0]
    assert "llm_metrics" in result["history"][0]

def test_drift_monitor():
    monitor = ModelDriftMonitor(window_size=20)

    # Add older stable history
    for _ in range(10):
        monitor.log_execution({"token_efficiency": 10.0, "latency_ms": 100}, llm_score=90.0)

    # Add recent degraded history
    for _ in range(10):
        monitor.log_execution({"token_efficiency": 5.0, "latency_ms": 200}, llm_score=70.0)

    drift_report = monitor.check_drift()
    assert drift_report["status"] == "drift_detected"
    assert drift_report["score_degradation_pct"] > 0.0

def test_markdown_logger(tmp_path):
    logger = EvaluationMarkdownLogger(log_dir=str(tmp_path))
    session_data = {
        "final_prompt": "Best prompt ever.",
        "iterations": 1,
        "history": [{
            "iteration": 1,
            "system_metrics": {"latency_ms": 150.0, "token_efficiency": 5.5, "format_valid": True},
            "llm_metrics": {"overall_score": 85.0, "critique": "Good start.", "improvement_suggestions": ["More context"]}
        }]
    }

    filepath = logger.log_refinement_session(session_data, drift_report={"status": "stable"})
    assert os.path.exists(filepath)
    with open(filepath, "r") as f:
        content = f.read()
        assert "Best prompt ever" in content
        assert "Good start" in content
        assert "```json" in content
