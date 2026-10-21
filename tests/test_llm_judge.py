"""
Unit tests for Clean-Room LLM-as-a-Judge Evaluator (evals/graders/llm_judge.py).
"""

from evals.graders.llm_judge import CleanRoomLLMJudge, DimensionScore, JudgeResult, grade_answer


def test_grade_answer_backward_compatibility():
    score = grade_answer(
        question="What is the revenue?",
        agent_answer="The revenue is 10.5B.",
        golden_answer="10.5B",
    )
    assert score == 1.0 or score >= 0.70


def test_clean_room_judge_heuristics_exact_match():
    judge = CleanRoomLLMJudge(pass_threshold=0.70)
    result = judge.evaluate(
        question="What is the net debt?",
        agent_answer="According to section 4, net debt is 4.50B because EBITDA was strong.",
        golden_answer="4.50B",
    )
    assert isinstance(result, JudgeResult)
    assert result.passed is True
    assert result.overall_score >= 0.70
    assert len(result.dimensions) == 4


def test_clean_room_judge_heuristics_numerical_tolerance():
    judge = CleanRoomLLMJudge(pass_threshold=0.60)
    result = judge.evaluate(
        question="What is the target price?",
        agent_answer="Target price is 100.5",
        golden_answer="100.0",
    )
    assert result.overall_score > 0.0
    correctness_dim = next(d for d in result.dimensions if d.name == "correctness")
    assert correctness_dim.score == 1.0 or correctness_dim.score == 0.75


def test_clean_room_judge_offline_replay():
    replay_data = {
        "q1_case": {
            "overall_score": 0.92,
            "verdict": "PASS",
            "feedback": "Offline benchmark verified.",
            "dimensions": [
                {"name": "correctness", "score": 1.0, "weight": 0.4},
                {"name": "faithfulness", "score": 0.9, "weight": 0.3},
            ],
        }
    }
    judge = CleanRoomLLMJudge(replay_store=replay_data, pass_threshold=0.80)
    result = judge.evaluate(
        question="Dummy Q",
        agent_answer="Dummy A",
        golden_answer="Dummy G",
        replay_key="q1_case",
    )
    assert result.overall_score == 0.92
    assert result.passed is True
    assert result.feedback == "Offline benchmark verified."
    assert len(result.dimensions) == 2


def test_clean_room_judge_llm_callable():
    def mock_llm(prompt: str) -> str:
        return "Dimension evaluation completed. OVERALL_SCORE: 0.88"

    judge = CleanRoomLLMJudge(llm_callable=mock_llm, pass_threshold=0.80)
    result = judge.evaluate(
        question="What was Q3 EPS?",
        agent_answer="Q3 EPS was $1.25",
        golden_answer="$1.25",
    )
    assert result.overall_score == 0.88
    assert result.passed is True
    assert result.verdict == "PASS"
