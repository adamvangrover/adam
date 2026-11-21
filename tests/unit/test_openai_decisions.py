"""
Unit tests for OpenAI Decisions API Engine and CLI runner.
"""

import json
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

from core.llm.engines.openai_decisions_engine import (
    ChoiceAnswer,
    ChoiceOption,
    ChoiceQuestion,
    DecisionsRequest,
    DecisionsResponse,
    InputImageContent,
    InputMessage,
    InputTextContent,
    OpenAIDecisionsEngine,
    PredicateAnswer,
    PredicateQuestion,
    ScoreAnswer,
    ScoreLevel,
    ScoreQuestion,
    calculate_expected_score,
)


def test_predicate_question_formatting_and_parsing() -> None:
    q = PredicateQuestion(
        name="visible_damage",
        instructions="Does the product have visible damage, such as a crack, tear, or dent?",
    )
    req = DecisionsRequest(
        model="gpt-6-luna",
        input="Inspect product image",
        questions=[q],
    )
    dumped = req.model_dump(exclude_none=True)
    assert dumped["model"] == "gpt-6-luna"
    assert dumped["questions"][0]["type"] == "predicate"
    assert dumped["questions"][0]["name"] == "visible_damage"

    resp_json = {
        "answers": [
            {
                "type": "predicate",
                "name": "visible_damage",
                "probability": 0.92,
            }
        ]
    }
    resp = DecisionsResponse.model_validate(resp_json)
    assert len(resp.answers) == 1
    assert isinstance(resp.answers[0], PredicateAnswer)
    assert resp.answers[0].name == "visible_damage"
    assert pytest.approx(resp.answers[0].probability) == 0.92


def test_choice_question_formatting_and_parsing() -> None:
    q = ChoiceQuestion(
        name="department",
        instructions="Which department should handle this complaint?",
        choices=[
            ChoiceOption(value="billing", description="Payments, invoices, and refunds."),
            ChoiceOption(value="technical", description="Problems using the product."),
            ChoiceOption(value="shipping", description="Delivery and tracking."),
            ChoiceOption(value="other", description="Requests outside these categories."),
        ],
    )
    req = DecisionsRequest(
        model="gpt-6-luna",
        input="I was charged twice for my order.",
        questions=[q],
    )
    dumped = req.model_dump(exclude_none=True)
    assert len(dumped["questions"][0]["choices"]) == 4

    resp_json = {
        "answers": [
            {
                "type": "choice",
                "name": "department",
                "choice": "billing",
                "probabilities": [
                    {"value": "billing", "probability": 0.95},
                    {"value": "technical", "probability": 0.02},
                    {"value": "shipping", "probability": 0.01},
                    {"value": "other", "probability": 0.02},
                ],
                "confidence": 0.93,
            }
        ]
    }
    resp = DecisionsResponse.model_validate(resp_json)
    assert isinstance(resp.answers[0], ChoiceAnswer)
    assert resp.answers[0].choice == "billing"
    assert pytest.approx(resp.answers[0].confidence) == 0.93


def test_score_question_formatting_and_parsing() -> None:
    q = ScoreQuestion(
        name="severity",
        instructions="How severe is this issue?",
        levels=[
            ScoreLevel(label="Cosmetic", description="Appearance only; no lost functionality."),
            ScoreLevel(label="Workaround available", description="A task fails, but another way works."),
            ScoreLevel(label="Fully blocked", description="A task fails with no workaround."),
        ],
    )
    req = DecisionsRequest(
        model="gpt-6-luna",
        input="Export fails in Safari but works in Chrome.",
        questions=[q],
    )
    dumped = req.model_dump(exclude_none=True)
    assert len(dumped["questions"][0]["levels"]) == 3

    resp_json = {
        "answers": [
            {
                "type": "score",
                "name": "severity",
                "score": 1.1,
                "probabilities": [
                    {"value": 0, "label": "Cosmetic", "probability": 0.1},
                    {"value": 1, "label": "Workaround available", "probability": 0.7},
                    {"value": 2, "label": "Fully blocked", "probability": 0.2},
                ],
                "confidence": 0.55,
            }
        ]
    }
    resp = DecisionsResponse.model_validate(resp_json)
    score_ans = resp.answers[0]
    assert isinstance(score_ans, ScoreAnswer)
    assert pytest.approx(score_ans.score) == 1.1

    # Verify probability-weighted score arithmetic helper: 0*0.1 + 1*0.7 + 2*0.2 = 1.1
    calculated = calculate_expected_score(score_ans.probabilities)
    assert pytest.approx(calculated) == 1.1


def test_multimodal_input_handling() -> None:
    msg = InputMessage(
        role="user",
        content=[
            InputTextContent(text="Inspect the product in this photo."),
            InputImageContent(
                image_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            ),
        ],
    )
    q = PredicateQuestion(name="has_image_defect", instructions="Is there any defect?")
    req = DecisionsRequest(
        model="gpt-6-luna",
        input=[msg],
        questions=[q],
    )
    dumped = req.model_dump(exclude_none=True)
    assert isinstance(dumped["input"], list)
    assert dumped["input"][0]["content"][0]["type"] == "input_text"
    assert dumped["input"][0]["content"][1]["type"] == "input_image"


def test_decisions_engine_simulation() -> None:
    engine = OpenAIDecisionsEngine(simulation_mode=True)
    q_pred = PredicateQuestion(name="damage", instructions="Is damaged?")
    q_choice = ChoiceQuestion(
        name="dept",
        instructions="Select dept",
        choices=[ChoiceOption(value="b"), ChoiceOption(value="t")],
    )
    q_score = ScoreQuestion(
        name="sev",
        instructions="Score severity",
        levels=[ScoreLevel(label="Low"), ScoreLevel(label="Medium"), ScoreLevel(label="High")],
    )
    req = DecisionsRequest(
        model="gpt-6-luna",
        input="Test sample input",
        questions=[q_pred, q_choice, q_score],
    )

    resp = engine.evaluate_decisions(req)
    assert len(resp.answers) == 3
    assert resp.answers[0].name == "damage"
    assert resp.answers[1].name == "dept"
    assert resp.answers[2].name == "sev"


def test_decisions_engine_mocked_api_call() -> None:
    mock_api_response = {
        "answers": [
            {
                "type": "predicate",
                "name": "is_valid",
                "probability": 0.99,
            }
        ]
    }

    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(mock_api_response).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    engine = OpenAIDecisionsEngine(api_key="mock-key-123", simulation_mode=False)
    req = DecisionsRequest(
        model="gpt-6-luna",
        input="Sample text",
        questions=[PredicateQuestion(name="is_valid", instructions="Is valid?")],
    )

    with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
        res = engine.evaluate_decisions(req)
        assert mock_urlopen.called
        assert res.answers[0].name == "is_valid"
        assert pytest.approx(res.answers[0].probability) == 0.99


def test_cli_script_execution() -> None:
    cmd = [
        sys.executable,
        "scripts/openai_decisions.py",
        "--input",
        "Export fails in Safari",
        "--questions-json",
        json.dumps(
            [
                {
                    "type": "predicate",
                    "name": "has_bug",
                    "instructions": "Is this a bug?",
                }
            ]
        ),
        "--simulate",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = json.loads(res.stdout)
    assert "answers" in output
    assert output["answers"][0]["name"] == "has_bug"
