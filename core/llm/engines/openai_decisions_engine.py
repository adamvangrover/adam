"""
OpenAI Decisions API Client Engine.

Provides typed interfaces, request/response models, score calculations,
and API execution for OpenAI's POST /v1/decisions endpoint using `gpt-6-luna`.
"""

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ChoiceOption(BaseModel):
    value: str = Field(..., description="Unique value for this choice option.")
    description: str | None = Field(None, description="Description explaining when this choice applies.")


class ScoreLevel(BaseModel):
    label: str = Field(..., description="Label for this score level.")
    description: str | None = Field(None, description="Description explaining criteria for this level.")


class PredicateQuestion(BaseModel):
    type: Literal["predicate"] = "predicate"
    name: str = Field(..., description="Unique question identifier.")
    instructions: str = Field(..., description="Instructions for evaluating this predicate.")


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    name: str = Field(..., description="Unique question identifier.")
    instructions: str = Field(..., description="Instructions for evaluating this choice.")
    choices: list[ChoiceOption] = Field(..., description="Available choice options.")


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    name: str = Field(..., description="Unique question identifier.")
    instructions: str = Field(..., description="Instructions for rating against ordered levels.")
    levels: list[ScoreLevel] = Field(..., description="Ordered criteria levels from lowest (0) to highest.")


QuestionType = PredicateQuestion | ChoiceQuestion | ScoreQuestion


class InputTextContent(BaseModel):
    type: Literal["input_text"] = "input_text"
    text: str


class InputImageContent(BaseModel):
    type: Literal["input_image"] = "input_image"
    image_url: str = Field(..., description="Base64 data URL, e.g., data:image/png;base64,...")


InputContentPart = InputTextContent | InputImageContent


class InputMessage(BaseModel):
    role: str = "user"
    content: list[InputContentPart]


class DecisionsRequest(BaseModel):
    model: str = Field("gpt-6-luna", description="Model name; defaults to gpt-6-luna.")
    input: str | list[InputMessage] = Field(..., description="Shared string or message input.")
    questions: list[QuestionType] = Field(..., description="Array of questions to evaluate.")


class ChoiceProbability(BaseModel):
    value: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ScoreProbability(BaseModel):
    value: int = Field(..., ge=0)
    label: str
    probability: float = Field(..., ge=0.0, le=1.0)


class PredicateAnswer(BaseModel):
    type: Literal["predicate"] = "predicate"
    name: str
    probability: float = Field(..., ge=0.0, le=1.0)


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    name: str
    choice: str
    probabilities: list[ChoiceProbability]
    confidence: float = Field(..., ge=0.0, le=1.0)


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    name: str
    score: float = Field(..., description="Probability-weighted average of level indices.")
    probabilities: list[ScoreProbability]
    confidence: float = Field(..., ge=0.0, le=1.0)


AnswerType = PredicateAnswer | ChoiceAnswer | ScoreAnswer


class DecisionsResponse(BaseModel):
    answers: list[AnswerType]


def calculate_expected_score(probabilities: list[ScoreProbability]) -> float:
    """Calculates the probability-weighted average score across level indices."""
    return round(sum(item.value * item.probability for item in probabilities), 4)


class OpenAIDecisionsEngine:
    """
    Engine to interface with OpenAI's Decisions API (POST /v1/decisions).
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-6-luna",
        simulation_mode: bool = False,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.simulation_mode = simulation_mode or not self.api_key

    def evaluate_decisions(self, request: DecisionsRequest) -> DecisionsResponse:
        """
        Sends a request to the Decisions API or simulates the response if in simulation mode.
        """
        if self.simulation_mode or not self.api_key:
            logger.info("OpenAIDecisionsEngine: Executing in simulation mode.")
            return self._simulate_response(request)

        endpoint = f"{self.base_url}/decisions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        data = request.model_dump(exclude_none=True)
        req_body = json.dumps(data).encode("utf-8")

        req = urllib.request.Request(endpoint, data=req_body, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req) as resp:  # nosec B310
                body = resp.read().decode("utf-8")
                res_json = json.loads(body)
                return DecisionsResponse.model_validate(res_json)
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            logger.error(f"Decisions API HTTP Error {e.code}: {error_body}")
            raise RuntimeError(f"Decisions API returned HTTP {e.code}: {error_body}") from e
        except Exception as e:
            logger.error(f"Failed to communicate with Decisions API: {e}")
            raise RuntimeError(f"Failed to call Decisions API: {e}") from e

    def _simulate_response(self, request: DecisionsRequest) -> DecisionsResponse:
        """Generates deterministic mock answers for evaluation/testing."""
        answers: list[AnswerType] = []

        for question in request.questions:
            is_typed = isinstance(question, PredicateQuestion | ChoiceQuestion | ScoreQuestion)
            q_type = question.type if is_typed else question.get("type")
            if q_type == "predicate":
                q_name = question.name if isinstance(question, PredicateQuestion) else question["name"]
                answers.append(
                    PredicateAnswer(
                        name=q_name,
                        probability=0.85,
                    )
                )
            elif q_type == "choice":
                q_name = question.name if isinstance(question, ChoiceQuestion) else question["name"]
                choices = question.choices if isinstance(question, ChoiceQuestion) else question["choices"]
                first_choice = choices[0].value if choices else "default"
                probs = []
                for idx, ch in enumerate(choices):
                    ch_val = ch.value if isinstance(ch, ChoiceOption) else ch["value"]
                    p = 0.85 if idx == 0 else round(0.15 / max(1, len(choices) - 1), 4)
                    probs.append(ChoiceProbability(value=ch_val, probability=p))

                answers.append(
                    ChoiceAnswer(
                        name=q_name,
                        choice=first_choice,
                        probabilities=probs,
                        confidence=0.85,
                    )
                )
            elif q_type == "score":
                q_name = question.name if isinstance(question, ScoreQuestion) else question["name"]
                levels = question.levels if isinstance(question, ScoreQuestion) else question["levels"]
                probs = []
                num_levels = len(levels)
                if num_levels == 3:
                    probs_vals = [0.1, 0.7, 0.2]
                else:
                    probs_vals = [1.0 / num_levels] * num_levels

                for idx, lvl in enumerate(levels):
                    lbl = lvl.label if isinstance(lvl, ScoreLevel) else lvl["label"]
                    probs.append(ScoreProbability(value=idx, label=lbl, probability=probs_vals[idx]))

                weighted_score = calculate_expected_score(probs)
                answers.append(
                    ScoreAnswer(
                        name=q_name,
                        score=weighted_score,
                        probabilities=probs,
                        confidence=0.75,
                    )
                )

        return DecisionsResponse(answers=answers)
