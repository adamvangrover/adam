import json
import argparse
import os
from datetime import datetime, timezone
import hashlib
from typing import Dict, Any

from litellm import completion
from pydantic import BaseModel, Field

from src.schemas.core_types import AgentOutput
from src.pdil.models import ProvenanceHeader

class DealAnalysis(BaseModel):
    succeed_plus: float = Field(..., description="Probability of Succeed+ outcome")
    fail_plus: float = Field(..., description="Probability of Fail+ outcome")
    fail_minus: float = Field(..., description="Probability of Fail- outcome")
    days_to_completion: int = Field(..., description="Estimated days to completion")
    deal_report: str = Field(..., description="Detailed deal report outlining the rationale")

def calculate_market_implied_probability(S_t: float, S_minus_t: float, S_plus_t: float) -> float:
    """Calculates market-implied probability of a favorable outcome for target shareholders."""
    if S_plus_t == S_minus_t:
        return 0.5
    prob = (S_t - S_minus_t) / (S_plus_t - S_minus_t)
    return max(0.0, min(1.0, prob))

def call_llm_for_deal_analysis(deal_data: dict, context: str, p_m: float) -> dict:
    """Calls an LLM using litellm to get the probabilistic analysis of the deal."""
    prompt = f"""
    You are an expert merger arbitrage forecaster. Analyze the following deal.

    Deal Data: {json.dumps(deal_data)}
    Context: {context}
    Market Implied Probability of positive outcome: {p_m:.4f}

    Based on this information, provide:
    1. A probability distribution across three outcomes: Succeed+, Fail+, and Fail-. The sum must equal 1.0.
    2. An estimate of the days to completion.
    3. A detailed deal report explaining your reasoning.

    Format the response as a JSON object matching this schema exactly:
    {json.dumps(DealAnalysis.model_json_schema(), indent=2)}
    """

    # We use a mocked LLM response if the API key isn't provided or for testing
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        return {
            "succeed_plus": p_m * 0.9,
            "fail_plus": (1.0 - p_m * 0.9) * 0.2,
            "fail_minus": (1.0 - p_m * 0.9) * 0.8,
            "days_to_completion": deal_data.get('company_guided_days', 175),
            "deal_report": f"Mocked LLM deal report based on context: {context}. The implied probability is {p_m:.4f}."
        }

    try:
        response = completion(
            model="gpt-4o",  # or another frontier model per the paper
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        # Fallback for testing/robustness if the LLM call fails
        return {
            "succeed_plus": p_m * 0.9,
            "fail_plus": (1.0 - p_m * 0.9) * 0.2,
            "fail_minus": (1.0 - p_m * 0.9) * 0.8,
            "days_to_completion": deal_data.get('company_guided_days', 175),
            "deal_report": f"Error calling LLM: {str(e)}. Fallback analysis used."
        }


def analyze_deal(deal_data: dict, context: str) -> dict:
    """Simulates the core logic of the LLM-based merger-arbitrage forecaster."""
    S_t = float(deal_data.get('current_price', 100.0))
    S_minus_t = float(deal_data.get('downside_price', 80.0))
    S_plus_t = float(deal_data.get('upside_price', 120.0))

    p_m = calculate_market_implied_probability(S_t, S_minus_t, S_plus_t)

    # Call the LLM (or mock)
    llm_analysis = call_llm_for_deal_analysis(deal_data, context, p_m)

    return {
        "probabilities": {
            "Succeed+": llm_analysis.get("succeed_plus"),
            "Fail+": llm_analysis.get("fail_plus"),
            "Fail-": llm_analysis.get("fail_minus")
        },
        "days_to_completion": llm_analysis.get("days_to_completion"),
        "deal_report": llm_analysis.get("deal_report"),
        "market_implied_probability": p_m
    }

def process_merger_arbitrage(data: dict, context: str) -> AgentOutput:
    analysis = analyze_deal(data, context)

    content_str = json.dumps(analysis, sort_keys=True)
    content_hash = hashlib.sha256(content_str.encode('utf-8')).hexdigest()

    header = ProvenanceHeader(
        git_commit_hash="mock_hash_for_script",
        timestamp=datetime.now(timezone.utc).isoformat(),
        content_hash=content_hash,
        jsonLogic_version="1.0.0",
        confidence_score=0.92,
        derivation_path="scripts/merger_arbitrage_forecaster.py",
        source_data_object=data.get('deal_id', 'unknown_deal')
    )

    return AgentOutput(
        provenance_trace=header,
        data=analysis,
        observed_drift=False
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--deal_data', type=str, required=True, help="JSON string of deal data")
    parser.add_argument('--context', type=str, default="", help="Context for the analysis")
    args = parser.parse_args()

    data = json.loads(args.deal_data)
    result = process_merger_arbitrage(data, args.context)
    print(result.model_dump_json(indent=2))
