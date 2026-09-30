import json
import argparse
import os
from datetime import datetime, timezone
import hashlib
from typing import Dict, Any
import math
import numpy as np
import statistics

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

def calculate_downside_price(S_minus_0: float, beta: float, market_return: float) -> float:
    """
    Calculates expected target price if the deal fails using the formula from the paper:
    S^-_t = S^-_0 exp(beta * r_{m,[0,t]})
    """
    return S_minus_0 * math.exp(beta * market_return)

def calculate_brier_scores(predictions: list[float], truths: list[int], S_plus_t_list: list[float], S_minus_t_list: list[float], S_t_list: list[float], p_m_t_list: list[float]) -> dict:
    """
    Calculates various Brier scores from the paper.
    predictions: predicted probability of positive outcome (Succeed+ or Fail+)
    truths: 1 if positive outcome, 0 if negative outcome (Fail-)
    """
    n = len(predictions)
    if n == 0:
        return {}

    brier = np.mean((np.array(truths) - np.array(predictions))**2)

    # Class-balanced Brier (BrierB)
    pos_count = sum(truths)
    neg_count = n - pos_count
    
    brier_b = 0.0
    if pos_count > 0 and neg_count > 0:
        weights_b = np.array([1.0 / pos_count if t == 1 else 1.0 / neg_count for t in truths])
        weights_b = weights_b / np.sum(weights_b) * n # normalize to mean 1
        brier_b = np.mean(weights_b * (np.array(truths) - np.array(predictions))**2)

    # Surprise-weighted Brier (BrierS)
    weights_s = np.array([1.0 - p_m if t == 1 else p_m for t, p_m in zip(truths, p_m_t_list)])
    if np.sum(weights_s) > 0:
        weights_s = weights_s / np.sum(weights_s) * n
        brier_s = np.mean(weights_s * (np.array(truths) - np.array(predictions))**2)
    else:
        brier_s = 0.0

    # P&L-weighted Brier (Brier$)
    weights_pl = []
    for i in range(n):
        if truths[i] == 1:
            w = (S_plus_t_list[i] - S_t_list[i]) / S_t_list[i] if S_t_list[i] > 0 else 0.0
        else:
            w = (S_t_list[i] - S_minus_t_list[i]) / S_t_list[i] if S_t_list[i] > 0 else 0.0
        weights_pl.append(w)
    
    weights_pl = np.array(weights_pl)
    if np.sum(weights_pl) > 0:
        weights_pl = np.clip(weights_pl, 0, np.percentile(weights_pl, 95)) # clip outliers
        weights_pl = weights_pl / np.sum(weights_pl) * n
        brier_pl = np.mean(weights_pl * (np.array(truths) - np.array(predictions))**2)
    else:
        brier_pl = 0.0

    return {
        "Brier": float(brier),
        "BrierB": float(brier_b),
        "BrierS": float(brier_s),
        "Brier$": float(brier_pl)
    }


def calculate_market_implied_probability(S_t: float, S_minus_t: float, S_plus_t: float) -> float:
    """Calculates market-implied probability of a favorable outcome for target shareholders."""
    if S_plus_t == S_minus_t:
        return 0.5
    prob = (S_t - S_minus_t) / (S_plus_t - S_minus_t)
    return max(0.0, min(1.0, prob))

def _mock_llm_response(deal_data: dict, context: str, p_m: float, i: int) -> dict:
    return {
        "succeed_plus": p_m * 0.9,
        "fail_plus": (1.0 - p_m * 0.9) * 0.2,
        "fail_minus": (1.0 - p_m * 0.9) * 0.8,
        "days_to_completion": deal_data.get('company_guided_days', 175),
        "deal_report": f"Mocked LLM deal report {i} based on context: {context}."
    }

def call_llm_for_deal_analysis(deal_data: dict, context: str, p_m: float) -> dict:
    """Calls an LLM using litellm to get the probabilistic analysis of the deal with N=5 ensemble."""
    prompt = f"""
    You are an expert merger arbitrage forecaster. Analyze the following deal.

    Deal Data: {json.dumps(deal_data)}
    Context: {context}
    Market Implied Probability of positive outcome: {p_m:.4f}
    
    Start from historical base rates (completion frequencies), incorporate deal-specific evidence, 
    and explicitly weigh green flags against red flags.

    Based on this information, provide:
    1. A probability distribution across three outcomes: Succeed+, Fail+, and Fail-. The sum must equal 1.0.
    2. An estimate of the days to completion.
    3. A detailed deal report explaining your reasoning.

    Format the response as a JSON object matching this schema exactly:
    {json.dumps(DealAnalysis.model_json_schema(), indent=2)}
    """

    N_ENSEMBLE = 5
    analyses = []

    # We use a mocked LLM response if the API key isn't provided or for testing
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"):
        analyses = [_mock_llm_response(deal_data, context, p_m, i) for i in range(N_ENSEMBLE)]
    else:
        try:
            for _ in range(N_ENSEMBLE):
                response = completion(
                    model="gpt-4o",  # or another frontier model per the paper
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.2
                )
                content = response.choices[0].message.content
                analyses.append(json.loads(content))
        except Exception as e:
            analyses = [_mock_llm_response(deal_data, context, p_m, i) for i in range(N_ENSEMBLE)]
            analyses[0]["deal_report"] = f"Error calling LLM: {str(e)}. Fallback analysis used."
            
    # Aggregation & verification via Median
    succeed_plus_vals = [a.get("succeed_plus", 0.0) for a in analyses]
    fail_plus_vals = [a.get("fail_plus", 0.0) for a in analyses]
    fail_minus_vals = [a.get("fail_minus", 0.0) for a in analyses]
    days_vals = [a.get("days_to_completion", 175) for a in analyses]
    
    med_succeed_plus = statistics.median(succeed_plus_vals)
    med_fail_plus = statistics.median(fail_plus_vals)
    med_fail_minus = statistics.median(fail_minus_vals)
    
    # Normalize probabilities
    total_prob = med_succeed_plus + med_fail_plus + med_fail_minus
    if total_prob > 0:
        med_succeed_plus /= total_prob
        med_fail_plus /= total_prob
        med_fail_minus /= total_prob
        
    med_days = int(statistics.median(days_vals))
    
    # Draft reports for secondary LLM consolidation
    draft_reports = [a.get("deal_report", "") for a in analyses]
    consolidated_report = " | ".join(draft_reports) # Fallback / simple join
    
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"):
        try:
             consolidation_prompt = f"Cross-check and consolidate the following {N_ENSEMBLE} deal reports into a single, factual, fact-checked report:\n\n" + "\n\n---\n\n".join(draft_reports)
             response = completion(
                model="gpt-4o",
                messages=[{"role": "user", "content": consolidation_prompt}],
                temperature=0.0
             )
             consolidated_report = response.choices[0].message.content
        except Exception:
             pass

    return {
        "succeed_plus": med_succeed_plus,
        "fail_plus": med_fail_plus,
        "fail_minus": med_fail_minus,
        "days_to_completion": med_days,
        "deal_report": consolidated_report
    }


def analyze_deal(deal_data: dict, context: str) -> dict:
    """Simulates the core logic of the LLM-based merger-arbitrage forecaster."""
    S_t = float(deal_data.get('current_price', 100.0))
    
    if 'S_minus_0' in deal_data and 'beta' in deal_data and 'market_return' in deal_data:
        S_minus_t = calculate_downside_price(
            deal_data['S_minus_0'], 
            deal_data['beta'], 
            deal_data['market_return']
        )
    else:
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
