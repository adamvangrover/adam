import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine.consensus_engine import ConsensusEngine

logging.basicConfig(level=logging.INFO, format='%(message)s')

def run_consensus_test():
    """
    Test harness to verify algorithmic conflict resolution between agents
    and Human-In-The-Loop escalation.
    """
    engine = ConsensusEngine()
    
    print("\n" + "="*80)
    print("TEST 1: CLEAR ALGORITHMIC CONSENSUS (MODERATE PROFILE)")
    print("="*80)
    
    # 2 Bulls vs 1 Bear. Bulls have a slight confidence advantage.
    test1_outputs = [
        {"agent_source": "FundamentalAnalystAgent", "answer": "BULLISH", "confidence": 0.85},
        {"agent_source": "MarketSentimentAgent", "answer": "BULLISH", "confidence": 0.80},
        {"agent_source": "RiskAssessmentAgent", "answer": "BEARISH", "confidence": 0.75}
    ]
    
    result1 = engine.arbitrate(test1_outputs, risk_profile="moderate")
    print(f"\n[Final Decision]: {result1['decision']} (Conviction Share: {result1['confidence']*100:.1f}%)")
    print(f"[Requires HITL?]: {result1['requires_human_review']}")
    
    
    print("\n" + "="*80)
    print("TEST 2: CONSERVATIVE RISK PROFILE OVERRIDE")
    print("="*80)
    
    # The exact same inputs, but the user's risk profile severely shifts the weighting
    # towards the RiskAssessmentAgent (Bearish in this scenario).
    result2 = engine.arbitrate(test1_outputs, risk_profile="conservative")
    print(f"\n[Final Decision]: {result2['decision']} (Conviction Share: {result2['confidence']*100:.1f}%)")
    print(f"[Requires HITL?]: {result2['requires_human_review']}")


    print("\n" + "="*80)
    print("TEST 3: HIGH CONVICTION DEADLOCK (HITL ESCALATION)")
    print("="*80)
    
    # 1 Bull vs 1 Bear. Both have extreme conviction. System should refuse to guess.
    test3_outputs = [
        {"agent_source": "FundamentalAnalystAgent", "answer": "BULLISH", "confidence": 0.95},
        {"agent_source": "RiskAssessmentAgent", "answer": "BEARISH", "confidence": 0.93}
    ]
    
    result3 = engine.arbitrate(test3_outputs, risk_profile="moderate")
    print(f"\n[Final Decision]: {result3['decision']}")
    print(f"[Requires HITL?]: {result3['requires_human_review']}")
    print(f"[Reason]: {result3['reason']}")
    print("\nTEST HARNESS COMPLETE.")


if __name__ == "__main__":
    run_consensus_test()
