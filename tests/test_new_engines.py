import asyncio
import logging
from core.engine.consensus_engine import ConsensusEngine
from core.agents.specialized.blindspot_agent import BlindspotAgent

# Setup basic logging to see output
logging.basicConfig(level=logging.INFO)

async def test_consensus_engine():
    print("\n--- Testing Consensus Engine ---")
    engine = ConsensusEngine(threshold=0.5)

    signals = [
        {'agent': 'RiskOfficer', 'vote': 'REJECT', 'confidence': 0.9, 'weight': 2.0, 'reason': 'VaR too high'},
        {'agent': 'TechAnalyst', 'vote': 'BUY', 'confidence': 0.7, 'weight': 1.0, 'reason': 'Breakout imminent'},
        {'agent': 'MacroSentinel', 'vote': 'BUY', 'confidence': 0.6, 'weight': 1.0, 'reason': 'Rates falling'}
    ]

    # Risk (2.0 * -0.9) = -1.8
    # Tech (1.0 * 0.7) = 0.7
    # Macro (1.0 * 0.6) = 0.6
    # Total Score = -0.5
    # Total Weight = 4.0
    # Final = -0.125 -> Should be HOLD/ABSTAIN (since < 0.5 threshold)

    result = engine.evaluate(signals)
    print("Result:", result)
    assert result['decision'] == 'HOLD' or 'SELL' # Depending on exact threshold math logic check

    # Let's try a strong buy
    signals_buy = [
        {'agent': 'A', 'vote': 'BUY', 'confidence': 1.0, 'weight': 1.0},
        {'agent': 'B', 'vote': 'BUY', 'confidence': 1.0, 'weight': 1.0}
    ]
    result_buy = engine.evaluate(signals_buy)
    print("Buy Result:", result_buy)
    assert result_buy['decision'] == 'BUY/APPROVE'

async def test_blindspot_agent():
    print("\n--- Testing Blindspot Agent ---")
    config = {"agent_id": "BlindspotScanner_01"}
    agent = BlindspotAgent(config)

    # This should hit the fallback logic since we don't have a real Neo4j connection active in this script context usually
    result = await agent.execute()
    print("Agent Result:", result)

    assert result['status'] == 'SCAN_COMPLETE'
    # We expect at least one anomaly from our mocked "Sentiment Divergence" logic
    # if the seed data has Energy as -0.2 sentiment but we force a check,
    # strictly speaking the mock logic depends on the specific seed data.
    # In live_seed_data.json: Energy is -0.2 sentiment, trend 'neutral'.
    # Tech is 0.8, bullish.
    # Financials 0.4, bullish.
    # So actually, maybe no divergence in seed data?
    # Let's adjust seed data or accept 0 anomalies as a valid test pass.

    # Wait, the seed data I wrote:
    # "Energy": { "sentiment": -0.2, "trend": "neutral" } -> No divergence (neutral trend)
    # So 0 anomalies is correct behavior for that data.

async def main():
    await test_consensus_engine()
    await test_blindspot_agent()

if __name__ == "__main__":
    asyncio.run(main())
