import sys
import os
import json
import time
import datetime
import random
from typing import List, Dict

# Ensure root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine.live_mock_engine import LiveMockEngine
from core.system.provenance_logger import ProvenanceLogger, ActivityType

def run_simulation():
    print("Initializing Enterprise Simulation...")
    # Ensure logs directory exists
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = ProvenanceLogger(log_dir=log_dir + "/", use_jsonl=True)
    engine = LiveMockEngine()

    # Define simulation phases
    phases = [
        ("T+0", 0),
        ("T+1", 1),
        ("T+7", 7)
    ]

    base_time = datetime.datetime.utcnow()

    for phase_name, days_offset in phases:
        print(f"\n--- Simulating {phase_name} ---")
        current_sim_time = base_time + datetime.timedelta(days=days_offset)

        # Simulate market updates
        # We run a few ticks to simulate intraday movement
        for i in range(5):
            pulse = engine.get_market_pulse()

            # Log the market state
            logger.log_activity(
                agent_id="MarketSimulator",
                activity_type=ActivityType.SIMULATION,
                input_data={"phase": phase_name, "tick": i},
                output_data=pulse,
                metadata={"simulated_time": current_sim_time.isoformat()}
            )

        # Run Synthesizer/Consensus
        score = engine.get_synthesizer_score()
        print(f"[{phase_name}] Synthesizer Score: {score.get('normalized_score')} ({score.get('decision', 'N/A')})")

        logger.log_activity(
            agent_id="Synthesizer",
            activity_type=ActivityType.DECISION,
            input_data={"phase": phase_name, "market_context": "simulated"},
            output_data=score,
            metadata={"simulated_time": current_sim_time.isoformat()}
        )

        # Run a specialized agent task (e.g. Credit Memo)
        ticker = random.choice(["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"])
        memo = engine.generate_credit_memo(ticker, name=f"{ticker} Corp", sector="Tech")
        print(f"[{phase_name}] Generated Credit Memo for {ticker}")

        logger.log_activity(
            agent_id="CreditAnalyst",
            activity_type=ActivityType.GENERATION,
            input_data={"ticker": ticker, "phase": phase_name},
            output_data={"memo_summary": memo['memo'][:100] + "..."}, # truncated for log
            metadata={"simulated_time": current_sim_time.isoformat()},
            capture_full_io=True
        )

        # Simulate Geopolitical Event
        geo_event = engine.get_geopolitical_event()
        print(f"[{phase_name}] Geo Event: {geo_event['text']}")
        logger.log_activity(
            agent_id="GeopoliticalAnalyst",
            activity_type=ActivityType.SIMULATION,
            input_data={"phase": phase_name},
            output_data=geo_event,
            metadata={"simulated_time": current_sim_time.isoformat()}
        )

    print("\nSimulation Complete.")

    # Generate Derived Logs
    print("Generating Training and Human Review Logs...")
    generate_derived_logs(logger.jsonl_path)

def generate_derived_logs(source_path):
    training_log_path = os.path.join(os.path.dirname(source_path), "training_logs.jsonl")
    human_log_path = os.path.join(os.path.dirname(source_path), "human_review_logs.jsonl")

    with open(source_path, 'r') as src, \
         open(training_log_path, 'w') as train_out, \
         open(human_log_path, 'w') as human_out:

        for line in src:
            try:
                entry = json.loads(line)

                # Training Log: Format for fine-tuning (Input -> Output)
                if entry['activity_type'] in ['decision', 'generation']:
                    training_entry = {
                        "prompt": json.dumps(entry['entity_context']['inputs']),
                        "completion": json.dumps(entry['entity_context']['outputs'])
                    }
                    train_out.write(json.dumps(training_entry) + "\n")

                # Human Review Log: Readable format with metadata
                human_entry = {
                    "timestamp": entry['timestamp'],
                    "agent": entry['agent_id'],
                    "action": entry['activity_type'],
                    "summary": f"Agent {entry['agent_id']} performed {entry['activity_type']}",
                    "risk_score": entry.get('metadata', {}).get('risk_score', 'N/A'),
                    "provenance": entry.get('build_provenance', {})
                }
                human_out.write(json.dumps(human_entry) + "\n")

            except Exception as e:
                print(f"Skipping malformed line: {e}")

if __name__ == "__main__":
    run_simulation()
