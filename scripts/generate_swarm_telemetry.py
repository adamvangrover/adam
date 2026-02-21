import json
import random
import time
from datetime import datetime, timedelta

AGENTS = [
    "MetaOrchestrator", "RiskAssessmentAgent", "FraudDetectionAgent",
    "CryptoArbitrageAgent", "DefiLiquidityAgent", "BlindspotScanner",
    "FundamentalAnalyst", "TechnicalAnalyst", "SentimentAnalyst",
    "PortfolioManager", "CreditRiskController", "SovereignAIAnalyst",
    "QuantumRiskAgent", "MarketOracle", "NewsBot", "ReportGenerator"
]

EVENT_TYPES = [
    "REQUEST_RECEIVED", "THOUGHT_TRACE", "TASK_START", "TASK_COMPLETE",
    "INTER_AGENT_COMM", "ANOMALY_DETECTED", "KNOWLEDGE_GRAPH_UPDATE"
]

TICKERS = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "BTC/USDT", "ETH/USDT"]

def generate_timestamp(base_time, offset_seconds):
    new_time = base_time + timedelta(seconds=offset_seconds, microseconds=random.randint(1000, 999999))
    return new_time.isoformat()

def create_event(timestamp, event_type, agent_id, details):
    return {
        "timestamp": timestamp,
        "event_type": event_type,
        "agent_id": agent_id,
        "details": details
    }

def generate_telemetry(num_events=500, output_file="swarm_telemetry_expanded.jsonl"):
    current_time = datetime.fromisoformat("2026-02-20T08:00:00.000000")

    with open(output_file, 'w') as f:
        # 1. System Boot
        f.write(json.dumps(create_event(
            current_time.isoformat(), "SYSTEM_BOOT", "MetaOrchestrator", {"version": "v24.5.0-Omega"}
        )) + "\n")

        for i in range(num_events):
            current_time += timedelta(seconds=random.uniform(0.1, 5.5))
            agent = random.choice(AGENTS)
            event_type = random.choice(EVENT_TYPES)
            target = random.choice(TICKERS)

            details = {}

            if event_type == "REQUEST_RECEIVED":
                agent = "MetaOrchestrator"
                details = {"query": f"Analyze risk exposure for {target}", "context_keys": ["macro_environment", "live_market_data"]}

            elif event_type == "THOUGHT_TRACE":
                thoughts = [
                    f"Evaluating cross-asset correlation for {target}.",
                    f"Detected high volatility in sector. Adjusting confidence score.",
                    "Awaiting input from SentimentAnalyst before finalizing.",
                    "Complexity high. Distributing sub-tasks to QuantumRiskAgent.",
                    "Cache hit for recent SEC filings. Skipping extraction phase."
                ]
                details = {"content": random.choice(thoughts), "confidence": round(random.uniform(0.6, 0.99), 2)}

            elif event_type == "TASK_START":
                details = {
                    "inputs": {"target": target, "parameters": {"timeframe": "1Y", "granularity": "1D"}},
                    "args": ["--deep-scan", "--ignore-cache"]
                }

            elif event_type == "TASK_COMPLETE":
                details = {
                    "output": {
                        "status": "success",
                        "risk_score": round(random.uniform(0.1, 0.9), 3),
                        "latency_ms": random.randint(150, 4500),
                        "recommendation": random.choice(["HOLD", "BUY", "SELL", "HEDGE"])
                    }
                }

            elif event_type == "INTER_AGENT_COMM":
                receiver = random.choice([a for a in AGENTS if a != agent])
                details = {
                    "receiver_id": receiver,
                    "message_type": "DATA_HANDOFF",
                    "payload_summary": f"Passing normalized tensor data for {target}"
                }

            elif event_type == "ANOMALY_DETECTED":
                agent = random.choice(["FraudDetectionAgent", "BlindspotScanner", "QuantumRiskAgent"])
                details = {
                    "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
                    "description": f"Irregular volume spike detected on dark pools for {target}.",
                    "deviation_sigma": round(random.uniform(2.5, 6.0), 2)
                }

            elif event_type == "KNOWLEDGE_GRAPH_UPDATE":
                details = {
                    "action": "UPSERT",
                    "node": f"{target}_MarketRegime",
                    "edges_added": random.randint(2, 15)
                }

            event = create_event(current_time.isoformat(), event_type, agent, details)
            f.write(json.dumps(event) + "\n")

    print(f"Successfully generated {num_events} telemetry events in {output_file}")

if __name__ == "__main__":
    # Generate 1000 lines of telemetry data
    generate_telemetry(1000)
