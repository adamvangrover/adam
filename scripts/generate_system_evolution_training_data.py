import json
import random
import datetime

# --- Constants & Templates ---

# Instead of string templates, we define functions that return the data structure

def generate_timestamp():
    # Generate a random timestamp within the last 30 days
    start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    random_seconds = random.randint(0, 30 * 24 * 60 * 60)
    return (start_date + datetime.timedelta(seconds=random_seconds)).isoformat() + "Z"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "GS"]
HEADLINES = [
    ("Tech giant surges to record highs on AI optimism", 0.9, 0.95),
    ("Market crashes as inflation fears return", -0.85, 0.9),
    ("Company reports mixed earnings, guidance flat", -0.1, 0.6),
    ("CEO steps down amid scandal allegations", -0.9, 0.95),
    ("New product launch delayed by supply chain issues", -0.6, 0.8)
]

ANALYSES = [
    ("The company is a strong buy due to robust cash flows.", "B", "Lacks specific numerical backing.", "Include recent free cash flow yield data."),
    ("Sell immediately. The chart looks bad.", "C-", "Purely technical, ignores fundamentals.", "Combine technicals with fundamental analysis."),
    ("Hold. Valuation is fair at 15x forward earnings, but growth is slowing to 5%.", "A", "Balanced and data-driven.", "Good use of forward multiples.")
]

def generate_evolution_record():
    timestamp = generate_timestamp()
    if random.random() > 0.5:
        prompt = "You are the System Evolution Monitor. Detect any version changes or self-improvements in the Adam platform."
        completion_obj = {
            "event": "System_Upgrade",
            "version_from": "v23.0",
            "version_to": "v23.5",
            "trigger": "Autonomous_Self_Improvement_Loop",
            "timestamp": timestamp,
            "description": "Meta-Cognitive Agent detected drift in Sentiment Analysis module. Initiated retraining cycle. New weights deployed."
        }
    else:
        prompt = "You are the Architecture Log. Record the deployment of a new agent capability."
        completion_obj = {
            "event": "Capability_Deployment",
            "agent": "RiskAssessmentAgent",
            "new_capability": "Deep_Learning_VaR_Model_v2",
            "timestamp": timestamp,
            "status": "Active",
            "validation_score": 0.98
        }
    return {"prompt": prompt, "completion": json.dumps(completion_obj)}

def generate_swarm_record():
    timestamp = generate_timestamp()
    task_id = f"task_{random.randint(1000, 9999)}"
    if random.random() > 0.5:
        prompt = "You are the Swarm Orchestrator. Log the delegation of a complex task."
        completion_obj = {
            "action": "Task_Delegation",
            "source": "Orchestrator",
            "target": "MacroeconomicAnalysisAgent",
            "task_id": task_id,
            "task_type": "Market_Overview",
            "priority": "High",
            "timestamp": timestamp
        }
    else:
        prompt = "You are a Specialist Agent. Report the completion of a sub-task back to the swarm."
        completion_obj = {
            "action": "Task_Completion",
            "source": "MacroeconomicAnalysisAgent",
            "target": "Orchestrator",
            "task_id": task_id,
            "result_summary": "Inflation trending down to 2.8%.",
            "confidence": 0.92,
            "timestamp": timestamp
        }
    return {"prompt": prompt, "completion": json.dumps(completion_obj)}

def generate_digital_twin_record():
    timestamp = generate_timestamp()
    ticker = random.choice(TICKERS)
    if random.random() > 0.5:
        prompt = f"You are the Digital Twin Synchronization Engine. Update the state of a financial entity ({ticker})."
        pe = round(random.uniform(10, 50), 2)
        completion_obj = {
            "entity": ticker,
            "metric": "PE_Ratio",
            "value": pe,
            "source": "Live_Market_Feed",
            "timestamp": timestamp,
            "provenance": "Bloomberg_API_v3"
        }
    else:
        prompt = f"You are the Digital Twin Monitor. Flag a discrepancy between the simulated twin and real-world data for {ticker}."
        sim_beta = round(random.uniform(0.5, 2.0), 2)
        real_beta = round(sim_beta * random.uniform(0.9, 1.1), 2)
        div = f"{round(abs(sim_beta - real_beta) / real_beta * 100, 1)}%"
        completion_obj = {
            "alert": "Twin_Divergence",
            "entity": ticker,
            "metric": "Beta",
            "simulated_value": sim_beta,
            "real_world_value": real_beta,
            "divergence": div,
            "timestamp": timestamp
        }
    return {"prompt": prompt, "completion": json.dumps(completion_obj)}

def generate_time_series_record():
    timestamp = generate_timestamp()
    ticker = random.choice(TICKERS)
    prompt = f"Ingest the following time-series data point for {ticker}."
    price = round(random.uniform(100, 3000), 2)
    completion_obj = {
        "type": "Time_Series_Point",
        "ticker": ticker,
        "interval": "1m",
        "open": price,
        "high": round(price*1.01, 2),
        "low": round(price*0.99, 2),
        "close": round(price*1.005, 2),
        "volume": random.randint(100000, 5000000),
        "timestamp": timestamp
    }
    return {"prompt": prompt, "completion": json.dumps(completion_obj)}

def generate_sentiment_record():
    timestamp = generate_timestamp()
    headline, sent, conv = random.choice(HEADLINES)
    prompt = f"Analyze the sentiment and conviction of this headline: '{headline}'"
    completion_obj = {
        "sentiment_score": sent,
        "conviction_score": conv,
        "reasoning": "Strong positive language ('surges', 'record highs') indicates high bullish sentiment. Conviction is high due to specific numerical targets mentioned.",
        "timestamp": timestamp
    }
    return {"prompt": prompt, "completion": json.dumps(completion_obj)}

def generate_judge_record():
    timestamp = generate_timestamp()
    analysis, grade, critique, feedback = random.choice(ANALYSES)
    prompt = f"You are the LLM Judge. Critique the following analysis from the Risk Agent: '{analysis}'"
    completion_obj = {
        "grade": grade,
        "critique": critique,
        "feedback": feedback,
        "timestamp": timestamp
    }
    return {"prompt": prompt, "completion": json.dumps(completion_obj)}

def generate_data(num_records=100):
    data = []
    generators = [
        generate_evolution_record,
        generate_swarm_record,
        generate_digital_twin_record,
        generate_time_series_record,
        generate_sentiment_record,
        generate_judge_record
    ]

    for _ in range(num_records):
        gen = random.choice(generators)
        data.append(gen())
    return data

def main():
    output_path = "data/artisanal_training_sets/artisanal_data_system_evolution_v1.jsonl"
    print(f"Generating synthetic data to {output_path}...")

    data = generate_data(100)

    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

    print(f"Successfully generated {len(data)} records.")

if __name__ == "__main__":
    main()
