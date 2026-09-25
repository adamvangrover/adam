import json
import os

def generate_data():
    qa_pairs = [
        {"q": "What happens to credit spreads during a liquidity crisis?", "a": "Credit spreads typically widen due to increased risk premiums and lack of market liquidity."},
        {"q": "How does an inverted yield curve affect model RWA?", "a": "It often signals impending economic stress, prompting models to stress PDs and subsequently increase RWA."}
    ]

    risk_buckets = {
        "buckets": [
            {"name": "Investment Grade", "pd_range": [0.0, 0.02]},
            {"name": "High Yield", "pd_range": [0.02, 0.15]},
            {"name": "Distressed", "pd_range": [0.15, 1.0]}
        ]
    }

    historic_shocks = {
        "shocks": [
            {"event": "2008 Financial Crisis", "spread_widening_bps": 500},
            {"event": "2020 COVID Shock", "spread_widening_bps": 350}
        ]
    }

    os.makedirs("evals/training_sets/qa", exist_ok=True)
    os.makedirs("evals/training_sets/risk_buckets", exist_ok=True)
    os.makedirs("evals/training_sets/historic_shocks", exist_ok=True)

    with open("evals/training_sets/qa/qa_pairs.jsonl", "w") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair) + "\n")

    with open("evals/training_sets/risk_buckets/representative_buckets.json", "w") as f:
        json.dump(risk_buckets, f, indent=2)

    with open("evals/training_sets/historic_shocks/shock_profiles.json", "w") as f:
        json.dump(historic_shocks, f, indent=2)

if __name__ == "__main__":
    generate_data()
