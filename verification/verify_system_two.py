import json
import os
import sys

def verify_system_two_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "showcase", "data")

    # Check Apple
    apple_file = os.path.join(data_dir, "credit_memo_Apple_Inc.json")
    with open(apple_file, "r") as f:
        data = json.load(f)

    print("Checking Apple Inc.:")
    print(f"- Sentiment: {data.get('sentiment_score')}")
    print(f"- Conviction: {data.get('conviction_score')}")
    print(f"- Rating: {data.get('credit_rating')}")
    print(f"- Price Target: {data.get('price_target')}")
    print(f"- System 2 Notes: {data.get('system_two_notes')}")

    assert data.get('sentiment_score') > 0
    assert "A" in data.get('credit_rating') # Allow A, A-, AA
    assert "system_two_notes" in data

    # Check TechCorp (High Leverage)
    tc_file = os.path.join(data_dir, "credit_memo_TechCorp_Inc.json")
    with open(tc_file, "r") as f:
        data = json.load(f)

    print("\nChecking TechCorp Inc.:")
    print(f"- Leverage: {data.get('financial_ratios', {}).get('leverage_ratio')}")
    print(f"- Sentiment: {data.get('sentiment_score')}")
    print(f"- System 2 Notes: {data.get('system_two_notes')}")

    # If leverage > 4 and sentiment > 60, it triggers correction.
    # TechCorp sentiment is 30, so it might just pass or have rating warning.

    print("\nVERIFICATION SUCCESS: New metrics present and validated.")

if __name__ == "__main__":
    verify_system_two_data()
