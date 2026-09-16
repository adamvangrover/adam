import json
import time
from datetime import datetime
import requests

BASE_URL = "https://headlinearena.com"

def fetch_open_challenges():
    print(f"Fetching open challenges from {BASE_URL}...")
    url = f"{BASE_URL}/api/v1/eval/challenges?status=open"
    headers = {"User-Agent": "adam-static-generator/1.0"}

    try:
        response = requests.get(url, headers=headers, timeout=10)

        # If the API requires auth even for GETs, or we hit a 404/403 before integration,
        # we will gracefully fallback to the known standard daily markets.
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                if "items" in data and isinstance(data["items"], list):
                    return data["items"]
                elif "data" in data and isinstance(data["data"], list):
                    return data["data"]
                elif "challenges" in data and isinstance(data["challenges"], list):
                    return data["challenges"]
            return [] # fallback to empty list instead of full object
        else:
            raise Exception(f"HTTP {response.status_code}")

    except Exception as e:
        print(f"[!] Live fetch failed ({e}). Loading standard macro daily fallback targets...")
        return [
            {"id": "CHAL-ZN-DAILY", "title": "Will 10-Year Treasury futures rise, fall, or stay neutral?", "asset": "ZN"},
            {"id": "CHAL-ES-DAILY", "title": "Will E-mini S&P 500 futures rise, fall, or stay neutral?", "asset": "ES"},
            {"id": "CHAL-DXY-DAILY", "title": "Will US Dollar Index futures rise or fall?", "asset": "DXY"},
            {"id": "CHAL-CL-DAILY", "title": "Will Crude Oil futures rise or fall?", "asset": "CL"},
            {"id": "CHAL-GC-DAILY", "title": "Will Gold Futures rise, fall, or stay neutral?", "asset": "GC"}
        ]

def generate_markdown(challenges):
    date_str = datetime.now().strftime("%Y-%m-%d")
    timestamp_full = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    filename = f"forecasts_{date_str}.md"

    md_content = f"# Headline Arena Static Forecasts\n\n"
    md_content += f"**Execution Epoch:** {timestamp_full}\n"
    md_content += f"**Status:** Pre-API Integration / Git-Timestamped Evidence\n\n"
    md_content += f"---\n\n"

    if not challenges:
        md_content += "> *No open challenges retrieved.*\n"
    else:
        for chal in challenges:
            c_id = chal.get("id", "UNKNOWN")
            title = chal.get("question", chal.get("title", "Unknown Target"))
            asset = chal.get("asset", "Unknown")
            deadline = chal.get("deadline", "Pending Resolution")

            md_content += f"### {title}\n"
            md_content += f"- **Challenge ID:** `{c_id}` | **Target:** `{asset}`\n"
            md_content += f"- **Deadline:** {deadline}\n"
            md_content += f"- **Direction:** `[ INSERT: bullish | bearish | neutral ]`\n"
            md_content += f"- **Confidence:** `[ INSERT: 0.50 to 1.00 ]`\n"
            md_content += f"- **System-2 Reasoning:**\n"
            md_content += f"  > [ INSERT ADAM RISK TELEMETRY AND MACRO THESIS HERE ]\n\n"
            md_content += f"---\n\n"

    with open(filename, "w") as f:
        f.write(md_content)

    print(f"\nSuccess! Generated Git-ready markdown template: {filename}")
    print("Pipeline instructions: Have 'adam' overwrite the brackets, then run `git commit -am 'Daily Macro Forecast'`.")

if __name__ == "__main__":
    open_challenges = fetch_open_challenges()
    generate_markdown(open_challenges)
