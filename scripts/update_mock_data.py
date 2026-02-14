import json
import os
import glob

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    print("Updating mock_data.js...")

    # Base Data
    mock_data = {
        "stats": {
            "version": "23.5",
            "status": "HYBRID_ONLINE",
            "cpu_load": 12,
            "memory_usage": 34,
            "active_tasks": 4
        },
        "files": [], # Minimal files list for mock
        "financial_data": {
            "synthetic_stock_data.csv": [],
            "synthetic_black_swan_scenario.csv": []
        }
    }

    # 1. Load Credit Library
    library = load_json("showcase/data/credit_memo_library.json")
    if library:
        mock_data["credit_library"] = library
        print(f"Loaded {len(library)} items from Credit Library.")

    # 2. Load Credit Memos (Individual Files)
    mock_data["credit_memos"] = {}
    memo_files = glob.glob("showcase/data/credit_memo_*.json")
    for f in memo_files:
        if "library" in f or "output" in f or "audit" in f: continue
        memo = load_json(f)
        if memo:
            # ID from filename: credit_memo_Apple_Inc.json -> Apple_Inc
            id_ = os.path.basename(f).replace("credit_memo_", "").replace(".json", "")
            mock_data["credit_memos"][id_] = memo
    print(f"Loaded {len(mock_data['credit_memos'])} Credit Memos.")

    # 3. Load Sovereign Artifacts
    mock_data["sovereign_data"] = {}
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META"]

    sovereign_dir = "showcase/data/sovereign_artifacts"
    if os.path.exists(sovereign_dir):
        for ticker in tickers:
            spread = load_json(os.path.join(sovereign_dir, f"{ticker}_spread.json"))
            memo = load_json(os.path.join(sovereign_dir, f"{ticker}_memo.json"))
            audit = load_json(os.path.join(sovereign_dir, f"{ticker}_audit.json"))
            report = load_json(os.path.join(sovereign_dir, f"{ticker}_report.json"))

            if spread and memo:
                mock_data["sovereign_data"][ticker] = {
                    "spread": spread,
                    "memo": memo,
                    "audit": audit,
                    "report": report
                }
        print(f"Loaded {len(mock_data['sovereign_data'])} Sovereign Artifacts.")

    # 4. Load Market Mayhem
    mm_data = load_json("showcase/data/market_mayhem_dec_2025.json")
    if mm_data:
        mock_data["market_mayhem"] = mm_data
        print("Loaded Market Mayhem Data.")

    # 5. Load Financial Data (Preserve existing if possible, or use minimal defaults)
    # Ideally read existing mock_data.js and extract, but parsing JS in Python is hard.
    # We will use the truncated data I saw earlier or just valid dummy data.
    # The read_file output for mock_data.js had `financial_data` populated.
    # I'll manually paste a small valid subset to keep file size manageable but functional.
    mock_data["financial_data"]["synthetic_stock_data.csv"] = [
        {"time": "0.1", "value": 0.50}, {"time": "0.2", "value": 0.51},
        {"time": "0.3", "value": 0.52}, {"time": "0.4", "value": 0.53},
        {"time": "0.5", "value": 0.54}, {"time": "0.6", "value": 0.55}
    ] # Simplified

    # Write to JS file
    output_path = "showcase/js/mock_data.js"
    json_str = json.dumps(mock_data, indent=2)
    js_content = f"window.MOCK_DATA = {json_str};\n"

    with open(output_path, "w") as f:
        f.write(js_content)

    print(f"Successfully generated {output_path}")

if __name__ == "__main__":
    main()
