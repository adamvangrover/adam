import os
import json
import sys
import glob
from datetime import datetime

# Ensure we can import from core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.pipelines.mock_edgar import MockEdgar

def main():
    print("Starting Mock Data Consolidation...")

    showcase_dir = os.path.join("showcase", "data")
    output_path = os.path.join("showcase", "js", "mock_data.js")

    # 1. Load Library Index
    library_path = os.path.join(showcase_dir, "credit_memo_library.json")
    if os.path.exists(library_path):
        with open(library_path, "r") as f:
            library_index = json.load(f)
    else:
        print("Warning: Library index not found.")
        library_index = []

    # 2. Load Individual Memos
    credit_memos = {}
    memo_files = glob.glob(os.path.join(showcase_dir, "credit_memo_*.json"))

    for memo_file in memo_files:
        if "library" in memo_file or "audit_log" in memo_file:
            continue

        try:
            with open(memo_file, "r") as f:
                memo_data = json.load(f)
                # Key by borrower name for easy lookup
                borrower_name = memo_data.get("borrower_name")
                if borrower_name:
                    credit_memos[borrower_name] = memo_data
        except Exception as e:
            print(f"Error loading {memo_file}: {e}")

    # 3. Construct Final Object
    # Instantiate MockEdgar to access instance methods if needed
    mock_edgar = MockEdgar()

    mock_data = {
        "stats": {
            "version": "26.0 (Sovereign)",
            "generated_at": datetime.utcnow().isoformat(),
            "status": "ONLINE"
        },
        "credit_library": library_index,
        "credit_memos": credit_memos,
        "market_data": {
            "tickers": mock_edgar.list_tickers(),
            "financials_db_snapshot": MockEdgar.FINANCIALS_DB
        }
    }

    # 4. Write to JS File
    js_content = f"window.MOCK_DATA = {json.dumps(mock_data, indent=2)};"

    with open(output_path, "w") as f:
        f.write(js_content)

    print(f"Successfully wrote mock data to {output_path}")
    print(f"Consolidated {len(credit_memos)} credit memos.")

if __name__ == "__main__":
    main()
