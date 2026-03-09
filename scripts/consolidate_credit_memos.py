import json
import os
import glob

DATA_DIR = "showcase/data"
INDEX_FILE = os.path.join(DATA_DIR, "credit_memo_library.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "unified_credit_memos.json")

def consolidate_memos():
    print(f"Reading index {INDEX_FILE}...")
    try:
        with open(INDEX_FILE, 'r') as f:
            index = json.load(f)
    except Exception as e:
        print(f"Error reading index: {e}")
        return

    # Deduplicate by ticker, preferring full reports (non-RAG) over RAG
    # In the index, RAG ones often have "_RAG" in id or file name
    unique_memos = {}
    for entry in index:
        ticker = entry.get("ticker")
        if not ticker:
            continue

        file_name = entry.get("file")
        if not file_name:
            continue

        is_rag = "_RAG" in file_name

        # Keep it if we haven't seen the ticker, or if the current one is better (non-RAG vs RAG)
        if ticker not in unique_memos:
            unique_memos[ticker] = entry
        else:
            current_is_rag = "_RAG" in unique_memos[ticker].get("file", "")
            if current_is_rag and not is_rag:
                unique_memos[ticker] = entry

    selected_entries = list(unique_memos.values())

    # Sort or limit to top 10 (e.g. by highest risk or just first 10)
    # Let's just take top 10 as requested
    selected_entries = selected_entries[:10]

    unified_data = []

    for entry in selected_entries:
        filepath = os.path.join(DATA_DIR, entry["file"])
        print(f"Loading {filepath}...")
        try:
            with open(filepath, 'r') as f:
                memo_data = json.load(f)

            # Embed some index metadata just in case
            memo_data["_metadata"] = {
                "id": entry.get("id"),
                "ticker": entry.get("ticker"),
                "sector": entry.get("sector"),
                "risk_score": entry.get("risk_score")
            }
            unified_data.append(memo_data)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    print(f"Consolidated {len(unified_data)} memos.")

    try:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(unified_data, f, indent=2)
        print(f"Successfully wrote {OUTPUT_FILE}")
    except Exception as e:
        print(f"Error writing output: {e}")

if __name__ == "__main__":
    consolidate_memos()
