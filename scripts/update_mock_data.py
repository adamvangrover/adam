import os
import json
import glob

def clean_json_text(text):
    if not text: return "{}"
    import re
    # Remove JS variable assignment if present
    text = re.sub(r'^window\.MOCK_DATA\s*=\s*', '', text)
    text = re.sub(r';\s*$', '', text)
    return text

def main():
    root_dir = os.getcwd()
    showcase_data_dir = os.path.join(root_dir, "showcase", "data")
    mock_data_js_path = os.path.join(root_dir, "showcase", "js", "mock_data.js")

    print(f"Reading mock data from {mock_data_js_path}...")

    current_data = {}
    if os.path.exists(mock_data_js_path):
        with open(mock_data_js_path, "r", encoding="utf-8") as f:
            content = f.read()
            try:
                json_str = clean_json_text(content)
                current_data = json.loads(json_str)
            except Exception as e:
                print(f"Error parsing existing mock data: {e}")
                current_data = {}

    # 1. Load Library Index
    library_path = os.path.join(showcase_data_dir, "credit_memo_library.json")
    if os.path.exists(library_path):
        with open(library_path, "r", encoding="utf-8") as f:
            current_data["credit_library"] = json.load(f)
            print(f"Loaded {len(current_data['credit_library'])} items from credit_memo_library.json")

    # 2. Load Individual Credit Memos
    credit_memos = current_data.get("credit_memos", {})

    memo_files = glob.glob(os.path.join(showcase_data_dir, "credit_memo_*.json"))
    count = 0
    for filepath in memo_files:
        filename = os.path.basename(filepath)
        if filename in ["credit_memo_library.json", "credit_memo_audit_log.json", "credit_memo_output.json"]:
            continue

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                memo_data = json.load(f)
                # Use filename or internal ID as key
                key = filename
                # Also try to map by borrower name if possible for easier lookup
                name_key = memo_data.get("borrower_name", "").replace(" ", "_").replace(".", "")

                credit_memos[key] = memo_data
                if name_key:
                    credit_memos[name_key] = memo_data

                count += 1
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    current_data["credit_memos"] = credit_memos
    print(f"Loaded {count} individual credit memos.")

    # 3. Write back to JS
    with open(mock_data_js_path, "w", encoding="utf-8") as f:
        f.write(f"window.MOCK_DATA = {json.dumps(current_data, indent=2)};")

    print(f"Updated {mock_data_js_path} successfully.")

if __name__ == "__main__":
    main()
