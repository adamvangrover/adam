import json
import re
import os

def extract_seed_data():
    input_path = 'showcase/js/mock_data.js'

    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Strip variable assignment
    # Assumption: File starts with 'window.MOCK_DATA = {' and ends with '};' or '};'

    # Simple regex to extract the JSON object
    match = re.search(r'window\.MOCK_DATA\s*=\s*(\{.*\});', content, re.DOTALL)
    if not match:
        # Try without the semicolon
        match = re.search(r'window\.MOCK_DATA\s*=\s*(\{.*\})', content, re.DOTALL)

    if not match:
        print("Error: Could not extract JSON object from JS file.")
        return

    json_str = match.group(1)

    try:
        data = json.loads(json_str)
        print("Successfully parsed JSON data.")
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        # Fallback: maybe there are comments or JS specific syntax?
        # Given the file content previously seen, it looks like standard JSON inside JS.
        return

    # Extract Subsets
    subsets = {
        'seed_reports.json': data.get('reports', []),
        'seed_credit_memos.json': data.get('credit_memos', {}),
        'seed_file_index.json': data.get('files', [])
    }

    output_dir = 'showcase/data'
    os.makedirs(output_dir, exist_ok=True)

    for filename, subset_data in subsets.items():
        output_path = os.path.join(output_dir, filename)
        print(f"Writing {len(subset_data)} items to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(subset_data, f, indent=2)

    print("Extraction complete.")

if __name__ == "__main__":
    extract_seed_data()
