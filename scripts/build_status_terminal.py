import json
import os
import glob
import shutil
from datetime import datetime, timezone


def is_valuation_report(data):
    """Check if the json data looks like a valuation report/credit memo."""
    if not isinstance(data, dict):
        return False
    required_keys = ['borrower_name', 'report_date']
    if not all(k in data for k in required_keys):
        return False
    # At least some of these
    valuation_keys = ['risk_score', 'historical_financials', 'pd_model', 'dcf_analysis']
    if not any(k in data for k in valuation_keys):
        return False
    return True


def process_files():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_dir = os.path.join(base_dir, 'showcase', 'status_terminal')
    before_dir = os.path.join(target_dir, 'archive', 'before')
    after_dir = os.path.join(target_dir, 'archive', 'after')
    jsonl_path = os.path.join(target_dir, 'valuation_history.jsonl')

    os.makedirs(before_dir, exist_ok=True)
    os.makedirs(after_dir, exist_ok=True)

    json_files = glob.glob(os.path.join(base_dir, '**', '*.json'), recursive=True)

    # Filter out node_modules, git, and the target dir itself
    json_files = [f for f in json_files if 'node_modules' not in f and '.git' not in f and 'status_terminal' not in f]

    processed_count = 0

    with open(jsonl_path, 'w', encoding='utf-8') as jsonl_out:
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception:
                continue

            if not is_valuation_report(data):
                continue

            # 1. Copy original to archive/before
            filename = os.path.basename(file_path)
            safe_filename = f"{filename.split('.')[0]}_{int(datetime.now(timezone.utc).timestamp())}.json"
            shutil.copy2(file_path, os.path.join(before_dir, safe_filename))

            # 2. Synthesize new comprehensive system status
            # For mock generation, we update the timestamp, and ensure some fields are present
            timestamp_iso = datetime.now(timezone.utc).isoformat()

            # Enhance data
            data['system_timestamp'] = timestamp_iso

            # Extract simple valuation/risk metrics to top level if they don't exist
            if 'risk_score' not in data:
                data['risk_score'] = 'N/A'
            if 'pd_model' in data and 'model_score' in data['pd_model']:
                data['pd_score'] = data['pd_model']['model_score']

            # 3. Save to archive/after
            with open(os.path.join(after_dir, safe_filename), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

            # 4. Append to JSONL
            # We dump the entire object so the terminal can show everything
            jsonl_out.write(json.dumps(data) + '\n')
            processed_count += 1

    print(f"Successfully processed {processed_count} valuation reports.")
    print(f"Output written to: {jsonl_path}")


if __name__ == '__main__':
    process_files()
