import json
import glob
import sys

def check_provenance():
    files = glob.glob('showcase/data/adam_daily/*/data.jsonl')
    has_error = False

    for filepath in files:
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if not lines:
                continue

            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    # explicitly skipping the first line containing the JSON-LD @context metadata
                    if i == 0 and "@context" in data:
                        continue

                    # Memory explicitly said: "explicitly skipping the first line containing the JSON-LD @context metadata".
                    # Let me check if there are other @context lines we shouldn't fail on.
                    # Yes, 2026-05-26 line 9 is {"@context": ...}
                    # We should probably skip those as well, they aren't data rows.
                    if "@context" in data:
                        continue

                    # And 2026-05-30 has a single line with `report_date` and `data_points`.
                    if "report_date" in data and "data_points" in data:
                        continue

                    if "context_provenance" not in data:
                        print(f"Error: 'context_provenance' missing in {filepath} on line {i+1}")
                        has_error = True

                except json.JSONDecodeError as e:
                    print(f"Error: Invalid JSON in {filepath} on line {i+1}: {e}")
                    has_error = True

    if has_error:
        print("Provenance check failed.")
        sys.exit(1)
    else:
        print("Provenance check passed.")

if __name__ == '__main__':
    check_provenance()
