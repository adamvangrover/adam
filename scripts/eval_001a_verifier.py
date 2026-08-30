import json
import os
import jsonschema

def verify_jsonl_schema():
    schema_path = "showcase/data/adam_daily/schema.json"
    data_path = "showcase/data/adam_daily/2026-08-29/data.jsonl"

    with open(schema_path, "r") as f:
        schema = json.load(f)

    # The JSON schema given for MarketMayhemLedger defines an object with a "data_points" array.
    # We generated JSONL (one object per line), but the schema given in the repository
    # defines a top level object.
    # To properly validate the array items, we extract the item schema.
    item_schema = schema["properties"]["data_points"]["items"]

    print("Validating JSONL entries...")
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            entry = json.loads(line)
            try:
                jsonschema.validate(instance=entry, schema=item_schema)
                print(f"Line {i+1}: Valid")
            except jsonschema.exceptions.ValidationError as e:
                print(f"Line {i+1}: Invalid - {e.message}")
                return False

    print("JSONL verification passed.")
    return True

if __name__ == "__main__":
    success = verify_jsonl_schema()
    if not success:
        exit(1)
