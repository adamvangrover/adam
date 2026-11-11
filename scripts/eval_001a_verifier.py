import json
import os
import jsonschema

def verify_jsonl_schema():
    schema_path = "showcase/data/adam_daily/schema.json"
    base_dir = "showcase/data/adam_daily"

    with open(schema_path, "r") as f:
        schema = json.load(f)

    # The JSON schema given for MarketMayhemLedger defines an object with a "data_points" array.
    # We generated JSONL (one object per line), but the schema given in the repository
    # defines a top level object.
    # To properly validate the array items, we extract the item schema.
    item_schema = schema["properties"]["data_points"]["items"]

    # ⚡ Bolt Optimization: Pre-compile jsonschema validator
    ValidatorClass = jsonschema.validators.validator_for(item_schema)
    validator = ValidatorClass(item_schema)

    all_passed = True
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path):
            data_path = os.path.join(item_path, "data.jsonl")
            if os.path.exists(data_path):
                print(f"Validating JSONL entries in {data_path}...")
                with open(data_path, "r") as f:
                    for i, line in enumerate(f):
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        try:
                            validator.validate(instance=entry)
                        except jsonschema.exceptions.ValidationError as e:
                            print(f"{data_path} - Line {i+1}: Invalid - {e.message}")
                            all_passed = False

    if all_passed:
        print("JSONL verification passed for all files.")
    return all_passed

if __name__ == "__main__":
    success = verify_jsonl_schema()
    if not success:
        exit(1)
