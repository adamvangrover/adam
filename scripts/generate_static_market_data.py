import json
import os

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    base_dir = "showcase/data"
    output_file = "showcase/js/market_mayhem_data_static.js"

    files_to_load = {
        "MARKET_MAYHEM_DATA": os.path.join(base_dir, "market_mayhem_index.json"),
        "STRATEGIC_COMMAND_DATA": os.path.join(base_dir, "strategic_command.json"),
        "SP500_MARKET_DATA": os.path.join(base_dir, "sp500_market_data.json")
    }

    js_content = "// AUTO-GENERATED FILE. DO NOT EDIT.\n// Fallback static data for serverless environments.\n\n"

    for var_name, filepath in files_to_load.items():
        data = load_json(filepath)
        # default to empty object/array if not found
        if data is None:
            data = [] if var_name != "STRATEGIC_COMMAND_DATA" else {}

        js_content += f"window.{var_name} = {json.dumps(data, indent=2)};\n\n"

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(js_content)

    print(f"Successfully generated {output_file}")

if __name__ == "__main__":
    main()
