import json

def generate_static_data():
    try:
        with open('showcase/data/strategic_command.json', 'r') as f:
            strategic = json.load(f)
    except FileNotFoundError:
        strategic = None

    try:
        with open('showcase/data/sp500_market_data.json', 'r') as f:
            market = json.load(f)
    except FileNotFoundError:
        market = None

    try:
        with open('showcase/data/market_mayhem_index.json', 'r') as f:
            archive = json.load(f)
    except FileNotFoundError:
        archive = None

    data = {
        "strategic": strategic,
        "market": market,
        "archive": archive
    }

    js_content = f"window.MARKET_MAYHEM_DATA = {json.dumps(data)};"

    with open('showcase/js/market_mayhem_data_static.js', 'w') as f:
        f.write(js_content)

    print("Successfully generated showcase/js/market_mayhem_data_static.js")

if __name__ == '__main__':
    generate_static_data()