import re

with open('scripts/generate_market_mayhem_archive.py', 'r') as f:
    content = f.read()

# Fix ValueError: invalid literal for int()
new_content = """    for item in sorted_items:
        year = item["date"].split("-")[0]
        try:
            year_int = int(year)
            if year_int < 2021:
                historical.append(item)
            else:
                if str(year_int) not in grouped: grouped[str(year_int)] = []
                grouped[str(year_int)].append(item)
        except ValueError:
            if "UNKNOWN" not in grouped: grouped["UNKNOWN"] = []
            grouped["UNKNOWN"].append(item)"""

old_content = """    for item in sorted_items:
        year = item["date"].split("-")[0]
        if int(year) < 2021: historical.append(item)
        else:
            if year not in grouped: grouped[year] = []
            grouped[year].append(item)"""

content = content.replace(old_content, new_content)

with open('scripts/generate_market_mayhem_archive.py', 'w') as f:
    f.write(content)
