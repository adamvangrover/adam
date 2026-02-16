import re
import json
import hashlib
import os
from datetime import datetime

CHANGELOG_PATH = 'CHANGELOG.md'
OUTPUT_PATH = 'showcase/data/evolution_data.json'

def parse_changelog(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by version headers (## v...)
    # Regex lookahead to split but keep delimiter part of the next chunk would be hard,
    # so we'll just split and re-add the "## " part.
    # Actually, let's go line by line or block by block.

    # Pattern to match version header: ## v26.6 - Title
    version_pattern = re.compile(r'^##\s+(v[\d\.]+.*)$', re.MULTILINE)

    sections = []
    current_section = None

    lines = content.split('\n')

    for line in lines:
        match = version_pattern.match(line)
        if match:
            if current_section:
                sections.append(current_section)
            current_section = {'header': match.group(1), 'content': []}
        elif current_section:
            current_section['content'].append(line)

    if current_section:
        sections.append(current_section)

    parsed_data = []

    for section in sections:
        header = section['header']
        # header format: v26.6 - Protocol ARCHITECT_INFINITE (Day 7)
        # or: [v23.5-patch-security] - 2025-05-21 (Operation Green Light)

        version_match = re.search(r'(v[\d\.]+(-[a-z]+)?)', header)
        version = version_match.group(1) if version_match else "Unknown"

        # Extract date if present (YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', header)
        date = date_match.group(1) if date_match else "2025-10-27" # Default future date for "Protocol" entries

        # Extract Title (everything after version and dash)
        # simplistic split
        parts = header.split(' - ', 1)
        title = parts[1].strip() if len(parts) > 1 else header

        # Parse content for Jules' Log, Added, Fixed
        content_text = '\n'.join(section['content'])

        jules_log = ""
        log_match = re.search(r"### Jules' Log\n> \"(.*?)\"", content_text, re.DOTALL)
        if log_match:
            jules_log = log_match.group(1).replace('\n', ' ').strip()

        added = re.findall(r'- \*\*New Organ\*\*: (.*)', content_text)
        added += re.findall(r'- \*\*Added\*\*: (.*)', content_text)

        # Fallback for other formats (e.g., v23.5)
        if not added:
             # Capture all bullet points that look like additions
             added += re.findall(r'- \*\*Dependency Management\*\*: (.*)', content_text)
             added += re.findall(r'- \*\*Core Schemas\*\*: (.*)', content_text)
             added += re.findall(r'- \*\*New Organ\*\*: (.*)', content_text)

        fixed = re.findall(r'- \*\*Fixed\*\*: (.*)', content_text)
        if not fixed:
             fixed += re.findall(r'- \*\*Syntax\*\*: (.*)', content_text)
             fixed += re.findall(r'- \*\*Critical:\*\* (.*)', content_text)

        # Calculate deterministic metrics
        # complexity = version number sum
        try:
            v_nums = re.findall(r'\d+', version)
            complexity = sum(int(x) for x in v_nums) * 5 + len(added) * 2
        except:
            complexity = 10

        # entropy = hash of log text length % 100
        entropy = int(hashlib.md5(jules_log.encode()).hexdigest(), 16) % 100 if jules_log else 20

        # generations = cumulative count of characters in log / 10
        generations = len(content_text) // 10

        parsed_data.append({
            "version": version,
            "date": date,
            "title": title,
            "log": jules_log,
            "added": added,
            "fixed": fixed,
            "metrics": {
                "complexity": complexity,
                "entropy": entropy,
                "generations": generations
            },
            "raw_content": content_text[:200] + "..." # Snippet
        })

    return parsed_data

def generate_json():
    if not os.path.exists(CHANGELOG_PATH):
        print(f"Error: {CHANGELOG_PATH} not found.")
        return

    data = parse_changelog(CHANGELOG_PATH)

    # Sort by version desc (it naturally is in changelog usually, but let's ensure)
    # Actually, changelog is usually newest first.

    output = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "source": "CHANGELOG.md",
            "count": len(data)
        },
        "timeline": data
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"Generated {OUTPUT_PATH} with {len(data)} entries.")

if __name__ == "__main__":
    generate_json()
