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

    # Regex to match version headers: ## v26.6... or ## [v23.5-patch-security]...
    # We want to capture the whole line after ##
    version_pattern = re.compile(r'^##\s+(.*)$', re.MULTILINE)

    sections = []
    current_section = None

    lines = content.split('\n')

    for line in lines:
        match = version_pattern.match(line)
        if match:
            if current_section:
                sections.append(current_section)
            current_section = {'header': match.group(1).strip(), 'content': []}
        elif current_section:
            current_section['content'].append(line)

    if current_section:
        sections.append(current_section)

    parsed_data = []

    for section in sections:
        header = section['header']
        # header formats:
        # v26.6 - Protocol ARCHITECT_INFINITE (Day 7)
        # [v23.5-patch-security] - 2025-05-21 (Operation Green Light)

        # Clean header for parsing
        clean_header = header.replace('[', '').replace(']', '')

        # Extract version
        version_match = re.search(r'(v[\d\.]+(-[a-z0-9]+)?)', clean_header)
        version = version_match.group(1) if version_match else "Unknown"

        # Extract date if present (YYYY-MM-DD)
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', clean_header)
        # Default dates based on version context if missing
        if date_match:
            date = date_match.group(1)
        elif "v26" in version:
            date = "2025-10-27"
        elif "v23" in version:
            date = "2025-05-20"
        else:
            date = "2025-01-01"

        # Extract Title (everything after version/date/dash)
        # Remove version and date from header to get title
        title = clean_header
        if version != "Unknown":
            title = title.replace(version, '')
        if date_match:
            title = title.replace(date_match.group(1), '')

        # Clean up leading/trailing dashes and spaces
        title = re.sub(r'^[\s\-]+', '', title)
        title = re.sub(r'[\s\-]+$', '', title)

        # If title is empty or just parens, fix it
        if not title:
            title = "System Update"

        # Parse content
        content_text = '\n'.join(section['content'])

        # Extract Jules' Log
        jules_log = ""
        log_match = re.search(r"### Jules' Log\n> \"(.*?)\"", content_text, re.DOTALL)
        if log_match:
            jules_log = log_match.group(1).replace('\n', ' ').strip()
        else:
            # Fallback: look for a quote block near top
            quote_match = re.search(r'> "(.*?)"', content_text, re.DOTALL)
            if quote_match:
                jules_log = quote_match.group(1).replace('\n', ' ').strip()

        # Extract "Added" items
        added = []
        # Look for explicit "Added" or "New Organ" lines
        added += re.findall(r'- \*\*New Organ\*\*: (.*)', content_text)
        added += re.findall(r'- \*\*Added\*\*: (.*)', content_text)

        # If section has "### Added" or "### Architecture", grab list items
        added_section = re.search(r'### (Added|Architecture)(.*?)(###|$)', content_text, re.DOTALL)
        if added_section:
            items = re.findall(r'- (.*?)$', added_section.group(2), re.MULTILINE)
            # Filter out lines that are just headers or empty
            items = [item.strip() for item in items if item.strip() and not item.startswith('**New Organ**:') and not item.startswith('**Added**:')]
            added.extend(items)

        # Extract "Fixed" items
        fixed = []
        fixed += re.findall(r'- \*\*Fixed\*\*: (.*)', content_text)
        fixed_section = re.search(r'### (Fixed|Security|Reliability)(.*?)(###|$)', content_text, re.DOTALL)
        if fixed_section:
            items = re.findall(r'- (.*?)$', fixed_section.group(2), re.MULTILINE)
            items = [item.strip() for item in items if item.strip() and not item.startswith('**Fixed**:')]
            fixed.extend(items)

        # Calculate metrics
        try:
            v_nums = re.findall(r'\d+', version)
            base_complexity = sum(int(x) for x in v_nums) * 5
            complexity = base_complexity + (len(added) * 3) + (len(fixed) * 1)
        except:
            complexity = 10 + len(added)

        # Using sha256 instead of md5 for bandit compliance, though not security critical here
        entropy = int(hashlib.sha256((title + str(len(content_text))).encode()).hexdigest(), 16) % 100
        generations = len(content_text) // 10

        parsed_data.append({
            "version": version,
            "date": date,
            "title": title,
            "log": jules_log,
            "added": list(set(added)), # Dedupe
            "fixed": list(set(fixed)),
            "metrics": {
                "complexity": complexity,
                "entropy": entropy,
                "generations": generations
            },
            "raw_content": content_text[:200] + "..."
        })

    return parsed_data

def generate_json():
    if not os.path.exists(CHANGELOG_PATH):
        print(f"Error: {CHANGELOG_PATH} not found.")
        return

    data = parse_changelog(CHANGELOG_PATH)

    # Sort chronologically for the timeline (oldest first for graphing usually, but let's keep array ordered)
    # Actually, changelog is reverse chronological (newest top).
    # Let's keep it as is, frontend can reverse if needed.

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
