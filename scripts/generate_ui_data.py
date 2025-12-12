import os
import json
import re
import glob

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
AGENTS_DIR = os.path.join(ROOT_DIR, 'core', 'agents')
ARCHIVES_DIR = os.path.join(ROOT_DIR, 'core', 'libraries_and_archives')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'services', 'webapp', 'client', 'public', 'data')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'manifest.json')

def scan_agents():
    agents = []
    print(f"Scanning agents in {AGENTS_DIR}...")
    for filepath in glob.glob(os.path.join(AGENTS_DIR, '**', '*.py'), recursive=True):
        if '__init__' in filepath: continue

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Regex to find class definition inheriting from AgentBase
        class_match = re.search(r'class\s+(\w+)\s*\(.*AgentBase.*\):', content)
        if class_match:
            name = class_match.group(1)

            # Try to find persona or description in __init__
            persona_match = re.search(r'self\.persona\s*=\s*self\.config\.get\([\'"]persona[\'"],\s*[\'"](.*?)[\'"]\)', content)
            desc_match = re.search(r'self\.description\s*=\s*self\.config\.get\([\'"]description[\'"],\s*[\'"](.*?)[\'"]\)', content)

            specialization = persona_match.group(1) if persona_match else "Generalist"
            description = desc_match.group(1) if desc_match else "No description available."

            agents.append({
                "id": name,
                "name": name.replace("Agent", " Agent"),
                "status": "active" if "Risk" in name or "Market" in name else "idle",
                "specialization": specialization,
                "description": description,
                "last_active": "Just now"
            })
    return agents

def scan_reports():
    reports = []
    print(f"Scanning reports in {ARCHIVES_DIR}...")
    if not os.path.exists(ARCHIVES_DIR):
        print("Archives directory not found, skipping.")
        return []

    for filepath in glob.glob(os.path.join(ARCHIVES_DIR, '**', '*.*'), recursive=True):
        filename = os.path.basename(filepath)
        if filename.endswith('.json') or filename.endswith('.md'):
            reports.append({
                "id": filename,
                "title": filename.replace('_', ' ').replace('.json', '').replace('.md', '').title(),
                "type": "Deep Dive" if "Deep_Dive" in filename else "Report",
                "date": "2023-10-27", # Placeholder
                "path": f"/data/archives/{filename}" # Hypothetical path if served
            })
    return reports

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    agents = scan_agents()
    reports = scan_reports()

    # Mock Knowledge Graph stats if file doesn't exist
    kg_path = os.path.join(DATA_DIR, 'knowledge_graph.json')
    kg_stats = {"nodes": 0, "edges": 0}
    if os.path.exists(kg_path):
        try:
            with open(kg_path, 'r') as f:
                kg = json.load(f)
                kg_stats["nodes"] = len(kg.get("nodes", []))
                kg_stats["edges"] = len(kg.get("edges", []))
        except:
            pass
    else:
        kg_stats = {"nodes": 1250, "edges": 4500} # Mock data

    manifest = {
        "system_version": "v23.5.0",
        "agents": agents,
        "reports": reports,
        "knowledge_graph_stats": kg_stats
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest generated at {OUTPUT_FILE}")
    print(f"Found {len(agents)} agents and {len(reports)} reports.")

if __name__ == "__main__":
    main()
