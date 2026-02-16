import os
import json
import ast
import glob

def parse_python_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read())
        except Exception as e:
            print(f"Error parsing {filepath}: {e}")
            return []

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Check if it looks like an agent
            is_agent = 'Agent' in node.name or 'Analyst' in node.name
            docstring = ast.get_docstring(node)
            classes.append({
                "name": node.name,
                "type": "Agent" if is_agent else "Class",
                "docstring": docstring,
                "file": filepath,
                "lineno": node.lineno
            })
    return classes

def scan_directory(root_dir):
    repo_data = {
        "agents": [],
        "verifications": [],
        "pages": []
    }

    # Scan for Agents
    agent_files = glob.glob(os.path.join(root_dir, "core/agents/**/*.py"), recursive=True)
    for f in agent_files:
        if "__init__" in f: continue
        classes = parse_python_file(f)
        for c in classes:
            if c['type'] == "Agent":
                repo_data['agents'].append(c)

    # Scan for Verifications
    verify_files = glob.glob(os.path.join(root_dir, "verification/verify_*.py"))
    for f in verify_files:
        repo_data['verifications'].append({
            "name": os.path.basename(f),
            "path": f,
            "target": os.path.basename(f).replace("verify_", "").replace(".py", "")
        })

    # Scan for HTML Pages
    html_files = glob.glob(os.path.join(root_dir, "showcase/*.html"))
    for f in html_files:
        repo_data['pages'].append({
            "name": os.path.basename(f),
            "path": f
        })

    return repo_data

def link_data(repo_data):
    # Link Agents to Verification Scripts based on loose naming matching
    for agent in repo_data['agents']:
        agent['verification_script'] = None
        normalized_name = agent['name'].lower().replace("agent", "").replace("_", "")

        for v in repo_data['verifications']:
            normalized_target = v['target'].lower().replace("_", "")
            if normalized_target in normalized_name or normalized_name in normalized_target:
                agent['verification_script'] = v['path']
                break
    return repo_data

if __name__ == "__main__":
    print("Scanning repository...")
    data = scan_directory(".")
    data = link_data(data)

    output_path = "showcase/data/repo_metadata.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Metadata generated at {output_path}")
    print(f"Found {len(data['agents'])} agents, {len(data['verifications'])} verification scripts, {len(data['pages'])} pages.")
