import os
import json
import ast
import glob
import re

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
            # We broaden the search to include Analysts, Managers, Engineers, etc.
            is_agent = any(k in node.name for k in ['Agent', 'Analyst', 'Manager', 'Engineer', 'Architect', 'Orchestrator'])
            docstring = ast.get_docstring(node)

            methods = []
            parent_classes = []

            # Get base classes
            for base in node.bases:
                if isinstance(base, ast.Name):
                    parent_classes.append(base.id)
                elif isinstance(base, ast.Attribute):
                    parent_classes.append(base.attr)

            # Get public methods
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    if not item.name.startswith('_'):
                        methods.append(item.name)

            classes.append({
                "name": node.name,
                "type": "Agent" if is_agent else "Class",
                "docstring": docstring,
                "file": filepath,
                "lineno": node.lineno,
                "methods": methods,
                "parents": parent_classes
            })
    return classes

def scan_directory(root_dir):
    repo_data = {
        "agents": [],
        "verifications": [],
        "pages": []
    }

    # Scan for Agents - Recursive in core/
    print(f"Scanning {os.path.join(root_dir, 'core/**/*.py')}...")
    agent_files = glob.glob(os.path.join(root_dir, "core/**/*.py"), recursive=True)

    for f in agent_files:
        if "__init__" in f: continue
        if "test" in f: continue

        classes = parse_python_file(f)
        for c in classes:
            if c['type'] == "Agent":
                c['file'] = os.path.relpath(f, root_dir)
                repo_data['agents'].append(c)

    # Scan for Verifications
    print(f"Scanning {os.path.join(root_dir, 'verification/verify_*.py')}...")
    verify_files = glob.glob(os.path.join(root_dir, "verification/verify_*.py"))
    for f in verify_files:
        repo_data['verifications'].append({
            "name": os.path.basename(f),
            "path": os.path.relpath(f, root_dir),
            "target": os.path.basename(f).replace("verify_", "").replace(".py", "")
        })

    # Scan for HTML Pages
    print(f"Scanning {os.path.join(root_dir, 'showcase/*.html')}...")
    html_files = glob.glob(os.path.join(root_dir, "showcase/*.html"))
    for f in html_files:
        repo_data['pages'].append({
            "name": os.path.basename(f),
            "path": os.path.relpath(f, root_dir)
        })

    return repo_data

def link_data(repo_data):
    print("Linking data...")
    # Link Agents to Verification Scripts
    for agent in repo_data['agents']:
        agent['verification_script'] = None
        # Normalize: remove "Agent", "Analyst", underscores, lowercase
        normalized_name = agent['name'].lower().replace("agent", "").replace("analyst", "").replace("_", "")

        for v in repo_data['verifications']:
            normalized_target = v['target'].lower().replace("_", "")
            # Fuzzy match: verification target inside agent name or vice versa
            if normalized_target in normalized_name or normalized_name in normalized_target:
                agent['verification_script'] = v['path']
                break

        # Link Agents to Pages
        agent['linked_pages'] = []
        for page in repo_data['pages']:
            try:
                with open(page['path'], 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Check for exact class name match first
                    if agent['name'] in content:
                        agent['linked_pages'].append(page['path'])
            except Exception as e:
                # print(f"Error reading {page['path']}: {e}")
                pass

    return repo_data

if __name__ == "__main__":
    print("Starting repo introspection...")
    data = scan_directory(".")
    data = link_data(data)

    output_path = "showcase/data/repo_metadata.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

    print(f"Metadata generated at {output_path}")
    print(f"Found {len(data['agents'])} agents, {len(data['verifications'])} verification scripts, {len(data['pages'])} pages.")
