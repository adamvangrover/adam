import os
import json
import ast

EXCLUDE_DIRS = {
    '.git', '__pycache__', 'node_modules', 'venv', '.idea', '.vscode',
    'dist', 'build', 'coverage', '.pytest_cache', 'site-packages', 'ui_archive_v1',
    'webapp', 'experimental' # Excluding some large/irrelevant folders to keep graph clean
}
EXCLUDE_FILES = {
    '.DS_Store', 'package-lock.json', 'yarn.lock', 'repo_data.js'
}

def get_file_info(filepath):
    try:
        size = os.path.getsize(filepath)
        line_count = 0
        imports = []

        # Simple line count and import detection for text files
        if filepath.endswith(('.py', '.js', '.html', '.css', '.md', '.json', '.txt', '.sh')):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                line_count = len(content.splitlines())

                if filepath.endswith('.py'):
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, ast.Import):
                                for n in node.names:
                                    imports.append(n.name)
                            elif isinstance(node, ast.ImportFrom):
                                if node.module:
                                    imports.append(node.module)
                    except:
                        pass # Ignore parse errors

        return {
            "size": size,
            "lines": line_count,
            "imports": imports
        }
    except Exception as e:
        return {"size": 0, "lines": 0, "imports": []}

def scan_repo(root_dir):
    nodes = []
    edges = []
    tree = {}

    # Map path to ID for graph
    path_to_id = {}
    node_counter = 0

    # Initialize Root
    path_to_id[""] = 0
    nodes.append({
        "id": 0,
        "label": "ADAM_CORE",
        "group": "root",
        "path": "",
        "value": 20
    })

    for root, dirs, files in os.walk(root_dir):
        # Filter excluded dirs in place
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        rel_root = os.path.relpath(root, root_dir)
        if rel_root == ".":
            rel_root = ""

        # Add folder node if not root (root already added)
        if rel_root != "":
             if rel_root not in path_to_id:
                 node_counter += 1
                 path_to_id[rel_root] = node_counter
                 nodes.append({
                     "id": node_counter,
                     "label": os.path.basename(rel_root),
                     "group": "folder",
                     "path": rel_root,
                     "value": 10
                 })

             # Edge from parent
             parent = os.path.dirname(rel_root)
             # If parent is empty string, it's root (id 0)
             parent_id = path_to_id.get(parent, 0)
             current_id = path_to_id[rel_root]

             # Check if edge already exists to avoid dupes (though simple logic handles tree)
             edges.append({"from": parent_id, "to": current_id, "color": "#334155"})

        # Process files
        for file in files:
            if file in EXCLUDE_FILES: continue

            full_path = os.path.join(root, file)
            rel_path = os.path.join(rel_root, file)

            info = get_file_info(full_path)

            node_counter += 1
            path_to_id[rel_path] = node_counter

            file_type = file.split('.')[-1] if '.' in file else 'txt'

            # Map types to groups for coloring
            group = file_type

            nodes.append({
                "id": node_counter,
                "label": file,
                "group": group,
                "path": rel_path,
                "size": info['size'],
                "lines": info['lines'],
                "imports": info['imports'],
                "value": 5
            })

            # Edge from folder
            folder_id = path_to_id.get(rel_root, 0)
            edges.append({"from": folder_id, "to": node_counter, "color": "#1e293b"})

            # Add to Tree Structure (nested dict)
            parts = rel_path.split(os.sep)
            current = tree
            for part in parts[:-1]:
                current = current.setdefault(part, {})
            current[parts[-1]] = {
                "__file__": True,
                "type": "file",
                "size": info['size'],
                "lines": info['lines']
            }

    # Second pass: dependency edges (simplified)
    # Create a map of module_path -> node_id
    module_map = {}
    for node in nodes:
        if str(node.get('group')) == 'py':
            # e.g. core/agents/agent_base.py -> core.agents.agent_base
            path_key = node['path'].replace('.py', '').replace(os.sep, '.')
            module_map[path_key] = node['id']

    for node in nodes:
        if 'imports' in node and node['imports']:
            for imp in node['imports']:
                # Try exact match
                if imp in module_map:
                    edges.append({
                        "from": node['id'],
                        "to": module_map[imp],
                        "color": "#38bdf8",
                        "opacity": 0.3,
                        "dashes": True,
                        "smooth": {"type": "curvedCW", "roundness": 0.2}
                    })
                # Check for relative imports or package matches? (Simplified for now)

    return {"nodes": nodes, "edges": edges, "tree": tree}

if __name__ == "__main__":
    print("Scanning repository...")
    data = scan_repo(".")

    js_content = f"window.REPO_DATA = {json.dumps(data, indent=2)};"

    output_path = "showcase/js/repo_data.js"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(js_content)

    print(f"Generated {output_path} with {len(data['nodes'])} nodes and {len(data['edges'])} edges.")
