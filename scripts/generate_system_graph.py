import os
import ast
import json
import logging
import re
from pathlib import Path

# Configuration
REPO_ROOT = "."
OUTPUT_FILE = "showcase/data/system_knowledge_graph.json"
IGNORE_DIRS = {".git", ".venv", "node_modules", "__pycache__", "dist", "build", "env", "venv"}
IGNORE_FILES = {".DS_Store"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphGen")

def get_file_type(filename):
    if filename.endswith(".py"): return "code"
    if filename.endswith(".js"): return "code"
    if filename.endswith(".html"): return "ui"
    if filename.endswith(".md"): return "doc"
    if filename.endswith(".json"): return "data"
    return "file"

def get_group(file_type, path_parts):
    if "agents" in path_parts: return "agent"
    if "simulations" in path_parts: return "simulation"
    if "core" in path_parts and file_type == "code": return "core"
    if "showcase" in path_parts: return "ui"
    if "docs" in path_parts: return "knowledge"
    return file_type

def parse_imports(filepath):
    """Parses a Python file to extract imported modules."""
    imports = []
    try:
        with open(filepath, "r") as f:
            tree = ast.parse(f.read())

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
    except Exception as e:
        # logger.warning(f"Could not parse {filepath}: {e}")
        pass
    return imports

def generate_graph():
    nodes = []
    edges = []

    node_id_map = {} # path -> id
    next_id = 1

    # 1. Scan Files
    logger.info("Scanning repository...")

    file_registry = []

    for root, dirs, files in os.walk(REPO_ROOT):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        for file in files:
            if file in IGNORE_FILES: continue

            path = Path(os.path.join(root, file))
            rel_path = str(path.relative_to(REPO_ROOT))

            if rel_path.startswith("showcase/data/"): continue # Skip generated data

            file_type = get_file_type(file)
            path_parts = rel_path.split(os.sep)
            group = get_group(file_type, path_parts)

            # Create Node
            node_id = next_id
            next_id += 1
            node_id_map[rel_path] = node_id

            # Simplify label
            label = file

            # Value/Size based on file size (logarithmicish)
            size = 10 + (os.path.getsize(path) / 1000)
            if size > 40: size = 40

            nodes.append({
                "id": node_id,
                "label": label,
                "group": group,
                "title": rel_path, # Tooltip
                "value": size,
                "path": rel_path
            })

            file_registry.append({
                "id": node_id,
                "path": rel_path,
                "type": file_type
            })

    # 2. Analyze Relationships (Imports)
    logger.info("Analyzing relationships...")

    for file_info in file_registry:
        if file_info["type"] == "code" and file_info["path"].endswith(".py"):
            imports = parse_imports(file_info["path"])

            for imp in imports:
                # Try to map import to a file in our registry
                # e.g. "core.agents.snc_analyst_agent" -> "core/agents/snc_analyst_agent.py"

                # Naive matching
                expected_path_part = imp.replace(".", "/")

                # Find matching nodes
                for candidate in file_registry:
                    if expected_path_part in candidate["path"]:
                         # Check if it's likely the right one (not perfect)
                         if candidate["path"].endswith(".py"):
                             edges.append({
                                 "from": file_info["id"],
                                 "to": candidate["id"],
                                 "arrows": "to",
                                 "color": {"opacity": 0.3}
                             })
                             break

    # 3. Add explicit "House View" / Concept Nodes (from docs)
    # Scan AGENTS.md or similar for high-level concepts

    logger.info("Extracting House Views...")
    house_view_root = next_id
    nodes.append({
        "id": house_view_root,
        "label": "JPM HOUSE VIEW",
        "group": "strategy",
        "size": 50,
        "color": "#ef4444"
    })
    next_id += 1

    # Fake/Extracted Strategy Nodes linked to docs
    strategies = [
        ("Gold Standard Pipeline", "docs/GOLD_STANDARD_PIPELINE.md"),
        ("Adaptive Hive Mind", "docs/v23_architecture_vision.md"),
        ("Risk Topography", "showcase/unified_banking.html")
    ]

    for name, link in strategies:
        strat_id = next_id
        next_id += 1
        nodes.append({
            "id": strat_id,
            "label": name,
            "group": "strategy",
            "title": "Strategic Pillar",
            "size": 30
        })
        edges.append({"from": house_view_root, "to": strat_id})

        # Link to file if exists
        if link in node_id_map:
            edges.append({"from": strat_id, "to": node_id_map[link], "dashes": True})

    output = {
        "nodes": nodes,
        "edges": edges
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Graph generated: {len(nodes)} nodes, {len(edges)} edges. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_graph()
