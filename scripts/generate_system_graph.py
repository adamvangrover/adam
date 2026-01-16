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

class CodeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.classes = []
        self.functions = []

    def visit_ClassDef(self, node):
        self.classes.append(node.name)
        # We don't recurse into classes to find methods *as global functions*,
        # but we could extract methods here if we wanted deeper granularity.
        # For now, let's keep it to top-level classes.
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Only capture top-level functions or we'd get methods too if we visited classes
        # But generic_visit goes deep.
        # Let's simple capture all defs for now, knowing some are methods.
        self.functions.append(node.name)
        self.generic_visit(node)

def parse_code_structure(filepath):
    """Parses a Python file to extract classes and functions."""
    structure = {"classes": [], "functions": [], "imports": []}
    try:
        with open(filepath, "r") as f:
            content = f.read()
            tree = ast.parse(content)

        # Extract Imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    structure["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    structure["imports"].append(node.module)

        # Extract Classes and Functions
        # We prefer top-level only to avoid cluttering the graph with every helper method
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                structure["classes"].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                structure["functions"].append(node.name)

    except Exception as e:
        # logger.warning(f"Could not parse {filepath}: {e}")
        pass
    return structure

def extract_agents_from_md(filepath):
    """Extracts agent names from AGENTS.md or similar files."""
    agents = []
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Look for headers like "### Meta-Agents" or "**Role:**"
            matches = re.findall(r'### (.*?) Agent', content, re.IGNORECASE)
            for m in matches:
                name = m.strip() + " Agent"
                agents.append(name)

            # Also look for explicit definitions
            matches_v2 = re.findall(r'\* \*\*(.*?):\*\* \(.*?\)', content)
            for m in matches_v2:
                agents.append(m.strip())
    except Exception as e:
        logger.warning(f"Could not parse agents from {filepath}: {e}")
    return list(set(agents))

def generate_graph():
    nodes = []
    edges = []

    node_id_map = {} # path or unique_name -> id
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

            # Create File Node
            node_id = next_id
            next_id += 1
            node_id_map[rel_path] = node_id

            label = file
            size = 10 + (os.path.getsize(path) / 1000)
            if size > 40: size = 40
            if group == "knowledge" and file.isupper():
                 color = "#f59e0b"
                 size += 10
            else:
                color = None

            node = {
                "id": node_id,
                "label": label,
                "group": group,
                "title": rel_path,
                "value": size,
                "path": rel_path,
                "level": "file"
            }
            if color: node["color"] = color
            nodes.append(node)

            file_registry.append({
                "id": node_id,
                "path": rel_path,
                "type": file_type
            })

            # 1b. Parse Code Structure (if Python)
            if file_type == "code" and file.endswith(".py"):
                structure = parse_code_structure(path)

                # Add Class Nodes
                for cls_name in structure["classes"]:
                    cls_id = next_id
                    next_id += 1
                    nodes.append({
                        "id": cls_id,
                        "label": cls_name,
                        "group": "class",
                        "size": 15,
                        "color": "#eab308", # Yellow
                        "level": "code"
                    })
                    # Link Class -> File
                    edges.append({"from": cls_id, "to": node_id, "color": "#555555"})

                # Add Function Nodes (only top level)
                for func_name in structure["functions"]:
                    func_id = next_id
                    next_id += 1
                    nodes.append({
                        "id": func_id,
                        "label": func_name + "()",
                        "group": "function",
                        "size": 10,
                        "color": "#3b82f6", # Blue
                        "level": "code"
                    })
                    # Link Function -> File
                    edges.append({"from": func_id, "to": node_id, "color": "#555555"})

                # Store imports for later linking
                file_registry[-1]["imports"] = structure["imports"]

    # 2. Analyze Relationships (Imports)
    logger.info("Analyzing relationships...")

    for file_info in file_registry:
        if file_info.get("imports"):
            for imp in file_info["imports"]:
                # Naive matching
                expected_path_part = imp.replace(".", "/")
                for candidate in file_registry:
                    if expected_path_part in candidate["path"]:
                         if candidate["path"].endswith(".py"):
                             edges.append({
                                 "from": file_info["id"],
                                 "to": candidate["id"],
                                 "arrows": "to",
                                 "color": {"opacity": 0.3}
                             })
                             break

    # 3. Add explicit "House View" / Concept Nodes
    house_view_root = next_id
    nodes.append({
        "id": house_view_root,
        "label": "JPM HOUSE VIEW",
        "group": "strategy",
        "size": 60,
        "color": "#ef4444",
        "font": {"size": 20, "face": "JetBrains Mono"},
        "level": "concept"
    })
    next_id += 1

    infra_root = next_id
    nodes.append({
        "id": infra_root,
        "label": "JPM AI INFRASTRUCTURE",
        "group": "core",
        "size": 60,
        "color": "#a855f7",
         "font": {"size": 20, "face": "JetBrains Mono"},
         "level": "concept"
    })
    next_id += 1

    edges.append({"from": house_view_root, "to": infra_root, "dashes": True})

    # 4. Scan for House View Docs
    logger.info("Extracting House Views...")
    for node in nodes:
        if node.get("level") == "file" and node["group"] == "knowledge" and "docs/" in node["path"]:
            if node["label"].isupper() or "vision" in node["label"] or "roadmap" in node["label"]:
                edges.append({"from": house_view_root, "to": node["id"]})
            elif "architecture" in node["label"] or "guide" in node["label"]:
                 edges.append({"from": infra_root, "to": node["id"]})

    # 5. Extract Abstract Agent Concepts
    logger.info("Extracting Agents from Markdown...")
    if "AGENTS.md" in node_id_map:
        agents_md_id = node_id_map["AGENTS.md"]
        agent_names = extract_agents_from_md("AGENTS.md")

        for agent_name in agent_names:
            if len(agent_name) < 3: continue

            agent_id = next_id
            next_id += 1
            nodes.append({
                "id": agent_id,
                "label": agent_name,
                "group": "agent",
                "shape": "diamond",
                "size": 25,
                "color": "#10b981",
                "level": "concept"
            })

            edges.append({"from": agents_md_id, "to": agent_id, "dashes": True})

            normalized_name = agent_name.lower().replace(" ", "_").replace("agent", "")
            for f in file_registry:
                if normalized_name in f["path"].lower():
                     edges.append({"from": agent_id, "to": f["id"], "color": "#10b981"})

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
