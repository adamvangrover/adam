import os
import ast
import json
import logging
import re
from pathlib import Path

# Configuration
REPO_ROOT = "."
OUTPUT_FILES = [
    "showcase/data/system_knowledge_graph.json",
    "services/v24_dashboard/public/data/system_knowledge_graph.json"
]
IGNORE_DIRS = {".git", ".venv", "node_modules", "__pycache__", "dist", "build", "env", "venv", "verification_artifacts"}
IGNORE_FILES = {".DS_Store"}
HOUSE_VIEW_DIR = "core/libraries_and_archives"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GraphGen")

def get_file_type(filename):
    if filename.endswith(".py"): return "code"
    if filename.endswith(".js"): return "code"
    if filename.endswith(".html"): return "ui"
    if filename.endswith(".md"): return "doc"
    if filename.endswith(".json"): return "data"
    if filename.endswith(".txt"): return "doc"
    return "file"

def get_group(file_type, path_parts):
    if HOUSE_VIEW_DIR in "/".join(path_parts): return "strategy"
    if "agents" in path_parts: return "agent"
    if "simulations" in path_parts: return "simulation"
    if "prompt_library" in path_parts: return "prompt"
    if "core" in path_parts and file_type == "code": return "core"
    if "showcase" in path_parts: return "ui"
    if "docs" in path_parts: return "knowledge"
    if "AGENTS.md" in path_parts: return "knowledge"
    return file_type

def parse_code_structure(filepath):
    """Parses a Python file to extract classes and functions with metadata."""
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
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
                docstring = ast.get_docstring(node)
                structure["classes"].append({
                    "name": node.name,
                    "docstring": docstring,
                    "bases": bases,
                    "lineno": node.lineno
                })
            elif isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                args = [a.arg for a in node.args.args]
                structure["functions"].append({
                    "name": node.name,
                    "docstring": docstring,
                    "args": args,
                    "lineno": node.lineno
                })

    except Exception as e:
        # logger.warning(f"Could not parse {filepath}: {e}")
        pass
    return structure

def extract_file_content_preview(filepath, file_type):
    """Reads a file and returns a preview of its content."""
    try:
        with open(filepath, 'r') as f:
            if file_type == "data" and filepath.suffix == ".json":
                try:
                    data = json.load(f)
                    # If list, take first few items
                    if isinstance(data, list):
                        preview = json.dumps(data[:2], indent=2) + "\n..."
                    elif isinstance(data, dict):
                        # Extract key fields if present
                        if "title" in data and "summary" in data:
                            return f"TITLE: {data['title']}\nSUMMARY: {data['summary']}\n\n" + json.dumps(data, indent=2)[:500]
                        preview = json.dumps(data, indent=2)[:500] + "..."
                    else:
                        preview = str(data)[:500]
                    return preview
                except:
                    f.seek(0)
                    return f.read(500)

            content = f.read(1000) # Read first 1000 chars for richer context
            return content
    except:
        return ""

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

            # Color logic
            color = None
            if group == "knowledge" and file.isupper():
                 color = "#f59e0b"
                 size += 10
            elif group == "prompt":
                 color = "#ec4899" # Pink
            elif group == "strategy":
                 color = "#ef4444" # Red for House Views
                 size = 25

            # Content preview for markdown/text/data/prompt
            content_preview = ""
            if file_type in ["doc", "data"] or group in ["prompt", "strategy", "knowledge"]:
                content_preview = extract_file_content_preview(path, file_type)

            node = {
                "id": node_id,
                "label": label,
                "group": group,
                "title": rel_path,
                "value": size,
                "path": rel_path,
                "level": "file",
                "preview": content_preview
            }
            if color: node["color"] = color
            nodes.append(node)

            file_registry.append({
                "id": node_id,
                "path": rel_path,
                "type": file_type,
                "group": group
            })

            # 1b. Parse Code Structure (if Python)
            if file_type == "code" and file.endswith(".py"):
                structure = parse_code_structure(path)

                # Add Class Nodes
                for cls_data in structure["classes"]:
                    cls_id = next_id
                    next_id += 1
                    nodes.append({
                        "id": cls_id,
                        "label": cls_data["name"],
                        "group": "class",
                        "size": 15,
                        "color": "#eab308", # Yellow
                        "level": "code",
                        "docstring": cls_data["docstring"],
                        "bases": cls_data["bases"],
                        "lineno": cls_data["lineno"]
                    })
                    # Link Class -> File
                    edges.append({"from": cls_id, "to": node_id, "color": "#555555"})

                # Add Function Nodes (only top level)
                for func_data in structure["functions"]:
                    func_id = next_id
                    next_id += 1
                    nodes.append({
                        "id": func_id,
                        "label": func_data["name"] + "()",
                        "group": "function",
                        "size": 10,
                        "color": "#3b82f6", # Blue
                        "level": "code",
                        "docstring": func_data["docstring"],
                        "args": func_data["args"],
                        "lineno": func_data["lineno"]
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
        "level": "concept",
        "preview": "Central Strategy & Vision Node. Aggregates all market reports, outlooks, and strategic archives."
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
         "level": "concept",
         "preview": "Core AI Infrastructure & Tooling"
    })
    next_id += 1

    edges.append({"from": house_view_root, "to": infra_root, "dashes": True})

    # 4. Link House View Docs and Infra
    logger.info("Linking House Views & Concepts...")
    for file_node in file_registry:
        node_id = file_node["id"]
        path = file_node["path"]
        group = file_node["group"]

        # Explicit Strategy/House View Link
        if group == "strategy":
            edges.append({"from": house_view_root, "to": node_id, "color": "#ef4444"})

        # Prompts to Infra
        elif group == "prompt":
             edges.append({"from": infra_root, "to": node_id, "color": {"opacity": 0.2}})

        # Knowledge linking
        elif group == "knowledge" and "docs/" in path:
            label = path.split("/")[-1]
            if label.isupper() or "vision" in label.lower() or "roadmap" in label.lower():
                edges.append({"from": house_view_root, "to": node_id})
            elif "architecture" in label.lower() or "guide" in label.lower():
                 edges.append({"from": infra_root, "to": node_id})

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
                "level": "concept",
                "preview": f"Agent Definition: {agent_name}"
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

    for output_file in OUTPUT_FILES:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved graph to {output_file}")

    logger.info(f"Graph generated: {len(nodes)} nodes, {len(edges)} edges.")

if __name__ == "__main__":
    generate_graph()
