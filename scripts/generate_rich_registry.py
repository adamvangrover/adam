import os
import ast
import json
import glob
import re

# --- Configuration ---
ROOT_DIR = os.path.abspath(".")
AGENTS_DIR = os.path.join(ROOT_DIR, "core/agents")
PROMPTS_DIR = os.path.join(ROOT_DIR, "prompt_library")
OUTPUT_JS_AGENTS = os.path.join(ROOT_DIR, "showcase/js/mock_agents_rich.js")
OUTPUT_JS_PROMPTS = os.path.join(ROOT_DIR, "showcase/js/mock_prompts.js")

# --- Agent Extraction (AST) ---

def extract_agent_metadata(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()

        tree = ast.parse(source)
        filename = os.path.basename(filepath)

        agents = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it looks like an agent
                name = node.name
                docstring = ast.get_docstring(node) or "No documentation provided."

                # Heuristic: Name ends in Agent or inherits from AgentBase
                is_agent = name.endswith("Agent")
                base_classes = [b.id for b in node.bases if isinstance(b, ast.Name)]
                if "AgentBase" in base_classes or is_agent:

                    # Extract Methods
                    methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef) and not n.name.startswith('_')]

                    # Determine Type
                    agent_type = "Specialized"
                    if "Orchestrator" in name: agent_type = "Orchestrator"
                    elif "Meta" in name: agent_type = "Meta-Agent"
                    elif "Sub" in name: agent_type = "Sub-Agent"

                    agents.append({
                        "name": name,
                        "type": agent_type,
                        "status": "ACTIVE", # Mock status
                        "docstring": docstring.strip(),
                        "file": f"core/agents/{filename}",
                        "methods": methods,
                        "context_awareness": ["Standard Context"], # Placeholder
                        "system_prompt_persona": f"You are {name}, a specialized component of the Adam Financial System.",
                        "task_description": f"Execute logic defined in {filename}."
                    })
        return agents
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        return []

def scan_agents():
    print(f"Scanning Agents in {AGENTS_DIR}...")
    all_agents = []

    # Python Files
    py_files = glob.glob(os.path.join(AGENTS_DIR, "**/*.py"), recursive=True)
    for f in py_files:
        if "__init__" in f: continue
        all_agents.extend(extract_agent_metadata(f))

    return all_agents

# --- Prompt Extraction ---

def scan_prompts():
    print(f"Scanning Prompts in {PROMPTS_DIR}...")
    all_prompts = []

    # Markdown and JSON
    files = glob.glob(os.path.join(PROMPTS_DIR, "**/*.*"), recursive=True)

    for f in files:
        if os.path.isdir(f): continue
        ext = os.path.splitext(f)[1].lower()
        if ext not in ['.md', '.json', '.yaml', '.yml']: continue

        try:
            with open(f, 'r', encoding='utf-8') as file:
                content = file.read()

            rel_path = os.path.relpath(f, PROMPTS_DIR)
            name = os.path.basename(f)

            # Determine Category from folder structure
            category = os.path.dirname(rel_path).replace(os.sep, ' ').title() or "General"

            # Determine Type
            p_type = "MARKET_TEMPLATE"
            if ext == '.json': p_type = "JSON_SCHEMA"
            if "system" in name.lower(): p_type = "SYSTEM_PROMPT"

            all_prompts.append({
                "name": name,
                "path": f"prompt_library/{rel_path}",
                "category": category,
                "type": p_type,
                "content": content
            })
        except Exception as e:
            print(f"Error reading prompt {f}: {e}")

    return all_prompts

# --- Main ---

def main():
    # 1. Agents
    agents = scan_agents()
    print(f"Found {len(agents)} agents.")

    # Inject into MOCK_DATA structure if needed, but here we update mock_agents_rich.js
    # We need to make sure the global MOCK_DATA.agents is populated by this file in agents.html
    # agents.html uses `window.MOCK_DATA.agents` primarily if mock_agents_rich.js puts it there.
    # Looking at `agents.html` logic:
    # if (window.MOCK_DATA && window.MOCK_DATA.agents) { allAgents = window.MOCK_DATA.agents; }

    # So we should write to window.MOCK_DATA.agents in the JS file.

    js_content_agents = f"""
if (!window.MOCK_DATA) window.MOCK_DATA = {{}};
window.MOCK_DATA.agents = {json.dumps(agents, indent=2)};
console.log("Loaded {len(agents)} rich agents.");
"""
    with open(OUTPUT_JS_AGENTS, "w", encoding="utf-8") as f:
        f.write(js_content_agents)
    print(f"Written {OUTPUT_JS_AGENTS}")

    # 2. Prompts
    prompts = scan_prompts()
    print(f"Found {len(prompts)} prompts.")

    js_content_prompts = f"""
if (!window.MOCK_DATA) window.MOCK_DATA = {{}};
window.MOCK_DATA.prompts = {json.dumps(prompts, indent=2)};
console.log("Loaded {len(prompts)} prompts.");
"""
    with open(OUTPUT_JS_PROMPTS, "w", encoding="utf-8") as f:
        f.write(js_content_prompts)
    print(f"Written {OUTPUT_JS_PROMPTS}")

if __name__ == "__main__":
    main()
