import os
import ast
import json
import re
import sys
from pathlib import Path

# Configuration
AGENTS_DIR = "core/agents"
OUTPUT_FILE = "showcase/js/mock_agents_rich.js"
EXISTING_FILE = "showcase/js/mock_agents_rich.js"

# Metadata mappings
AGENT_TYPE_MAP = {
    "Analyst": "Analysis Agent",
    "Specialist": "Specialized Agent",
    "Orchestrator": "Orchestrator",
    "Risk": "Risk Agent",
    "Trading": "Execution Agent",
    "Bot": "Intelligence Agent",
    "Compliance": "Governance Agent"
}

def load_existing_data(filepath):
    """
    Attempts to load the existing JS file and parse the agents list.
    Since it's loose JS, we use some regex heuristics to convert to JSON.
    """
    if not os.path.exists(filepath):
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract the array content: window.MOCK_DATA.agents = [...];
        match = re.search(r'window\.MOCK_DATA\.agents\s*=\s*(\[.*\]);', content, re.DOTALL)
        if not match:
            # Try finding just the list if the assignment varies
            match = re.search(r'(\[\s*\{.*\}\s*\])', content, re.DOTALL)

        if match:
            json_str = match.group(1)
            # lax parsing:
            # 1. quote keys: name: -> "name":
            # Avoid replacing inside strings.
            # This is hard with regex.
            # Alternative: Use Python AST literal_eval if we transform syntax?

            # Transform to Python syntax
            py_str = json_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            # Quote keys?
            # Let's rely on the fact that we might not need to parse it perfectly to merge.
            # Actually, to merge, we need the names.

            # Fallback: Simple text scan for names to avoid over-engineering JS parser
            names = re.findall(r'name:\s*"([^"]+)"', content)

            # If we want to preserve the FULL object, we might just append NEW objects to the array string
            # rather than parsing/serializing the whole thing.
            return content, set(names)

    except Exception as e:
        print(f"Error reading existing file: {e}")

    return "", set()

def parse_agent_file(filepath):
    """
    Parses a python agent file to extract metadata.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    agent_info = {
        "file": filepath,
        "workflow": [],
        "context_awareness": ["Standard Context"],
        "environment_plugin": [],
        "system_prompt_persona": "Standard Agent Persona",
        "task_description": "Execute defined tasks.",
        "reasoning_process": "Standard logic execution.",
        "markdown_copy": "",
        "yaml_config": "",
        "json_config": ""
    }

    class_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Assume the first class is the agent
            if "Agent" in node.name:
                agent_info["name"] = node.name
                agent_info["docstring"] = ast.get_docstring(node) or "No documentation provided."

                # Infer type
                for key, val in AGENT_TYPE_MAP.items():
                    if key in node.name:
                        agent_info["type"] = val
                        break
                else:
                    agent_info["type"] = "Agent"

                class_found = True

                # Try to find methods like 'execute' to infer workflow
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'execute':
                        agent_info["workflow"].append("1. Receive task input")
                        if ast.get_docstring(item):
                            agent_info["task_description"] = ast.get_docstring(item)
                        # Analyze body for calls
                        step = 2
                        for stmt in item.body:
                            if isinstance(stmt, ast.Assign) or isinstance(stmt, ast.Expr):
                                # Crude step extraction
                                if hasattr(stmt, 'value') and isinstance(stmt.value, ast.Call):
                                    func_name = "unknown_func"
                                    if isinstance(stmt.value.func, ast.Name):
                                        func_name = stmt.value.func.id
                                    elif isinstance(stmt.value.func, ast.Attribute):
                                        func_name = stmt.value.func.attr
                                    agent_info["workflow"].append(f"{step}. Execute {func_name}")
                                    step += 1
                break

    if not class_found:
        return None

    # Generate Mock Data for missing rich fields
    agent_info["markdown_copy"] = f"# {agent_info['name']}\n\n## Overview\n{agent_info['docstring']}\n"
    agent_info["portable_prompt"] = f"### SYSTEM PROMPT: {agent_info['name']} ###\n\n# MISSION\n{agent_info['docstring']}"

    # Generate Configs
    agent_info["yaml_config"] = f"agent:\n  name: {agent_info['name']}\n  enabled: true"
    agent_info["json_config"] = json.dumps({"agent": agent_info['name'], "enabled": True}, indent=2)

    return agent_info

def generate_js_entry(agent):
    """
    Generates a JS object string for the agent.
    """
    # Helper to dump json and strip outer braces to embed in the object
    def js_val(v):
        return json.dumps(v)

    workflow_js = js_val(agent.get('workflow', []))
    context_js = js_val(agent.get('context_awareness', []))
    plugins_js = js_val(agent.get('environment_plugin', []))

    return f"""    {{
        name: {js_val(agent.get('name'))},
        type: {js_val(agent.get('type'))},
        status: "ACTIVE",
        docstring: {js_val(agent.get('docstring'))},
        file: {js_val(agent.get('file'))},
        system_prompt_persona: {js_val(agent.get('system_prompt_persona'))},
        task_description: {js_val(agent.get('task_description'))},
        workflow: {workflow_js},
        reasoning_process: {js_val(agent.get('reasoning_process'))},
        context_awareness: {context_js},
        environment_plugin: {plugins_js},
        yaml_config: {js_val(agent.get('yaml_config'))},
        json_config: {js_val(agent.get('json_config'))},
        markdown_copy: {js_val(agent.get('markdown_copy'))},
        portable_prompt: {js_val(agent.get('portable_prompt'))}
    }}"""

def main():
    print("Scanning for agents...")

    # Load existing content to preserve it
    existing_content, existing_names = load_existing_data(EXISTING_FILE)

    new_agents = []

    for root, _, files in os.walk(AGENTS_DIR):
        for file in files:
            if file.endswith(".py") and file != "__init__.py" and "test" not in file:
                filepath = os.path.join(root, file)
                agent_data = parse_agent_file(filepath)

                if agent_data:
                    # Check if already exists (fuzzy match name)
                    # The existing file uses "Risk_Assessment_Agent" (underscores) vs "RiskAssessmentAgent" (CamelCase)
                    # We need to normalize for check.
                    name = agent_data["name"]

                    # Heuristic: Convert CamelCase to Underscore for comparison if needed,
                    # but the existing file seems to mix them or use Underscores for display names.
                    # Let's check both.

                    is_duplicate = False
                    for existing in existing_names:
                        if name == existing or name.replace("_", "") == existing.replace("_", ""):
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        print(f"Found new agent: {name}")
                        new_agents.append(agent_data)
                    else:
                        print(f"Skipping existing agent: {name}")

    if not new_agents:
        print("No new agents found to add.")
        return

    # If we have existing content, we insert into the array
    if existing_content:
        # Find the closing bracket of the array
        last_bracket = existing_content.rfind('];')
        if last_bracket == -1:
            last_bracket = existing_content.rfind(']')

        if last_bracket != -1:
            # Prepare new entries
            entries_str = ",\n".join([generate_js_entry(a) for a in new_agents])

            # Insert before the closing bracket
            # Ensure we have a comma if the list wasn't empty
            insertion_point = last_bracket
            prefix = ""
            if "{" in existing_content[0:last_bracket]: # simplistic check if array is not empty
                prefix = ",\n"

            new_content = existing_content[:insertion_point] + prefix + entries_str + existing_content[insertion_point:]

            # Fix helper function append if it got cut off or something?
            # The regex grab might have been just the array.
            # Wait, load_existing_data returns the FULL content.
            pass
        else:
            # Fallback regen
            print("Could not find array end in existing file. Regenerating.")
            new_content = generate_full_file(new_agents)
    else:
        new_content = generate_full_file(new_agents)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Updated {OUTPUT_FILE} with {len(new_agents)} new agents.")

def generate_full_file(agents):
    entries = ",\n".join([generate_js_entry(a) for a in agents])
    return f"""if (!window.MOCK_DATA) window.MOCK_DATA = {{}};

window.MOCK_DATA.agents = [
{entries}
];

// Helper to load by ID
window.MOCK_DATA.getAgent = function(name) {{
    return window.MOCK_DATA.agents.find(a => a.name === name);
}};
"""

if __name__ == "__main__":
    main()
