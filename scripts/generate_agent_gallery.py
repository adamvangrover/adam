import os
import ast
import json
import re
import sys
from pathlib import Path

# Configuration
AGENTS_DIR = "core/agents"
MCP_TOOLS_FILE = "core/mcp/tools.json"
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

def load_mcp_tools():
    """
    Loads the MCP tools definition file.
    """
    if not os.path.exists(MCP_TOOLS_FILE):
        print(f"Warning: {MCP_TOOLS_FILE} not found.")
        return []
    try:
        with open(MCP_TOOLS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading tools file: {e}")
        return []

ALL_TOOLS = load_mcp_tools()

def extract_skills_from_docstring(docstring):
    """
    Parses docstring for sections like ## Skills or ## Capabilities
    and extracts list items.
    """
    skills = []
    if not docstring:
        return skills

    lines = docstring.split('\n')
    in_skills_section = False

    for line in lines:
        stripped = line.strip()
        # Check for headers
        if stripped.startswith('#'):
            # Check if it is a skills header
            lower_header = stripped.lower()
            if "skill" in lower_header or "capability" in lower_header or "feature" in lower_header:
                in_skills_section = True
            else:
                in_skills_section = False
            continue

        if in_skills_section:
            # Check for list items
            if stripped.startswith('- ') or stripped.startswith('* '):
                skill = stripped[2:].strip()
                if skill:
                    skills.append(skill)
            # Check for numbered list
            elif re.match(r'^\d+\.', stripped):
                skill = re.sub(r'^\d+\.\s*', '', stripped).strip()
                if skill:
                    skills.append(skill)
            elif stripped == "":
                continue
            else:
                # If we hit a non-list line that isn't empty, maybe end of section?
                pass

    return skills

def map_tools_to_agent(agent_name, docstring, all_tools):
    """
    Heuristic mapping of tools to agents.
    """
    agent_tools = []
    text_corpus = (agent_name + " " + docstring).lower()

    for tool in all_tools:
        # Simple keyword matching
        keywords = tool['name'].split('_')
        score = 0
        for kw in keywords:
            if kw in text_corpus:
                score += 1

        # If significant overlap or specific matches
        if score >= 1:
            agent_tools.append({
                "name": tool['name'],
                "description": tool.get('description', '')
            })

    return agent_tools

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
        "json_config": "",
        "skills": [],
        "tools": [],
        "mcp_connection": ""
    }

    class_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # Assume the first class is the agent
            if "Agent" in node.name:
                agent_info["name"] = node.name
                docstring = ast.get_docstring(node) or "No documentation provided."
                agent_info["docstring"] = docstring

                # Infer type
                for key, val in AGENT_TYPE_MAP.items():
                    if key in node.name:
                        agent_info["type"] = val
                        break
                else:
                    agent_info["type"] = "Agent"

                class_found = True

                # Extract Skills
                agent_info["skills"] = extract_skills_from_docstring(docstring)

                # Try to find methods like 'execute' to infer workflow
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'execute':
                        agent_info["workflow"].append("1. Receive task input")
                        if ast.get_docstring(item):
                            agent_info["task_description"] = ast.get_docstring(item)
                        step = 2
                        for stmt in item.body:
                            if isinstance(stmt, ast.Assign) or isinstance(stmt, ast.Expr):
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

    # MCP
    agent_info["mcp_connection"] = f"mcp://{agent_info['name'].lower()}.adam.internal"
    agent_info["tools"] = map_tools_to_agent(agent_info["name"], agent_info["docstring"], ALL_TOOLS)

    return agent_info

def generate_js_entry(agent):
    """
    Generates a JS object string for the agent.
    """
    def js_val(v):
        return json.dumps(v)

    workflow_js = js_val(agent.get('workflow', []))
    context_js = js_val(agent.get('context_awareness', []))
    plugins_js = js_val(agent.get('environment_plugin', []))
    skills_js = js_val(agent.get('skills', []))
    tools_js = js_val(agent.get('tools', []))

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
        portable_prompt: {js_val(agent.get('portable_prompt'))},
        skills: {skills_js},
        tools: {tools_js},
        mcp_connection: {js_val(agent.get('mcp_connection'))}
    }}"""

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

def main():
    print("Scanning for agents...")

    # We will regenerate the full file to ensure everything is consistent
    new_agents = []

    for root, _, files in os.walk(AGENTS_DIR):
        for file in files:
            if file.endswith(".py") and file != "__init__.py" and "test" not in file:
                filepath = os.path.join(root, file)
                agent_data = parse_agent_file(filepath)

                if agent_data:
                    print(f"Processed: {agent_data['name']}")
                    new_agents.append(agent_data)

    if not new_agents:
        print("No agents found.")
        return

    new_content = generate_full_file(new_agents)

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"Updated {OUTPUT_FILE} with {len(new_agents)} agents.")

if __name__ == "__main__":
    main()
