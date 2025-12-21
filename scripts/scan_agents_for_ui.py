import ast
import json
import os
import re
import time

# Configuration
CORE_AGENTS_DIR = "core/agents"
MOCK_DATA_PATH = "showcase/js/mock_data.js"

def scan_agents():
    agents = []
    agent_id_counter = 1

    for root, dirs, files in os.walk(CORE_AGENTS_DIR):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        tree = ast.parse(content)

                        for node in ast.walk(tree):
                            if isinstance(node, ast.ClassDef):
                                # Heuristic: Class name ends with 'Agent' or inherits from known bases
                                is_agent = False
                                if node.name.endswith("Agent"):
                                    is_agent = True
                                else:
                                    for base in node.bases:
                                        if isinstance(base, ast.Name) and base.id in ["AgentBase", "AsyncAgentBase", "MetaAgent", "SubAgent"]:
                                            is_agent = True
                                        elif isinstance(base, ast.Attribute) and base.attr in ["AgentBase", "AsyncAgentBase"]:
                                            is_agent = True

                                if is_agent:
                                    # Extract Docstring
                                    docstring = ast.get_docstring(node) or "No description available."

                                    # Infer Type
                                    agent_type = "Specialist"
                                    lower_name = node.name.lower()
                                    if "orchestrator" in lower_name:
                                        agent_type = "Orchestrator"
                                    elif "meta" in lower_name:
                                        agent_type = "Meta-Agent"
                                    elif "sub" in lower_name or "scout" in lower_name:
                                        agent_type = "Sub-Agent"
                                    elif "red_team" in lower_name or "adversary" in lower_name:
                                        agent_type = "Adversarial"

                                    # Extract System Prompt (heuristic search in class body)
                                    system_prompt = "System prompt not explicitly defined in class."
                                    for body_item in node.body:
                                        if isinstance(body_item, ast.Assign):
                                            for target in body_item.targets:
                                                if isinstance(target, ast.Name) and target.id in ["SYSTEM_PROMPT", "PROMPT", "DEFAULT_PROMPT"]:
                                                    if isinstance(body_item.value, ast.Constant): # Python 3.8+
                                                        system_prompt = body_item.value.value
                                                    elif isinstance(body_item.value, ast.Str): # Python < 3.8
                                                        system_prompt = body_item.value.s

                                    # Create Agent Object
                                    agents.append({
                                        "id": f"A{agent_id_counter:03d}",
                                        "name": node.name,
                                        "file_path": filepath,
                                        "type": agent_type,
                                        "status": "IDLE", # Default
                                        "load": 0, # Default
                                        "description": docstring.strip(),
                                        "prompt": system_prompt,
                                        "capabilities": [n.name for n in node.body if isinstance(n, ast.FunctionDef) and not n.name.startswith("_")]
                                    })
                                    agent_id_counter += 1
                except Exception as e:
                    print(f"Error parsing {filepath}: {e}")
    return agents

def update_mock_data(agents):
    try:
        with open(MOCK_DATA_PATH, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract the JSON object part (window.MOCK_DATA = {...})
        match = re.search(r"window\.MOCK_DATA\s*=\s*({.*});", content, re.DOTALL)
        if match:
            json_str = match.group(1)
            data = json.loads(json_str)

            # Update agents
            print(f"Found {len(agents)} agents in codebase. Updating mock_data...")
            data["agents"] = agents
            data["generated_at"] = time.time()

            # Reconstruct file content
            new_content = f"window.MOCK_DATA = {json.dumps(data, indent=2)};"

            with open(MOCK_DATA_PATH, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("Successfully updated showcase/js/mock_data.js")
        else:
            print("Could not find window.MOCK_DATA in file.")

    except Exception as e:
        print(f"Error updating mock data: {e}")

if __name__ == "__main__":
    found_agents = scan_agents()
    update_mock_data(found_agents)
