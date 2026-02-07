import os
import sys
import inspect
import importlib
import glob
from typing import List, Type

# Add repo root to path
sys.path.append(os.getcwd())

from core.agents.agent_base import AgentBase

AGENTS_DIR = "core/agents"
OUTPUT_FILE = "docs/agent_skills.md"

def get_agent_classes() -> List[Type[AgentBase]]:
    agents = []

    # Walk through the directory
    for root, dirs, files in os.walk(AGENTS_DIR):
        for file in files:
            if file.endswith("_agent.py") and not file.startswith("__"):
                module_path = os.path.join(root, file)
                module_name = module_path.replace("/", ".").replace(".py", "")

                try:
                    module = importlib.import_module(module_name)
                    # Find classes inheriting from AgentBase
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and issubclass(obj, AgentBase) and obj is not AgentBase:
                            agents.append(obj)
                except Exception as e:
                    print(f"Skipping {module_name}: {e}")

    return list(set(agents)) # Deduplicate

def generate_markdown(agents: List[Type[AgentBase]]) -> str:
    md = "# Agent Skills Registry\n\n"
    md += "Auto-generated documentation of agent capabilities and tool schemas.\n\n"

    for agent_cls in sorted(agents, key=lambda x: x.__name__):
        try:
            # Instantiate to access instance methods if needed, or use class method if static
            # Assuming get_skill_schema might be static or instance
            # In base it raises NotImplemented, let's try to instantiate
            agent_instance = agent_cls(config={"name": "doc_gen"})
            schema = agent_instance.get_skill_schema()

            md += f"## {schema.get('name', agent_cls.__name__)}\n"
            md += f"{schema.get('description', 'No description provided.')}\n\n"

            skills = schema.get('skills', [])
            if skills:
                md += "### Skills\n"
                for skill in skills:
                    md += f"#### `{skill['name']}`\n"
                    md += f"- **Description**: {skill['description']}\n"

                    params = skill.get('parameters', {})
                    if params:
                        md += "- **Parameters**:\n"
                        props = params.get('properties', {})
                        required = params.get('required', [])

                        for prop_name, prop_details in props.items():
                            req_str = "*" if prop_name in required else ""
                            md += f"  - `{prop_name}{req_str}` ({prop_details.get('type')}): {prop_details.get('description')}\n"
            else:
                md += "*No explicit skills exposed via MCP.*\n"

            md += "\n---\n"

        except Exception as e:
             # Some agents might fail instantiation without specific config
             # We skip them for now
             print(f"Could not document {agent_cls.__name__}: {e}")

    return md

def main():
    print("Scanning for agents...")
    agents = get_agent_classes()
    print(f"Found {len(agents)} agents.")

    md_content = generate_markdown(agents)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(md_content)

    print(f"Documentation generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
