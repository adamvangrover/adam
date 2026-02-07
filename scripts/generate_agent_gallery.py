import json
import re
import os

DOCS_FILE = "docs/agent_skills.md"
OUTPUT_FILE = "showcase/data/agent_gallery_data.json"

def parse_agent_docs():
    if not os.path.exists(DOCS_FILE):
        return []

    with open(DOCS_FILE, 'r') as f:
        content = f.read()

    agents = []
    # Regex to capture sections
    # ## AgentName
    # Description
    # ### Skills
    # #### skill_name

    # Use Regex to split by Top-Level Headers (## AgentName)
    # This avoids splitting on ### Skills or #### SkillName
    agent_blocks = re.split(r'\n## (?=[A-Z])', content)[1:]

    for block in agent_blocks:
        lines = block.strip().split("\n")
        name = lines[0].strip()

        # Parse Body
        body = "\n".join(lines[1:])

        # Split Description from Skills
        parts = body.split("### Skills")
        description_text = parts[0].strip()

        skills = []
        if len(parts) > 1:
            skill_text = parts[1]
            # Parse individual skills (#### `skill_name`)
            skill_blocks = skill_text.split("#### ")
            for sb in skill_blocks:
                if not sb.strip(): continue

                sb_lines = sb.strip().split("\n")
                skill_name = sb_lines[0].replace("`", "").strip()

                # Find description
                skill_desc = "No description."
                for line in sb_lines:
                    if "- **Description**:" in line:
                        skill_desc = line.split("**Description**:", 1)[1].strip()
                        break

                skills.append({"name": skill_name, "description": skill_desc})

        # Categorize (Heuristic)
        category = "Specialist"
        if "Architect" in name or "Meta" in name:
            category = "Meta-Agent"
        elif "Guardian" in name or "Compliance" in name:
            category = "Governance"
        elif "Risk" in name:
            category = "Risk Engine"

        agents.append({
            "name": name,
            "description": description_text,
            "skills": skills,
            "category": category
        })

    return agents

def main():
    agents = parse_agent_docs()

    # Mock data if empty (fallback)
    if not agents:
        agents = [
            {"name": "RepoGuardianAgent", "description": "CI/CD Gatekeeper.", "category": "Governance", "skills": [{"name": "review_pr", "description": "Reviews code."}]},
            {"name": "DidacticArchitectAgent", "description": "Generates tutorials.", "category": "Meta-Agent", "skills": [{"name": "generate_tutorial", "description": "Creates guides."}]}
        ]

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(agents, f, indent=2)

    print(f"Gallery data generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
