"""Instruction compiler: Generates AGENTS.md and llms-full.txt from YAML schema."""
import sys
from pathlib import Path
import yaml


def build_agent_docs() -> None:
    schema_path = Path("config/agent_schema.yaml")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found at {schema_path}")

    with open(schema_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    version = cfg["version"]
    thresholds = cfg["governance_policy"]["thresholds"]
    roles = cfg["roles"]

    content = [
        f"# AFOS v{version} Agent System Instructions",
        "\n> AUTOMATICALLY GENERATED FROM `config/agent_schema.yaml`. DO NOT EDIT DIRECTLY.\n",
        "## 1. Operating Confidence Tiers",
        f"- **Autonomous (>= {thresholds['autonomous_execution']['min_confidence']}):** {thresholds['autonomous_execution']['description']}",
        f"- **HITL Required ({thresholds['human_in_the_loop']['min_confidence']} - {thresholds['human_in_the_loop']['max_confidence']}):** {thresholds['human_in_the_loop']['description']}",
        f"- **Rejection (< {thresholds['human_in_the_loop']['min_confidence']}):** {thresholds['rejection']['description']}\n",
        "## 2. Agent Roles and Contractual Constraints",
    ]

    for role_name, role_data in roles.items():
        content.append(f"### Role: {role_name} (`{role_data['id']}`)")
        content.append(f"- **Authority:** {role_data['authority']}")
        content.append("- **Constraints:**")
        for constraint in role_data["constraints"]:
            content.append(f"  - {constraint}")
        content.append("- **Allowed Tools:** " + ", ".join(role_data["allowed_tools"]))
        content.append("")

    rendered = "\n".join(content)

    Path("AGENTS.md").write_text(rendered, encoding="utf-8")
    Path("llms-full.txt").write_text(rendered, encoding="utf-8")
    print(f"Successfully compiled AGENTS.md and llms-full.txt for version {version}")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        schema = Path("config/agent_schema.yaml").read_text(encoding="utf-8")
        if "v23.5" in schema or "v26.0" in schema:
            print("Verification failed: Stale version identifiers found.", file=sys.stderr)
            sys.exit(1)
        print("Schema verified cleanly.")
    else:
        build_agent_docs()