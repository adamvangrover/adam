import ast
import json
import hashlib
import os

def generate_ast_hash(node):
    source = ast.unparse(node)
    return f"sha256:{hashlib.sha256(source.encode('utf-8')).hexdigest()[:16]}"

def parse_agent_file(filepath):
    with open(filepath, 'r') as f:
        source_code = f.read()

    try:
        tree = ast.parse(source_code)
    except SyntaxError:
        return None

    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            # check if it looks like an Agent
            if 'Agent' in node.name or any(
                (isinstance(base, ast.Name) and 'Agent' in base.id)
                for base in node.bases
            ):
                classes.append(node)

    if not classes:
        return None

    class_node = classes[0]
    agent_class_name = class_node.name

    docstring = ast.get_docstring(class_node)

    # Extract capabilities
    capabilities = []
    if docstring:
        lines = docstring.split('\n')
        in_capabilities = False
        for line in lines:
            line = line.strip()
            if line.startswith('Core Capabilities:'):
                in_capabilities = True
                continue

            if in_capabilities:
                if line.startswith('- '):
                    capabilities.append(line[2:])
                elif not line:
                    continue
                else:
                    in_capabilities = False

    # Extract core logic signatures
    core_logic_signatures = []
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            source = ast.get_source_segment(source_code, node)
            loc = len(source.splitlines())

            signature = {
                "name": node.name,
                "ast_hash": generate_ast_hash(node),
                "loc": loc
            }

            if loc > 30: # Arbitrary threshold
                signature["source_omitted_for_brevity"] = True
                signature["note"] = "In a real telemetry pipeline, large functions can be truncated or omitted entirely in favor of the 'ast_hash' to save network bandwidth and storage costs."
            else:
                signature["source"] = source

            core_logic_signatures.append(signature)

    return {
        "agent_id": agent_class_name,
        "extracted_capabilities": capabilities,
        "core_logic_signatures": core_logic_signatures
    }

def get_agents_list():
    agents_dir = "core/agents"
    agent_files = []
    for root, _, files in os.walk(agents_dir):
        for file in files:
            if file.endswith("_agent.py"):
                agent_files.append(os.path.join(root, file))
    return agent_files

def build_telemetry_jsonl():
    agents_files = get_agents_list()

    os.makedirs("logs/swarm_telemetry_expanded", exist_ok=True)

    for filepath in agents_files:
        try:
            parsed_data = parse_agent_file(filepath)
        except Exception as e:
            continue

        if not parsed_data:
            continue

        # Match template details
        payload = {
            "spec_version": "1.4.0-otel",
            "timestamp": "2026-04-24T00:13:03.355457Z",
            "trace_id": "4bf92f3577b34da6a3ce929d0e0e4736",
            "span_id": "00f067aa0ba902b7",
            "severity_text": "WARN",
            "severity_number": 13,
            "event_name": "swarm.agent.lifecycle.deprecated",
            "resource": {
                "service.name": "swarm-orchestrator",
                "service.version": "4.2.0",
                "deployment.environment": "production",
                "node.id": "worker-node-aws-us-east-1a-04"
            },
            "attributes": {
                "agent.id": parsed_data["agent_id"],
                "agent.previous_version": "v29.x",
                "agent.target_version": "v30",
                "deprecation.reason": "Transitioning to v30 architecture.",
                "deprecation.action_required": True
            },
            "metrics": {
                "agent.core_logic.function_count": len(parsed_data["core_logic_signatures"]),
                "agent.capabilities.count": len(parsed_data["extracted_capabilities"])
            },
            "artifacts": {
                "provenance": {
                    "repository": f"git+ssh://repo/{filepath}",
                    "commit_sha": "a1b2c3d4e5f6g7h8i9j0",
                    "docstring_hash": "sha256:d8c4b12389a..."
                },
                "extracted_capabilities": parsed_data["extracted_capabilities"],
                "core_logic_signatures": parsed_data["core_logic_signatures"]
            }
        }

        # Save to individual files following the template format
        out_filename = f"logs/swarm_telemetry_expanded/human_template_2026-04-24_{parsed_data['agent_id']}.json"

        with open(out_filename, 'w') as f:
            json.dump(payload, f, indent=2)

if __name__ == "__main__":
    build_telemetry_jsonl()
