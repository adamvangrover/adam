import json
import os
import glob
from typing import Dict, List, Any

AUDIT_DIR = "core/libraries_and_archives/audit_trails"
OUTPUT_FILE = "showcase/js/mock_knowledge_graph_data.js"

def load_audit_logs(directory: str) -> List[Dict[str, Any]]:
    logs = []
    pattern = os.path.join(directory, "*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    logs.extend(data)
                else:
                    logs.append(data)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return logs

def generate_graph_data(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    nodes = []
    links = []

    node_ids = set()

    def add_node(id, group, label=None):
        if id not in node_ids:
            nodes.append({"id": id, "group": group, "label": label or id})
            node_ids.add(id)

    def add_link(source, target, value=1, type="related"):
        if source in node_ids and target in node_ids:
            links.append({"source": source, "target": target, "value": value, "type": type})

    for log in logs:
        # Agent Node
        agent = log.get('agent_id', log.get('agent'))
        if agent:
            add_node(agent, 1, label=agent) # Group 1 = Agents

        # Entity Nodes (Deal, Borrower, etc.)
        details = log.get('entity_context', log.get('details', {}))

        # Try to find meaningful entities
        # Assuming typical fields like 'borrower_name', 'deal_id'
        deal_id = details.get('deal_id') or log.get('record_id')
        if deal_id:
            add_node(deal_id, 2, label=f"Deal: {deal_id}")
            if agent:
                add_link(agent, deal_id, type="analyzed")

        borrower = details.get('borrower_name')
        if borrower:
            add_node(borrower, 3, label=borrower)
            if deal_id:
                add_link(deal_id, borrower, type="involves")

        # Connect outputs to deals
        outcome = log.get('outcome', {})
        if isinstance(outcome, dict):
            # If outcome has a rating, maybe make a node for the rating?
            rating = outcome.get('proposed_rating')
            if rating:
                rating_id = f"Rating_{rating}"
                add_node(rating_id, 4, label=f"Rating: {rating}")
                if deal_id:
                    add_link(deal_id, rating_id, type="assigned")

    # Add dummy data if empty (for showcase)
    if not nodes:
        add_node("SNCAnalystAgent", 1)
        add_node("RiskAssessmentAgent", 1)
        add_node("BlackSwanAgent", 1)
        add_node("Deal_Alpha", 2)
        add_node("Deal_Beta", 2)
        add_node("Acme Corp", 3)
        add_node("Globex Inc", 3)

        add_link("SNCAnalystAgent", "Deal_Alpha")
        add_link("RiskAssessmentAgent", "Deal_Alpha")
        add_link("BlackSwanAgent", "Deal_Beta")
        add_link("Deal_Alpha", "Acme Corp")
        add_link("Deal_Beta", "Globex Inc")

    return {"nodes": nodes, "links": links}

def main():
    logs = load_audit_logs(AUDIT_DIR)
    graph_data = generate_graph_data(logs)

    content = f"window.KNOWLEDGE_GRAPH_DATA = {json.dumps(graph_data, indent=2)};"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)

    print(f"Generated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
