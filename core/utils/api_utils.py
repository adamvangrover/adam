#core/utils/api_utils.py

import json

def get_knowledge_graph_data(module, concept):
    """
    Retrieves data from the knowledge graph for the specified module and concept.
    """
    try:
        with open("data/knowledge_graph.json", "r") as f:
            knowledge_graph = json.load(f)
        return knowledge_graph[module][concept]
    except KeyError:
        return {"error": "Module or concept not found in the knowledge graph."}


def update_knowledge_graph_node(module, concept, node_id, new_value):
    """
    Updates a specific node in the knowledge graph with a new value.
    """
    try:
        with open("data/knowledge_graph.json", "r") as f:
            knowledge_graph = json.load(f)
        knowledge_graph[module][concept]["machine_readable"]["nodes"][node_id] = new_value
        with open("data/knowledge_graph.json", "w") as f:
            json.dump(knowledge_graph, f, indent=4)
        return {"status": "success"}
    except KeyError:
        return {"error": "Module, concept, or node not found in the knowledge graph."}


def validate_api_request(request_data, required_parameters):
    """
    Validates API request data against a list of required parameters.
    """
    for param in required_parameters:
        if param not in request_data:
            raise ValueError(f"Missing required parameter: {param}")

# Add more API utility functions as needed
