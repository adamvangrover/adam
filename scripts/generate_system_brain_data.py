import os
import json
import psutil
import random
from collections import defaultdict, Counter

GRAPH_FILE = "showcase/data/system_knowledge_graph.json"
OUTPUT_FILE = "showcase/js/system_brain_data.js"

def get_hardware_metrics():
    """
    Captures real or simulated hardware telemetry.
    """
    # Real CPU/Memory
    cpu_percent = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()

    # Simulated GPU/TPU (since we are in a container without them)
    # We simulate load based on CPU load for correlation
    gpu_load = min(100, max(0, cpu_percent * 1.2 + random.uniform(-10, 10)))
    tpu_load = min(100, max(0, cpu_percent * 0.8 + random.uniform(-5, 5)))

    return {
        "cpu_usage": cpu_percent,
        "memory_usage_pct": mem.percent,
        "memory_available_gb": round(mem.available / (1024**3), 2),
        "gpu_usage": round(gpu_load, 1),
        "tpu_usage": round(tpu_load, 1),
        "gpu_memory_used_gb": round(16 * (gpu_load/100), 2), # Assuming 16GB VRAM
        "tpu_memory_used_gb": round(32 * (tpu_load/100), 2)  # Assuming 32GB TPU RAM
    }

def analyze_graph():
    if not os.path.exists(GRAPH_FILE):
        return {
            "node_count": 0,
            "edge_count": 0,
            "agents": [],
            "knowledge_nodes": 0,
            "top_synapses": []
        }

    with open(GRAPH_FILE, 'r') as f:
        graph = json.load(f)

    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    # 1. Active Agents
    agents = [n['label'] for n in nodes if n.get('group') == 'agent']

    # 2. Knowledge Base Size
    knowledge_nodes = sum(1 for n in nodes if n.get('group') in ['knowledge', 'doc', 'strategy'])

    # 3. Connection Density (Synapses)
    connection_counts = Counter()
    for e in edges:
        connection_counts[e['from']] += 1
        connection_counts[e['to']] += 1

    # Map IDs back to labels/titles
    node_map = {n['id']: n for n in nodes}

    top_synapses = []
    for node_id, count in connection_counts.most_common(10):
        node = node_map.get(node_id)
        if node:
            top_synapses.append({
                "name": node.get('label', 'Unknown'),
                "type": node.get('group', 'unknown'),
                "connections": count
            })

    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "agents": agents,
        "knowledge_nodes": knowledge_nodes,
        "top_synapses": top_synapses
    }

def generate_brain_data():
    graph_metrics = analyze_graph()
    hw_metrics = get_hardware_metrics()

    # Structure for Frontend
    data = {
        "metrics": {
            "active_agents_count": len(graph_metrics['agents']),
            "active_agents_list": graph_metrics['agents'],
            "knowledge_nodes_count": graph_metrics['knowledge_nodes'],
            "knowledge_coverage": round((graph_metrics['knowledge_nodes'] / max(1, graph_metrics['node_count'])) * 100, 1),
            "total_nodes": graph_metrics['node_count'],
            "total_edges": graph_metrics['edge_count'],
            "cognitive_load": round((graph_metrics['edge_count'] / max(1, graph_metrics['node_count'])) * 10, 2) # Mock calc
        },
        "synapses": graph_metrics['top_synapses'],
        "hardware": hw_metrics,
        "system_status": "OPERATIONAL",
        "timestamp": os.path.getmtime(GRAPH_FILE) if os.path.exists(GRAPH_FILE) else 0
    }

    content = f"window.SYSTEM_BRAIN_DATA = {json.dumps(data, indent=2)};"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)

    print(f"System Brain data generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_brain_data()
