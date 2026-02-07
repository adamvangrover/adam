import os
import json
import glob
import psutil
import random
from collections import defaultdict

SNAPSHOTS_DIR = "core/libraries_and_archives/snapshots"
AUDIT_DIR = "core/libraries_and_archives/audit_trails"
OUTPUT_FILE = "showcase/js/mock_system_brain_data.js"

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

def generate_brain_data():
    # 1. Active Agents (from Snapshots)
    active_agents = set()
    memory_usage = defaultdict(int)

    snapshot_files = glob.glob(os.path.join(SNAPSHOTS_DIR, "*.json"))
    for f in snapshot_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                agent = data.get('agent_id')
                active_agents.add(agent)
                # Estimate memory size
                mem_size = len(json.dumps(data.get('memory_state', {})))
                memory_usage[agent] += mem_size
        except:
            pass

    # 2. Activity Heatmap (from Audit Trails)
    activity_counts = defaultdict(int)
    audit_files = glob.glob(os.path.join(AUDIT_DIR, "*.json"))
    for f in audit_files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                agent = data.get('agent_id')
                activity_counts[agent] += 1
        except:
            pass

    # Structure for Frontend
    data = {
        "active_agents": list(active_agents),
        "memory_usage_bytes": memory_usage,
        "activity_metrics": activity_counts,
        "hardware_telemetry": get_hardware_metrics(),
        "system_status": "OPERATIONAL",
        "timestamp": os.path.getmtime(SNAPSHOTS_DIR) if os.path.exists(SNAPSHOTS_DIR) else 0
    }

    # Mock data if empty
    if not data["active_agents"]:
        data["active_agents"] = ["SNCAnalystAgent", "BlackSwanAgent", "DeepSectorAnalyst"]
        data["memory_usage_bytes"] = {"SNCAnalystAgent": 4500, "BlackSwanAgent": 8200}
        data["activity_metrics"] = {"SNCAnalystAgent": 12, "BlackSwanAgent": 5}

    content = f"window.SYSTEM_BRAIN_DATA = {json.dumps(data, indent=2)};"

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)

    print(f"System Brain data generated at {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_brain_data()
