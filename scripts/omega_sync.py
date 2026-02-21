import os
import shutil
import json
import time

def sync_logs():
    """
    Syncs backend logs to frontend data directory for Project OMEGA visualization.
    """
    log_dir = "logs"
    data_dir = "showcase/data"

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Files to sync
    files = {
        "proof_of_thought_ledger.json": "proof_of_thought_ledger.json",
        "dream_journal.json": "dream_journal.json"
    }

    for src, dst in files.items():
        src_path = os.path.join(log_dir, src)
        dst_path = os.path.join(data_dir, dst)

        if os.path.exists(src_path):
            try:
                shutil.copy(src_path, dst_path)
                print(f"Synced {src} to {dst}")
            except Exception as e:
                print(f"Error syncing {src}: {e}")
        else:
            print(f"Source {src} not found. Generating mock data...")
            generate_mock_data(dst_path, src)

def generate_mock_data(filepath, type_key):
    if "proof" in type_key:
        data = [
            {
                "index": 0,
                "timestamp": time.time(),
                "agent": "SYSTEM",
                "thought": "GENESIS_BLOCK",
                "previous_hash": "0"*64,
                "hash": "a1b2c3d4e5..."
            },
            {
                "index": 1,
                "timestamp": time.time() + 1,
                "agent": "RiskAgent",
                "thought": "{\"analysis\": \"VIX Spike detected\", \"decision\": \"Hedge\"}",
                "previous_hash": "a1b2c3d4e5...",
                "hash": "f9e8d7c6b5..."
            }
        ]
    elif "dream" in type_key:
        data = [
            {
                "timestamp": time.time(),
                "scenario": {"name": "Mock Nightmare", "type": "CYBER"},
                "solution": {"steps": ["Step 1", "Step 2"], "outcome": "SURVIVED"}
            }
        ]
    else:
        data = {}

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Generated mock data at {filepath}")

if __name__ == "__main__":
    sync_logs()
