import subprocess
import time
import os
import json
import signal
import sys

BRIDGE_SCRIPT = "scripts/bridge_nexus.py"
DATA_FILE = "showcase/data/nexus_live.json"

def verify_nexus_live():
    # Remove existing file if any
    if os.path.exists(DATA_FILE):
        try:
            os.remove(DATA_FILE)
        except OSError:
            pass

    print(f"Starting {BRIDGE_SCRIPT}...")
    # Use sys.executable to ensure we use the same python interpreter
    process = subprocess.Popen([sys.executable, BRIDGE_SCRIPT])

    try:
        print("Waiting for bridge to generate data...")
        # Loop sleep 3s, so wait 5-6s to ensure at least one write
        time.sleep(6)

        if not os.path.exists(DATA_FILE):
            print(f"FAILED: {DATA_FILE} was not created.")
            return False

        with open(DATA_FILE, 'r') as f:
            data = json.load(f)

        print("Data loaded successfully.")

        # Verify structure
        if "metrics" not in data or "global_stability" not in data["metrics"]:
            print("FAILED: Missing metrics.global_stability")
            return False

        if "stream" not in data or not isinstance(data["stream"], list):
            print("FAILED: Missing stream or stream is not a list")
            return False

        if len(data["stream"]) == 0:
            print("FAILED: Stream is empty")
            return False

        print(f"Verified metrics: {data['metrics']}")
        print(f"Verified latest event: {data['stream'][-1]}")
        print("SUCCESS: Nexus Live Bridge is working.")
        return True

    finally:
        print("Stopping bridge process...")
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()

if __name__ == "__main__":
    if verify_nexus_live():
        exit(0)
    else:
        exit(1)
