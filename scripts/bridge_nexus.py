#!/usr/bin/env python3
import json
import time
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine.live_mock_engine import live_engine

DATA_PATH = "showcase/data/nexus_simulation.json"
LIVE_PATH = "showcase/data/nexus_live.json"

def load_initial_state():
    if os.path.exists(DATA_PATH):
        try:
            with open(DATA_PATH, 'r') as f:
                data = json.load(f)
                return {
                    "stream": data.get("stream", [])[-50:], # Keep only last 50
                    "metrics": data.get("metrics", {"global_stability": 0.5})
                }
        except Exception as e:
            print(f"Error loading initial state: {e}")

    return {
        "stream": [],
        "metrics": {"global_stability": 0.5}
    }

def run_bridge():
    print("Starting Nexus Bridge...")
    state = load_initial_state()

    # Ensure metrics has global_stability
    if "global_stability" not in state["metrics"]:
        state["metrics"]["global_stability"] = 0.5

    while True:
        try:
            event = live_engine.get_geopolitical_event()

            # Update Stream
            state["stream"].append(event["text"])
            if len(state["stream"]) > 50:
                state["stream"] = state["stream"][-50:]

            # Update Stability
            current_stability = state["metrics"]["global_stability"]
            # Apply delta with damping factor to avoid rapid swings
            new_stability = max(0.0, min(1.0, current_stability + event["stability_delta"] * 0.05))
            state["metrics"]["global_stability"] = new_stability

            # Write Live Data
            # Use atomic write pattern if possible or just write
            with open(LIVE_PATH, 'w') as f:
                json.dump(state, f, indent=2)

            print(f"[BRIDGE] Generated: {event['text']} (Stability: {new_stability:.2f})")

            time.sleep(3)

        except KeyboardInterrupt:
            print("Bridge stopped.")
            break
        except Exception as e:
            print(f"Error in bridge loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    run_bridge()
