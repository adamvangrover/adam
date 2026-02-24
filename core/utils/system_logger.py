import json
import os
import sys
from datetime import datetime, timezone

try:
    from zoneinfo import ZoneInfo
except ImportError:
    # Fallback for environments without zoneinfo
    # Returns UTC timezone object which is a valid tzinfo
    def ZoneInfo(key):
        return timezone.utc

class SystemLogger:
    def __init__(self, log_file="logs/system_events.jsonl"):
        self.log_file = log_file
        # Ensure directory exists
        if os.path.dirname(log_file):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

    def log_event(self, tag: str, details: dict):
        """
        Log an event with a specific tag (BRANCH, PUSH, PULL, DELETION, CREATION, AGENT_INTERACTION, RUNTIME, SERVER_BUILD).
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag": tag,
            "details": details
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # If running as script, output to stdout too
        if __name__ == "__main__":
            print(json.dumps(entry))

    def consolidate_logs(self):
        """
        Consolidate logs into a single immutable system state file using the provided logic.
        """
        # Read events
        events = []
        if os.path.exists(self.log_file):
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        line = line.strip()
                        if line:
                            events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Create mock payload structure if real data isn't provided (for demonstration of structure)
        # In a real scenario, this might load current state from other files.
        payload = {
            "market_mayhem_dec_2025.json": {"v23_knowledge_graph": {"meta": {}}},
            "market_state.json": {"metadata": {}},
            "retail_alpha.json": {},
            "system_events": events
        }

        # Use provided logic
        create_timestamped_system_file(payload)

def create_timestamped_system_file(input_data: dict, output_filename: str = None):
    # Set current time (Feb 21, 2026 13:53 EST / 18:53 UTC)
    try:
        est_zone = ZoneInfo("America/New_York")
    except Exception:
        # Fallback to UTC if timezone data is missing or error occurs
        est_zone = timezone.utc

    current_time_est = datetime.now(est_zone)
    current_time_utc_iso = current_time_est.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Update Market Mayhem Metadata
    if "market_mayhem_dec_2025.json" in input_data:
        # Update the timestamp
        if "v23_knowledge_graph" not in input_data["market_mayhem_dec_2025.json"]:
             input_data["market_mayhem_dec_2025.json"]["v23_knowledge_graph"] = {}
        if "meta" not in input_data["market_mayhem_dec_2025.json"]["v23_knowledge_graph"]:
             input_data["market_mayhem_dec_2025.json"]["v23_knowledge_graph"]["meta"] = {}

        input_data["market_mayhem_dec_2025.json"]["v23_knowledge_graph"]["meta"]["generated_at"] = current_time_utc_iso

        # Rename the key to reflect the current system state
        input_data["market_mayhem_current.json"] = input_data.pop("market_mayhem_dec_2025.json")

    # 2. Update Market State Metadata
    if "market_state.json" in input_data:
        if "metadata" not in input_data["market_state.json"]:
            input_data["market_state.json"]["metadata"] = {}
        input_data["market_state.json"]["metadata"]["generated_at"] = current_time_utc_iso

    # 3. Update Retail Alpha Timestamp
    if "retail_alpha.json" in input_data:
        input_data["retail_alpha.json"]["timestamp"] = current_time_utc_iso

    # 4. Wrap everything in a master system payload
    system_payload = {
        "system_metadata": {
            "compilation_timestamp": current_time_utc_iso,
            "system_version": "Adam-v23.5-Apex",
            "environment": "Production"
        },
        "data_payload": input_data
    }

    # Generate filename based on current timestamp if not provided
    if not output_filename:
        timestamp_str = current_time_est.strftime("%Y%m%d_%H%M%S")
        output_filename = f"system_state_{timestamp_str}.json"

    # Write to disk
    with open(output_filename, 'w') as f:
        json.dump(system_payload, f, indent=2)

    print(f"Successfully generated populated system file: {output_filename}")
    print(f"Timestamp applied: {current_time_utc_iso}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        logger = SystemLogger()
        command = sys.argv[1]
        if command == "log" and len(sys.argv) > 3:
            tag = sys.argv[2]
            try:
                details = json.loads(sys.argv[3])
            except json.JSONDecodeError:
                details = {"message": sys.argv[3]}
            logger.log_event(tag, details)
        elif command == "consolidate":
            logger.consolidate_logs()
        else:
            print("Usage: python -m core.utils.system_logger [log <tag> <json_details> | consolidate]")
