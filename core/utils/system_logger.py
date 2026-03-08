import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

class SystemLogger:
    """
    A robust system logger designed to record and consolidate system events.

    This logger manages a JSON Lines file where each line represents a distinct event.
    It provides capabilities to log structured data and later consolidate these logs
    into a comprehensive system state representation.
    """
    def __init__(self, log_file: str = "logs/system_events.jsonl") -> None:
        """
        Initializes the SystemLogger.

        Args:
            log_file: The path to the log file. Defaults to 'logs/system_events.jsonl'.
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, tag: str, details: Dict[str, Any]) -> None:
        """
        Logs a specific event with a given tag and details.

        Args:
            tag: A string categorizing the event (e.g., 'BRANCH', 'AGENT_INTERACTION').
            details: A dictionary containing the specific details of the event.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tag": tag,
            "details": details
        }

        with self.log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        # Output to stdout if running as main script
        if __name__ == "__main__":
            print(json.dumps(entry))

    def _read_events(self) -> List[Dict[str, Any]]:
        """
        Internal helper to read and parse all valid JSON events from the log file.

        Returns:
            A list of dictionary objects representing the events.
        """
        events: List[Dict[str, Any]] = []
        if self.log_file.exists():
            with self.log_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            # Skip malformed lines to ensure resilience
                            continue
        return events

    def consolidate_logs(self) -> None:
        """
        Consolidates all logged events into a master system state file.

        This method retrieves all valid events and wraps them, along with
        simulated market data payloads, into a comprehensive JSON file representing
        the system's current state.
        """
        events = self._read_events()

        # Create mock payload structure if real data isn't provided (for demonstration of structure)
        # In a real scenario, this might load current state from other files.
        payload = {
            "market_mayhem_dec_2025.json": {"v23_knowledge_graph": {"meta": {}}},
            "market_state.json": {"metadata": {}},
            "retail_alpha.json": {},
            "system_events": events
        }

        create_timestamped_system_file(payload)

def create_timestamped_system_file(input_data: Dict[str, Any], output_filename: Optional[str] = None) -> None:
    """
    Wraps input data into a master system payload with current timestamps and saves it to a file.

    Args:
        input_data: A dictionary containing the system state data to be timestamped and saved.
        output_filename: The desired output filename. If not provided, it generates one based on the timestamp.
    """
    try:
        est_zone = ZoneInfo("America/New_York")
    except Exception:
        # Fallback to UTC if timezone data is missing or error occurs
        est_zone = timezone.utc

    current_time_est = datetime.now(est_zone)
    current_time_utc_iso = current_time_est.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Update Market Mayhem Metadata
    if "market_mayhem_dec_2025.json" in input_data:
        market_mayhem = input_data.setdefault("market_mayhem_dec_2025.json", {})
        knowledge_graph = market_mayhem.setdefault("v23_knowledge_graph", {})
        meta = knowledge_graph.setdefault("meta", {})

        meta["generated_at"] = current_time_utc_iso
        input_data["market_mayhem_current.json"] = input_data.pop("market_mayhem_dec_2025.json")

    # 2. Update Market State Metadata
    if "market_state.json" in input_data:
        market_state = input_data.setdefault("market_state.json", {})
        metadata = market_state.setdefault("metadata", {})
        metadata["generated_at"] = current_time_utc_iso

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
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding="utf-8") as f:
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
