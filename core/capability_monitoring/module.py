# core/capability_monitoring/module.py

import json
import uuid
from collections import defaultdict
from datetime import datetime

# Configuration thresholds (would be externalized in a real system)
CONFIG = {
    "task_failure_threshold": 5,
    "manual_intervention_threshold": 3,
    "time_window_hours": 24
}

class CapabilityMonitoringModule:
    """
    Monitors the performance and interactions of agents to self-diagnose analytical gaps.
    This is the initial codebase as per the Phase 1 deliverable for Adam v20.0.
    """
    def __init__(self, event_bus, agent_forge_trigger):
        """
        Initializes the CMM.

        Args:
            event_bus: A mock or real event bus to subscribe to system events.
            agent_forge_trigger: A function to call to trigger the Agent Forge.
        """
        self.event_bus = event_bus
        self.agent_forge_trigger = agent_forge_trigger
        self.event_log = defaultdict(list)
        print("CapabilityMonitoringModule initialized.")

    def subscribe_to_events(self):
        """Subscribes to relevant events from the event bus."""
        self.event_bus.subscribe("task_failed", self.handle_event)
        self.event_bus.subscribe("manual_intervention_required", self.handle_event)
        self.event_bus.subscribe("data_processing_error", self.handle_event)
        print("CMM subscribed to task_failed, manual_intervention_required, and data_processing_error events.")

    def handle_event(self, event_type, event_data):
        """
        Handles an incoming event, logs it, and checks for capability gaps.

        Args:
            event_type (str): The type of event (e.g., "task_failed").
            event_data (dict): The data associated with the event.
        """
        print(f"CMM received event: {event_type} with data: {event_data}")

        timestamp = datetime.utcnow()
        log_entry = {"timestamp": timestamp, "data": event_data}

        # Use a key to group similar events for analysis
        event_key = self._get_event_key(event_type, event_data)
        self.event_log[event_key].append(log_entry)

        self.analyze_for_gaps(event_key)

    def _get_event_key(self, event_type, event_data):
        """Creates a consistent key for grouping events."""
        if event_type == "task_failed":
            return f"{event_type}_{event_data.get('task_name', 'unknown_task')}"
        elif event_type == "manual_intervention_required":
            return f"{event_type}_{event_data.get('workflow_step', 'unknown_step')}"
        elif event_type == "data_processing_error":
            return f"{event_type}_{event_data.get('data_type', 'unknown_data')}"
        return event_type

    def analyze_for_gaps(self, event_key):
        """Analyzes the event log for a given key to identify patterns that indicate a gap."""

        now = datetime.utcnow()
        recent_events = [
            e for e in self.event_log[event_key]
            if (now - e["timestamp"]).total_seconds() / 3600 < CONFIG["time_window_hours"]
        ]

        gap_detected = False
        if "task_failed" in event_key and len(recent_events) >= CONFIG["task_failure_threshold"]:
            gap_detected = True
            reason = f"Task '{event_key.split('_')[-1]}' failed {len(recent_events)} times recently."

        elif "manual_intervention_required" in event_key and len(recent_events) >= CONFIG["manual_intervention_threshold"]:
            gap_detected = True
            reason = f"Manual intervention at '{event_key.split('_')[-1]}' was required {len(recent_events)} times recently."

        elif "data_processing_error" in event_key and len(recent_events) >= 1:
            # A single data processing error for a new data type is enough to flag a gap
            gap_detected = True
            reason = f"A new or unprocessable data type '{event_key.split('_')[-1]}' was encountered."

        if gap_detected:
            print(f"Capability Gap DETECTED: {reason}")
            self.generate_gap_report(event_key, reason, recent_events)
            # In a real system, we might clear the log for this key to avoid repeated reports
            # self.event_log[event_key] = []

    def generate_gap_report(self, event_key, reason, events):
        """
        Generates a structured report for the identified gap and triggers the Agent Forge.
        """
        report_id = f"GAP-{uuid.uuid4()}"
        report = {
            "report_id": report_id,
            "timestamp": datetime.utcnow().isoformat(),
            "detected_by": "CapabilityMonitoringModule",
            "gap_summary": reason,
            "event_key": event_key,
            "event_count": len(events),
            "events": [e["data"] for e in events]
        }

        print(f"Generated Gap Report {report_id}:\n{json.dumps(report, indent=2)}")

        # Trigger the next step in the autonomy workflow
        self.agent_forge_trigger(report)


# --- Mock classes for demonstration purposes ---

class MockEventBus:
    def __init__(self):
        self.subscribers = defaultdict(list)
    def subscribe(self, event_type, callback):
        self.subscribers[event_type].append(callback)
    def publish(self, event_type, event_data):
        for callback in self.subscribers[event_type]:
            callback(event_type, event_data)

def mock_agent_forge_trigger(gap_report):
    print("\n---!!! Agent Forge Triggered !!!---")
    print(f"Received Gap Report ID: {gap_report['report_id']}")
    print("The Agent Forge would now parse this report and generate a new agent proposal.")
    print("---!!!---------------------------!!!---")

# --- Example Usage ---

if __name__ == "__main__":
    # 1. Setup
    event_bus = MockEventBus()
    cmm = CapabilityMonitoringModule(event_bus, mock_agent_forge_trigger)
    cmm.subscribe_to_events()

    print("\n--- Simulating System Events ---")

    # 2. Simulate a repeated task failure
    print("\n[Simulation] Simulating repeated task failures for 'ProcessQuarterlyFilings'...")
    for i in range(CONFIG["task_failure_threshold"]):
        event_bus.publish(
            "task_failed",
            {"task_name": "ProcessQuarterlyFilings", "error": "API rate limit exceeded", "agent": "SECDataAgent"}
        )

    # 3. Simulate an unprocessable data type
    print("\n[Simulation] Simulating an unprocessable data type 'geospatial_shipping_data'...")
    event_bus.publish(
        "data_processing_error",
        {"data_type": "geospatial_shipping_data", "source": "PortAuthorityAPI", "agent": "SupplyChainAgent"}
    )
