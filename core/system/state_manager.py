import json
import os
import time
from typing import Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field

class AgentSnapshot(BaseModel):
    """
    Schema for a point-in-time snapshot of an agent's state.
    """
    snapshot_id: str
    timestamp: float
    iso_time: str
    agent_id: str
    step_description: str
    memory_state: Dict[str, Any] = Field(default_factory=dict)
    context_state: Dict[str, Any] = Field(default_factory=dict)

class StateManager:
    """
    Manages the serialization and retrieval of agent state snapshots ("The Rewind Button").
    """

    def __init__(self, snapshot_dir: str = "core/libraries_and_archives/snapshots"):
        self.snapshot_dir = snapshot_dir
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def save_snapshot(self, agent_id: str, step_description: str, memory: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Saves the current state of an agent. Returns the snapshot ID.
        """
        ts = time.time()
        iso = datetime.utcfromtimestamp(ts).isoformat() + "Z"
        snap_id = f"{agent_id}_{int(ts*1000)}"

        snapshot = AgentSnapshot(
            snapshot_id=snap_id,
            timestamp=ts,
            iso_time=iso,
            agent_id=agent_id,
            step_description=step_description,
            memory_state=memory,
            context_state=context
        )

        path = os.path.join(self.snapshot_dir, f"{snap_id}.json")
        with open(path, 'w') as f:
            f.write(snapshot.model_dump_json(indent=2))

        return snap_id

    def load_snapshot(self, snapshot_id: str) -> AgentSnapshot:
        """
        Loads a snapshot by ID.
        """
        path = os.path.join(self.snapshot_dir, f"{snapshot_id}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Snapshot {snapshot_id} not found.")

        with open(path, 'r') as f:
            data = json.load(f)

        return AgentSnapshot(**data)

    def list_snapshots(self, agent_id: str = None) -> List[str]:
        """
        Lists available snapshots, optionally filtered by agent.
        """
        snaps = []
        for f in os.listdir(self.snapshot_dir):
            if f.endswith(".json"):
                if agent_id and not f.startswith(agent_id):
                    continue
                snaps.append(f.replace(".json", ""))
        return sorted(snaps)
