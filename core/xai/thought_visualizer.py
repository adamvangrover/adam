from typing import List, Dict, Any
import json
import os
from core.system.state_manager import StateManager, AgentSnapshot

class ThoughtNode:
    def __init__(self, id: str, description: str, children: List['ThoughtNode'] = None, data: Dict[str, Any] = None):
        self.id = id
        self.description = description
        self.children = children or []
        self.data = data or {}

    def to_dict(self):
        return {
            "id": self.id,
            "description": self.description,
            "children": [c.to_dict() for c in self.children],
            "data": self.data
        }

class ThoughtVisualizer:
    """
    Generates a 'Chain of Thought' tree from agent execution snapshots.
    """

    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager

    def generate_tree(self, agent_id: str) -> Dict[str, Any]:
        """
        Reconstructs the execution path of an agent from its snapshots.
        """
        snapshot_ids = self.state_manager.list_snapshots(agent_id)
        snapshots = [self.state_manager.load_snapshot(sid) for sid in snapshot_ids]

        # Sort by timestamp
        snapshots.sort(key=lambda x: x.timestamp)

        root = ThoughtNode(id="root", description=f"Execution Trace: {agent_id}")
        current_parent = root

        for snap in snapshots:
            node = ThoughtNode(
                id=snap.snapshot_id,
                description=snap.step_description,
                data={
                    "timestamp": snap.iso_time,
                    "memory_keys": list(snap.memory_state.keys()),
                    "context_keys": list(snap.context_state.keys())
                }
            )
            # Linear chain for now, but structure supports branching if we had parent_id in snapshot
            current_parent.children.append(node)
            current_parent = node

        return root.to_dict()

    def export_tree(self, agent_id: str, output_path: str):
        tree = self.generate_tree(agent_id)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(tree, f, indent=2)
