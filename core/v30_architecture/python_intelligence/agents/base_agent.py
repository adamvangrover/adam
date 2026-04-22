import logging
import os
import sys

# Configure logging
logger = logging.getLogger("BaseAgent")

# Import from the mesh bridge
try:
    from core.v30_architecture.python_intelligence.bridge.neural_mesh import NeuralPacket, emit_packet
except ImportError:
    # If run as script or different context, try to adjust path
    # Assuming this file is at core/v30_architecture/python_intelligence/agents/base_agent.py
    # We need to go up 4 levels to reach repo root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, '../../../../'))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from core.v30_architecture.python_intelligence.bridge.neural_mesh import NeuralPacket, emit_packet


class BaseAgent:
    """
    The foundational agent class for the v30 architecture.
    All modular agents in the Python Intelligence swarm must inherit from this class.
    It provides standard telemetry and communication channels via the NeuralMesh.

    Usage Example:
        class ConcreteAgent(BaseAgent):
            def __init__(self):
                super().__init__(name="Agent-X", role="Analyzer")

            async def run(self, input_data: dict):
                # Agents perform their specific operational logic
                result = self._process_data(input_data)

                # Agents broadcast their state/results to the mesh using emit()
                await self.emit("THOUGHT", {"msg": "analyzing", "output": result})
    """
    def __init__(self, name: str, role: str):
        """
        Initialize the agent with a unique identity.

        Args:
            name (str): The unique identifier for the agent instance.
            role (str): The functional role of the agent in the swarm.
        """
        self.name = name
        self.role = role

    async def emit(self, packet_type: str, payload: dict, **kwargs):
        """
        Asynchronously emit a telemetry packet to the NeuralMesh.

        Args:
            packet_type (str): The classification of the emitted data (e.g., "THOUGHT", "ACTION").
            payload (dict): The structured data payload to transmit.

        Usage Example:
            await self.emit(packet_type="THOUGHT", payload={"status": "processing"})
        """
        packet = NeuralPacket(
            source_agent=self.name,
            packet_type=packet_type,
            payload=payload
        )
        await emit_packet(packet)
        logger.debug(f"{self.name} emitted: {packet_type}")
