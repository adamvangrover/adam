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
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role

    async def emit(self, packet_type: str, payload: dict):
        packet = NeuralPacket(
            source_agent=self.name,
            packet_type=packet_type,
            payload=payload
        )
        await emit_packet(packet)
        logger.debug(f"{self.name} emitted: {packet_type}")
