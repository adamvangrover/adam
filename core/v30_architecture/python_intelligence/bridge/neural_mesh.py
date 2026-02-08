from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NeuralMesh")

# --- Protocol Definitions ---

class NeuralPacket(BaseModel):
    """
    Standardized data packet for the V30 Neural Mesh.
    Wraps all inter-agent communication and frontend updates.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    source_agent: str
    packet_type: str  # e.g., 'thought', 'market_data', 'risk_alert', 'system_status'
    payload: Dict[str, Any]
    priority: int = 1  # 0=Critical, 1=High, 2=Normal, 3=Debug

# --- Mesh Infrastructure ---

class MeshEmitter:
    """
    Advanced broadcaster for the Neural Mesh.
    Supports topic-based routing (future) and structured packet handling.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.history: List[NeuralPacket] = []  # Keep a short history for new clients

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected to Mesh. Total: {len(self.active_connections)}")

        # Send recent history to catch up
        for packet in self.history[-10:]:
            await websocket.send_text(packet.model_dump_json())

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected from Mesh. Total: {len(self.active_connections)}")

    async def broadcast(self, packet: NeuralPacket):
        """
        Sends a structured packet to all connected clients.
        """
        message = packet.model_dump_json()

        # Update history
        self.history.append(packet)
        if len(self.history) > 50:
            self.history.pop(0)

        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")
                to_remove.append(connection)

        for conn in to_remove:
            self.disconnect(conn)

# --- FastAPI App ---

app = FastAPI(title="Neural Mesh v2.0")
mesh = MeshEmitter()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/mesh")
async def websocket_mesh(websocket: WebSocket):
    await mesh.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        mesh.disconnect(websocket)
    except Exception:
        mesh.disconnect(websocket)

# Helper for agents to emit into the mesh
async def emit_packet(packet: NeuralPacket):
    await mesh.broadcast(packet)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
