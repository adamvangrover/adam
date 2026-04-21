from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import logging
import asyncio
import uuid
import os
import json

from core.v30_architecture.python_intelligence.bridge.ephemeral_cortex import EphemeralCortex

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

class TopicRouter:
    """Manages subscriptions to specific topics."""
    def __init__(self):
        # map of topic -> set of WebSockets
        self.subscriptions: Dict[str, Set[WebSocket]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, websocket: WebSocket, topic: str):
        async with self._lock:
            if topic not in self.subscriptions:
                self.subscriptions[topic] = set()
            self.subscriptions[topic].add(websocket)
            logger.info(f"Client subscribed to topic: {topic}")

    async def unsubscribe(self, websocket: WebSocket, topic: str):
        async with self._lock:
            if topic in self.subscriptions and websocket in self.subscriptions[topic]:
                self.subscriptions[topic].remove(websocket)
                if not self.subscriptions[topic]:
                    del self.subscriptions[topic]

    async def get_subscribers(self, topic: str) -> Set[WebSocket]:
        async with self._lock:
            return self.subscriptions.get(topic, set())

class MeshEmitter:
    """
    Advanced broadcaster for the Neural Mesh with topic routing
    and ephemeral context storage.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.cortex = EphemeralCortex(max_history_per_topic=200)
        self.router = TopicRouter()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected to Mesh. Total: {len(self.active_connections)}")

        # By default, clients are subscribed to a "global" topic, but they can explicitly request others.
        await self.router.subscribe(websocket, "global")

    async def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            # Remove from all topics
            for topic in list(self.router.subscriptions.keys()):
                await self.router.unsubscribe(websocket, topic)
            logger.info(f"Client disconnected from Mesh. Total: {len(self.active_connections)}")

    async def broadcast(self, packet: NeuralPacket):
        """
        Sends a structured packet to relevant connected clients
        and stores it in the Ephemeral Cortex.
        """
        # Store context
        await self.cortex.ingest(packet.packet_type, packet)

        message = packet.model_dump_json()

        # Send to "global" listeners AND topic-specific listeners
        global_subs = await self.router.get_subscribers("global")
        topic_subs = await self.router.get_subscribers(packet.packet_type)

        target_connections = global_subs.union(topic_subs)

        to_remove = []
        for connection in target_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")
                to_remove.append(connection)

        for conn in to_remove:
            await self.disconnect(conn)

    async def handle_client_message(self, websocket: WebSocket, data: str):
        """Processes incoming messages from clients (e.g., subscription requests, context queries)."""
        try:
            msg = json.loads(data)
            action = msg.get("action")

            if action == "subscribe":
                topic = msg.get("topic")
                if topic:
                    await self.router.subscribe(websocket, topic)
            elif action == "query_context":
                topic = msg.get("topic")
                limit = msg.get("limit", 10)
                if topic:
                    history = await self.cortex.query(topic, limit)
                    response = {
                        "type": "context_response",
                        "topic": topic,
                        "data": [p.model_dump() for p in history]
                    }
                    await websocket.send_text(json.dumps(response))
        except json.JSONDecodeError:
            logger.warning(f"Received malformed JSON from client: {data}")
        except Exception as e:
            logger.error(f"Error handling client message: {e}")

# --- FastAPI App ---

app = FastAPI(title="Neural Mesh v3.0 - Ephemeral Cortex")
mesh = MeshEmitter()

# 🛡️ Sentinel: Restrict CORS to known frontend origins to prevent CSWSH
allowed_origins = [origin.strip() for origin in os.environ.get("CORS_ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/mesh")
async def websocket_mesh(websocket: WebSocket):
    # 🛡️ Sentinel: Manual Origin Check for WebSockets
    origin = websocket.headers.get("origin")
    is_allowed = False

    if "*" in allowed_origins:
        is_allowed = True
    elif origin is None:
        is_allowed = True
    elif origin in allowed_origins:
        is_allowed = True

    if not is_allowed:
        logger.warning(f"Blocking connection from unauthorized origin: {origin}")
        await websocket.close(code=1008)  # Policy Violation
        return

    await mesh.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await mesh.handle_client_message(websocket, data)
    except WebSocketDisconnect:
        await mesh.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await mesh.disconnect(websocket)

# Helper for agents to emit into the mesh
async def emit_packet(packet: NeuralPacket):
    await mesh.broadcast(packet)

# Expose cortex for programmatic access by other backend agents
def get_cortex():
    return mesh.cortex

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # nosec B104
