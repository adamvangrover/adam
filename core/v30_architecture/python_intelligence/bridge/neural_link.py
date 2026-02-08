import asyncio
import json
import logging
import uvicorn
from typing import List
from datetime import datetime
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Thought(BaseModel):
    timestamp: str
    agent_name: str
    content: str
    conviction_score: float

class NeuralEmitter:
    """
    Broadcasts thoughts from the backend agents to connected frontend clients.
    """
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, thought: Thought):
        """
        Sends a thought to all connected clients.
        """
        message = thought.model_dump_json()
        to_remove = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")
                to_remove.append(connection)

        for conn in to_remove:
            self.disconnect(conn)

# Initialize FastAPI app
app = FastAPI()
emitter = NeuralEmitter()

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await emitter.connect(websocket)
    try:
        while True:
            # Keep connection alive, maybe receive commands later
            data = await websocket.receive_text()
            logger.info(f"Received from client: {data}")
    except WebSocketDisconnect:
        emitter.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        emitter.disconnect(websocket)

# Helper for agents to emit thoughts (in-process or via HTTP/RPC in future)
# For now, we'll expose a simple function to inject thoughts into the event loop
async def emit_thought(thought: Thought):
    await emitter.broadcast(thought)

if __name__ == "__main__":
    # Self-contained execution for testing
    uvicorn.run(app, host="0.0.0.0", port=8000)