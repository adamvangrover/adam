import asyncio
import uvicorn
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from datetime import datetime

class Thought(BaseModel):
    timestamp: str
    agent_name: str
    content: str
    conviction_score: float

class NeuralEmitter:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, thought: Thought):
        message = thought.model_dump_json()
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                # If sending fails, we might want to remove the connection,
                # but let's leave it to the disconnect handler for now
                # or handle it gracefully.
                pass

app = FastAPI()
emitter = NeuralEmitter()

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await emitter.connect(websocket)
    try:
        while True:
            # Keep the connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        emitter.disconnect(websocket)
    except Exception:
        emitter.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
