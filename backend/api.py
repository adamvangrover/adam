from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Any
import datetime
import json
import asyncio

app = FastAPI(title="Adam v30 Telemetry API")

# Mock database for historical evaluation logs
MOCK_DB: List[Dict[str, Any]] = [
    {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event_type": "FINAL_APPROVAL",
        "data": {
            "iteration": 1,
            "metrics": {"spread_bps": 1500.0, "inventory": 500},
            "status": "PASS",
            "feedback": "Passed all evaluations"
        }
    }
]

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str, sender: WebSocket = None):
        for connection in self.active_connections:
            if connection != sender:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    print(f"Error sending message: {e}")

manager = ConnectionManager()

@app.get("/api/v1/eval-logs")
async def get_eval_logs():
    """
    REST API endpoint to query past evaluation reports.
    """
    return {"logs": MOCK_DB}

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket Server (Real-Time State)
    """
    await manager.connect(websocket)
    try:
        while True:
            # Controller script will act as a client pushing JSON payloads to this socket.
            data = await websocket.receive_text()
            try:
                # Try to parse as JSON to validate it's a structured payload
                payload = json.loads(data)
                # Broadcast the payload to all connected clients (like the React dashboard)
                # Do not broadcast back to the sender
                await manager.broadcast(json.dumps(payload), sender=websocket)

                # Optional: Save to mock DB for history if it's FINAL_APPROVAL
                if payload.get("event_type") == "FINAL_APPROVAL":
                    MOCK_DB.append(payload)
            except json.JSONDecodeError:
                # If not JSON, ignore or log
                print(f"Received non-JSON message: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
