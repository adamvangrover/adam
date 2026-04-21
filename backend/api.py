from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Any
import datetime
import json
import asyncio

background_tasks = set()


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
                    task = asyncio.create_task(connection.send_text(message))
                    background_tasks.add(task)
                    task.add_done_callback(background_tasks.discard)
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




class ClearanceRequest(BaseModel):
    module_id: str
    key: str

@app.post("/api/v1/secure-payload")
async def get_secure_payload(req: ClearanceRequest):
    """
    Mock endpoint for securely fetching gated content.
    Requires a valid clearance key to return the decrypted payload.
    """
    VALID_KEYS = ["admin", "adam", "genesis"]

    if req.key.lower().strip() not in VALID_KEYS:
        return {"status": "error", "message": "ACCESS DENIED. INVALID KEY."}

    # In a real system, we'd fetch the module content from the DB using module_id.
    # For now, we return a mock success payload.
    return {
        "status": "success",
        "payload": f"SECURE PAYLOAD FOR MODULE {req.module_id}\nDecryption successful."
    }


from pydantic import BaseModel
from typing import List, Optional
import os

class MemoryInput(BaseModel):
    content: str
    category: str
    tags: Optional[List[str]] = None

@app.post("/api/v1/memory")
async def add_memory(req: MemoryInput):
    """
    Stores memory logs from the UI or models.
    """
    try:
        from core.memory.engine import MemoryEngine
        engine = MemoryEngine()
        engine.store_memory(content=req.content, category=req.category, tags=req.tags)
        return {"status": "success", "message": "Memory stored."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/v1/memory")
async def get_memories(query: str = "", limit: int = 10):
    """
    Fetches memories based on query.
    """
    try:
        from core.memory.engine import MemoryEngine
        engine = MemoryEngine()

        if query:
            results = engine.query_memory(query, limit)
        else:
            conn = engine._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT content, category, tags, timestamp FROM memories ORDER BY timestamp DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            engine._close_conn(conn)
            import json
            results = [
                {
                    "content": r[0],
                    "category": r[1],
                    "tags": json.loads(r[2]),
                    "timestamp": r[3]
                }
                for r in rows
            ]

        return {"status": "success", "data": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)