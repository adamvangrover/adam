import uuid
import os
import json
import asyncio
from typing import Annotated
from fastapi import APIRouter, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from src.orchestration.tasks import process_data_pipeline, redis_client
from src.core.logging import logger

router = APIRouter()

# Directory for temporary file storage
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

class IngestionResponse(BaseModel):
    task_id: str
    message: str
    status: str

    model_config = ConfigDict(populate_by_name=True)

@router.post("/upload", response_model=IngestionResponse, status_code=202)
async def upload_file(
    file: UploadFile = File(...),
    instructions: str = Form(...),
    model_name: str = Form("openai/gpt-4o")
):
    """
    Accepts a file upload (Excel, CSV, PDF), saves it to a temporary directory,
    and kicks off a background Celery task. Returns a task UUID immediately (202 Accepted).
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Get extension
    _, ext = os.path.splitext(file.filename)
    if not ext:
        raise HTTPException(status_code=400, detail="File has no extension")

    # Generate unique ID and save file
    task_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{task_id}{ext}")

    with open(file_path, "wb") as buffer:
        # Stream file to disk to prevent memory overflow on large files
        chunk = await file.read(1024 * 1024) # 1MB chunks
        while chunk:
            buffer.write(chunk)
            chunk = await file.read(1024 * 1024)

    logger.info(f"File saved to {file_path}, starting Celery task {task_id}")

    # Enqueue task to Celery
    process_data_pipeline.apply_async(
        args=[file_path, ext, instructions, model_name],
        task_id=task_id
    )

    return IngestionResponse(
        task_id=task_id,
        message="File ingested and processing started in background.",
        status="accepted"
    )

@router.get("/status/{task_id}")
async def stream_status(task_id: str):
    """
    Server-Sent Events (SSE) endpoint to stream real-time task progress.
    Polls Redis for the latest task state and yields updates.
    """
    async def event_generator():
        # Continually poll the task state until complete or error
        while True:
            # Note: For production, consider using Redis Pub/Sub instead of polling
            state_data = redis_client.get(f"task_progress:{task_id}")

            if state_data:
                state = json.loads(state_data)

                # Format as SSE
                yield f"data: {json.dumps(state)}\n\n"

                if state.get("status") in ["COMPLETED", "FAILED", "ERROR"]:
                    break
            else:
                yield f"data: {json.dumps({'status': 'PENDING', 'message': 'Task queued or not found'})}\n\n"

            await asyncio.sleep(2) # Poll every 2 seconds

    return StreamingResponse(event_generator(), media_type="text/event-stream")
