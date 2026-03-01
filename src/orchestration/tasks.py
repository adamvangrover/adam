import os
import json
import redis
import math
from typing import List, Dict, Any, Tuple
from celery import shared_task
from .celery_app import celery_app
from src.core.config import settings
from src.core.logging import logger
from src.ingestion.plugin_manager import plugin_manager
from src.llm.agent import get_agent
from src.llm.schemas import SpreadsheetBatchOutput

# Initialize Redis client for tracking progress (SSE)
redis_client = redis.from_url(settings.redis_url)

def update_task_state(task_id: str, status: str, progress: int, message: str, meta: dict = None):
    """
    Updates the state of a specific task within Redis.
    This shared state object is continuously polled by the SSE endpoint.
    """
    state = {
        "status": status,
        "progress": progress,
        "message": message,
        "meta": meta or {}
    }
    # Store for 1 hour to prevent Redis bloat
    redis_client.setex(f"task_progress:{task_id}", 3600, json.dumps(state))

def calculate_optimal_chunk_size(total_rows: int, tokens_per_row: int = 50,
                               max_context: int = 128000, prompt_tokens: int = 500,
                               safety_margin: float = 0.8) -> int:
    """
    Implements the contextual window math to determine batch sizes.
    C_size = ((T_max - T_prompt) / T_row_avg) * S_safety_margin
    """
    max_chunk_size = int(((max_context - prompt_tokens) / tokens_per_row) * safety_margin)

    # Sensible defaults for realistic spreadsheets
    return min(max_chunk_size, 100) # Processing 100 rows at a time max

@shared_task(bind=True)
def process_data_pipeline(self, file_path: str, file_extension: str, instructions: str, model_name: str = "openai/gpt-4o"):
    """
    The core orchestration function executed by a Celery worker.
    Handles Ingestion -> Formatting -> Batching -> LLM Inference.
    """
    task_id = self.request.id
    update_task_state(task_id, "STARTED", 0, "Initializing pipeline...", {"file": os.path.basename(file_path)})

    try:
        # 1. Ingestion Phase
        update_task_state(task_id, "INGESTING", 10, f"Loading and parsing {file_extension} file...")

        # Dynamically retrieve the correct parsing strategy based on extension
        strategy = plugin_manager.get_strategy(file_extension)

        # Execute the parser (Strategy Pattern)
        with open(file_path, "rb") as f:
            raw_data = strategy.parse(f.read(), file_extension=file_extension)

        total_rows = len(raw_data)
        update_task_state(task_id, "CHUNK_COMPUTATION", 20, f"Parsed {total_rows} rows. Calculating batch size...")

        if total_rows == 0:
            update_task_state(task_id, "COMPLETED", 100, "File is empty.")
            return {"status": "success", "processed_rows": 0}

        # 2. Batching Phase
        chunk_size = calculate_optimal_chunk_size(total_rows)
        batches = [raw_data[i:i + chunk_size] for i in range(0, total_rows, chunk_size)]
        total_batches = len(batches)

        update_task_state(task_id, "BATCHING", 30, f"Segmented data into {total_batches} batches of up to {chunk_size} rows.")

        # Initialize LLM Agent
        agent = get_agent(model_name)

        all_results = []

        # 3. Inference Phase (Iterative Batching)
        for i, batch in enumerate(batches):
            current_progress = 30 + int((i / total_batches) * 60)
            update_task_state(task_id, "PROCESSING", current_progress, f"Processing batch {i + 1} of {total_batches} through {model_name}...")

            # Use SpreadsheetLLM format: Convert to Markdown
            # We assume the strategy implements a serialize_to_markdown or we use pd.DataFrame
            import pandas as pd
            df = pd.DataFrame(batch)
            markdown_table = df.to_markdown(index=False)

            # Construct the final prompt for this batch
            prompt = f"User Instructions:\n{instructions}\n\nData Batch (Markdown):\n{markdown_table}"

            # Synchronous LLM call (Celery workers are independent processes)
            try:
                # pydantic_ai handles proxying via LiteLLM and structural validation
                result = agent.run_sync(prompt)

                # result.data is guaranteed to be a SpreadsheetBatchOutput
                for row in result.data.processed_rows:
                    all_results.append(row.model_dump())

            except Exception as e:
                # Need to use specific Retry exception type
                from celery.exceptions import Retry
                if isinstance(e, Retry):
                    raise
                logger.error(f"Error processing batch {i+1}: {e}")
                update_task_state(task_id, "ERROR", current_progress, f"Failed on batch {i + 1}: {str(e)}")
                raise self.retry(exc=e, countdown=5) # Exponential backoff retry

        # 4. Finalization
        update_task_state(task_id, "COMPLETED", 100, f"Successfully processed {len(all_results)} rows.")

        # In a real system, you would save all_results back to a DB or a new file here.
        # For demonstration, we simply return success.

        # Cleanup temporary file
        if os.path.exists(file_path):
            os.remove(file_path)

        return {"status": "success", "processed_rows": len(all_results)}

    except self.retry.Retry:
        # Re-raise celery retry exceptions so it can actually retry
        raise
    except Exception as e:
        logger.error(f"Fatal error in pipeline: {e}")
        update_task_state(task_id, "FAILED", 0, f"Pipeline failed: {str(e)}")
        raise e
