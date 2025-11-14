import tinker
import json
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
STUDENT_MODEL = "meta-llama/Llama-3.1-8B"
DATA_PATH = "../data/distill_behavioral.jsonl"
ADAPTER_PATH = "adam_distilled_mind_v1"

async def main():
    """
    Asynchronously trains the 'Mind' (Student) agent using a high-performance,
    non-blocking distillation pipeline.
    """
    print(f"Starting Stage 2: Async Distillation Training for {STUDENT_MODEL}...")

    # 1. Initialize Client
    service_client = tinker.ServiceClient()

    # 2. Create LoRA Training Client for the Student asynchronously
    training_client = await service_client.create_lora_training_client_async(
        base_model=STUDENT_MODEL,
        lora_rank=16
    )

    # 3. Load Distilled Data
    try:
        with open(DATA_PATH, 'r') as f:
            dataset = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Data file {DATA_PATH} not found.")
        print("Run 'stage2_create_data.py' first.")
        exit(1)

    print(f"Data loaded. Starting Async Distillation Training Loop on {len(dataset)} examples...")

    # 4. The "Simple Loop on CPU" - Asynchronous Version
    for epoch in range(3):
        for step, item in enumerate(dataset):
            prompt = f"User Query: {item['input']}\\n\\nAnalysis:"
            target = item['output']

            # --- High-Performance Overlapping Pattern ---

            # 1. Submit the forward/backward pass request
            fwd_bwd_future = await training_client.forward_backward_async(
                input_text=prompt,
                target_text=target
            )

            # 2. Immediately submit the optimizer step request
            optim_future = await training_client.optim_step_async()

            # 3. Now, await the results
            metrics = await fwd_bwd_future
            await optim_future

            # --- End Pattern ---

            if step % 10 == 0:
                loss = metrics.get('loss', 'N/A')
                print(f"Epoch {epoch} | Step {step} | Loss: {loss}")

    # 5. Save the Distilled Adapter
    print(f"Distillation complete. Saving 'Mind' adapter to {ADAPTER_PATH}...")
    await training_client.save_state_async(path=ADAPTER_PATH)
    print(f"Stage 2 complete. Adapter '{ADAPTER_PATH}' saved.")

if __name__ == "__main__":
    asyncio.run(main())
