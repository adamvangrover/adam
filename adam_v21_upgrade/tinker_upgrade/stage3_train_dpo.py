import tinker
import json
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
BASE_MODEL = "meta-llama/Llama-3.1-8B"
BASE_ADAPTER_PATH = "adam_distilled_mind_v1"
DATA_PATH = "../data/adam_preference_data.jsonl"
NEW_ADAPTER_PATH = "adam_aligned_soul_v1"
DPO_BETA = 0.1
LEARNING_RATE = 1e-5

async def main():
    """
    Asynchronously aligns the 'Soul' of the agent using a high-performance,
    non-blocking DPO pipeline.
    """
    print(f"Starting Stage 3: Async DPO Alignment for {BASE_MODEL}...")

    # 1. Initialize Client
    service_client = tinker.ServiceClient()

    # 2. Create DPO Training Client asynchronously
    try:
        training_client = await service_client.create_lora_training_client_async(
            base_model=BASE_MODEL,
            lora_rank=16,
            base_adapter_path=BASE_ADAPTER_PATH
        )
        await training_client.configure_dpo_loss_async(beta=DPO_BETA, learning_rate=LEARNING_RATE)
    except Exception as e:
        print(f"Error creating client. Does adapter '{BASE_ADAPTER_PATH}' exist? {e}")
        exit(1)

    # 3. Load Preference Data
    try:
        with open(DATA_PATH, 'r') as f:
            dataset = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Error: Data file {DATA_PATH} not found.")
        print("Run 'stage3_dpo_prep.py' first.")
        exit(1)

    print(f"Data loaded. Starting Async DPO Training Loop on {len(dataset)} examples...")

    # 4. The "Simple Loop on CPU" for DPO - Asynchronous Version
    for epoch in range(2):
        for step, item in enumerate(dataset):
            if not all(k in item for k in ("prompt", "chosen", "rejected")):
                print(f"Skipping malformed data at step {step}")
                continue

            # --- High-Performance Overlapping Pattern ---

            # 1. Submit the DPO forward/backward pass request
            dpo_future = await training_client.forward_backward_dpo_async(
                prompt=item['prompt'],
                chosen_completion=item['chosen'],
                rejected_completion=item['rejected']
            )

            # 2. Immediately submit the optimizer step request
            optim_future = await training_client.optim_step_async()

            # 3. Now, await the results
            metrics = await dpo_future
            await optim_future

            # --- End Pattern ---

            if step % 5 == 0:
                loss = metrics.get('loss', 'N/A')
                print(f"Epoch {epoch} | Step {step} | Loss: {loss}")

    # 5. Save the Final Aligned Adapter
    print(f"DPO Alignment complete. Saving 'Soul' adapter to {NEW_ADAPTER_PATH}...")
    await training_client.save_state_async(path=NEW_ADAPTER_PATH)
    print(f"Stage 3 complete. Adapter '{NEW_ADAPTER_PATH}' saved.")

if __name__ == "__main__":
    asyncio.run(main())
