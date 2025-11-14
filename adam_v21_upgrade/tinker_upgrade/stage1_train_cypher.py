import tinker
import json
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def main():
    """
    Asynchronously trains the 'Hands' (Cypher) agent using a high-performance,
    non-blocking pipeline.
    """
    service_client = tinker.ServiceClient()

    BASE_MODEL = "meta-llama/Llama-3.1-8B"
    print(f"Initializing Training Client for {BASE_MODEL}...")

    # Create the LoRA training client asynchronously
    training_client = await service_client.create_lora_training_client_async(
        base_model=BASE_MODEL,
        lora_rank=16
    )

    data_path = "../data/neo4j_tool_use.jsonl"
    with open(data_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    print("Starting Asynchronous Training Loop (Remote GPUs)...")
    for epoch in range(3):
        for step, item in enumerate(dataset):
            prompt = f"Question: {item['question']}\\nCypher Query:"
            target = item['query']

            # --- High-Performance Overlapping Pattern ---

            # 1. Submit the forward/backward pass request
            fwd_bwd_future = await training_client.forward_backward_async(
                input_text=prompt,
                target_text=target
            )

            # 2. Immediately submit the optimizer step request without waiting
            optim_future = await training_client.optim_step_async()

            # 3. Now, await the results of both operations
            metrics = await fwd_bwd_future
            await optim_future

            # --- End Pattern ---

            if step % 10 == 0:
                # The result from the forward_backward future contains the metrics
                loss = metrics.get('loss', 'N/A')
                print(f"Epoch {epoch} | Step {step} | Loss: {loss}")

    print("Training complete. Saving LoRA weights...")
    # Save the final adapter asynchronously
    await training_client.save_state_async(path="adam_cypher_lora_v1")
    print("Weights saved to Tinker cloud storage as 'adam_cypher_lora_v1'")

if __name__ == "__main__":
    asyncio.run(main())
