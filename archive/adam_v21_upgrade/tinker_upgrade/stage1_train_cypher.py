import tinker
import json
import os
from dotenv import load_dotenv

load_dotenv()

def train_cypher_agent():
    service_client = tinker.ServiceClient()

    BASE_MODEL = "meta-llama/Llama-3.1-8B"
    print(f"Initializing Training Client for {BASE_MODEL}...")

    # Create the LoRA training client
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=16
    )

    data_path = "../data/neo4j_tool_use.jsonl" # This will use the expanded dataset
    with open(data_path, 'r') as f:
        dataset = [json.loads(line) for line in f]

    print("Starting Training Loop (Remote GPUs)...")
    for epoch in range(3):
        for step, item in enumerate(dataset):
            prompt = f"Question: {item['question']}\\nCypher Query:"
            target = item['query']

            # Send data to Tinker for forward/backward pass
            metrics = training_client.forward_backward(
                input_text=prompt,
                target_text=target
            )

            # Apply gradient updates
            training_client.optim_step()

            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {metrics.get('loss', 'N/A')}")

    print("Training complete. Saving LoRA weights...")
    # Save the final adapter to cloud storage
    training_client.save_state(path="adam_cypher_lora_v1")
    print("Weights saved to Tinker cloud storage as 'adam_cypher_lora_v1'")

if __name__ == "__main__":
    train_cypher_agent()
