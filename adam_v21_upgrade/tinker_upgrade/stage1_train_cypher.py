import tinker
import json
import os
from dotenv import load_dotenv
load_dotenv()
def train_cypher_agent():
    # 1. Initialize Client
    # Establishes connection to the Tinker API
    service_client = tinker.ServiceClient()
    # 2. Select Model
    BASE_MODEL = "meta-llama/Llama-3.1-8B" # The "Workhorse"
    print(f"Initializing Training Client for {BASE_MODEL}...")
    # Instantiates the remote LoRA training process
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        lora_rank=16 # Common default for text tasks
    )
    # 3. Load Data
    # This path now points to the expanded 50+ example dataset
    data_path = "../data/neo4j_tool_use.jsonl"
    with open(data_path, 'r') as f:
        dataset = [json.loads(line) for line in f]
    print("Starting Training Loop (Remote GPUs)...")
    # 4. The "Simple Loop on CPU"
    # This local Python loop orchestrates the entire distributed
    # training job on remote GPUs.
    for epoch in range(3): # Short run for demo
        total_loss = 0
        for step, item in enumerate(dataset):
            # Construct the input/target pair
            prompt = f"Question: {item['question']}\nCypher Query:"
            target = item['query']
            # Send to Tinker (Forward/Backward)
            # This non-blocking API call sends the data and loss function
            # to the remote cluster for computation.
            metrics = training_client.forward_backward(
                input_text=prompt,
                target_text=target
            )
            # Optimizer Step
            # This call instructs the remote optimizer to update the
            # LoRA weights using the accumulated gradients.
            training_client.optim_step()
            if step % 10 == 0:
                print(f"Epoch {epoch} | Step {step} | Loss: {metrics.get('loss', 'N/A')}")
    # 5. Save the Adapter
    print("Training complete. Saving LoRA weights...")
    # Saves the final 'adam_cypher_lora_v1' adapter
    # to Tinker's cloud storage for later retrieval.
    training_client.save_state(path="adam_cypher_lora_v1")
    print("Weights saved to Tinker cloud storage as 'adam_cypher_lora_v1'")
if __name__ == "__main__":
    train_cypher_agent()
