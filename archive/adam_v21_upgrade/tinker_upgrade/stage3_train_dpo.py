import tinker
import json
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# We are aligning the model that has *already been distilled*.
# This demonstrates adapter stacking/chaining.
BASE_MODEL = "meta-llama/Llama-3.1-8B"
# We load the "Mind" adapter from Stage 2 as our starting point.
BASE_ADAPTER_PATH = "adam_distilled_mind_v1"
DATA_PATH = "../data/adam_preference_data.jsonl"
NEW_ADAPTER_PATH = "adam_aligned_soul_v1" # The final DPO adapter

# DPO-specific hyperparameters
# Beta is the "strength" of the alignment
DPO_BETA = 0.1
# DPO typically uses a lower learning rate than SFT
LEARNING_RATE = 1e-5

print(f"Starting Stage 3: DPO Alignment for {BASE_MODEL}...")

# 1. Initialize Client
service_client = tinker.ServiceClient()

# 2. Create DPO Training Client
# Based on the Tinker DPO guide, we initialize
# a LoRA client and specify our DPO parameters.
# We also load the base adapter from Stage 2.
try:
    training_client = service_client.create_lora_training_client(
        base_model=BASE_MODEL,
        rank=16,
        # Load the weights from our Stage 2 "Mind" adapter
        base_adapter_path=BASE_ADAPTER_PATH
    )
    # Configure the client for DPO loss
    training_client.configure_dpo_loss(beta=DPO_BETA, learning_rate=LEARNING_RATE)
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

print(f"Data loaded. Starting DPO Training Loop on {len(dataset)} examples...")

# 4. The "Simple Loop on CPU" for DPO
# This loop is slightly different, as the forward_backward
# pass for DPO requires three inputs: prompt, chosen, rejected.
for epoch in range(2): # DPO typically runs for fewer epochs
    for step, item in enumerate(dataset):
        # Ensure all keys are present
        if not all(k in item for k in ("prompt", "chosen", "rejected")):
            print(f"Skipping malformed data at step {step}")
            continue

        # Tinker's DPO implementation handles the log-probability
        # calculations remotely.
        metrics = training_client.forward_backward_dpo(
            prompt=item['prompt'],
            chosen_completion=item['chosen'],
            rejected_completion=item['rejected']
        )
        training_client.optim_step()

        if step % 5 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {metrics.get('loss', 'N/A')}")

# 5. Save the Final Aligned Adapter
print(f"DPO Alignment complete. Saving 'Soul' adapter to {NEW_ADAPTER_PATH}...")
training_client.save_state(path=NEW_ADAPTER_PATH)
print(f"Stage 3 complete. Adapter '{NEW_ADAPTER_PATH}' saved.")
