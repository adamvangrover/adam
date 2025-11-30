import tinker
import json
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
# The "Workhorse" model being trained
STUDENT_MODEL = "meta-llama/Llama-3.1-8B"
DATA_PATH = "../data/distill_behavioral.jsonl"
ADAPTER_PATH = "adam_distilled_mind_v1" # The new "Mind" adapter

print(f"Starting Stage 2: Distillation Training for {STUDENT_MODEL}...")

# 1. Initialize Client
service_client = tinker.ServiceClient()

# 2. Create LoRA Training Client for the Student
training_client = service_client.create_lora_training_client(
    base_model=STUDENT_MODEL,
    rank=16
)

# 3. Load Distilled Data
try:
    with open(DATA_PATH, 'r') as f:
        dataset = [json.loads(line) for line in f]
except FileNotFoundError:
    print(f"Error: Data file {DATA_PATH} not found.")
    print("Run 'stage2_create_data.py' first.")
    exit(1)

print(f"Data loaded. Starting Distillation Training Loop on {len(dataset)} examples...")

# 4. The "Simple Loop on CPU"
# This loop trains the Student to mimic the Teacher.
for epoch in range(3):
    for step, item in enumerate(dataset):
        # The student is trained to produce the teacher's
        # high-quality output given only the simple input query.
        # This "bakes" the reasoning into the model weights.
        prompt = f"User Query: {item['input']}\\n\\nAnalysis:"
        target = item['output']

        metrics = training_client.forward_backward(
            input_text=prompt,
            target_text=target
        )
        training_client.optim_step()

        if step % 10 == 0:
            print(f"Epoch {epoch} | Step {step} | Loss: {metrics.get('loss', 'N/A')}")

# 5. Save the Distilled Adapter
print(f"Distillation complete. Saving 'Mind' adapter to {ADAPTER_PATH}...")
training_client.save_state(path=ADAPTER_PATH)
print(f"Stage 2 complete. Adapter '{ADAPTER_PATH}' saved.")
