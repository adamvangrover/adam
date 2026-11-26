import tinker
import os
import json
from dotenv import load_dotenv
load_dotenv()
# --- Configuration ---
# The "Mentor" model, selected for its "thinking mode"
TEACHER_MODEL = "Qwen/Qwen3-235B-A22B"
OUTPUT_FILE = "../data/distill_behavioral.jsonl"
PROMPT_FILE = "SYSTEM_PROMPT_BEHAVIORAL_ECON.md"
# Load the "brain" of the teacher
try:
    with open(PROMPT_FILE, 'r') as f:
        TEACHER_PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    print(f"Error: Could not find {PROMPT_FILE}. Exiting.")
    exit(1)
# This list would be expanded to thousands of queries
# drawn from news headlines, analyst reports, etc.
QUERIES = [
    "TSLA stock dropped 10% on delivery misses. Is it a buy?",
    "Company X missed earnings by 5% but raised guidance. Analyze market reaction.",
    "Review this trading signal: Buy TSLA on RSI dip."
]
print(f"Initializing Teacher Model ({TEACHER_MODEL}) for data generation...")
# 1. Initialize Tinker Client
service_client = tinker.ServiceClient()
# 2. Create a SAMPLING client for the Teacher model
# We are not training the Teacher, only generating text from it.
sampling_client = service_client.create_sampling_client(
    model_name=TEACHER_MODEL
)
print(f"Generating synthetic distillation data to {OUTPUT_FILE}...")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
with open(OUTPUT_FILE, 'w') as f:
    for i, query in enumerate(QUERIES):
        print(f"Generating example {i+1}/{len(QUERIES)}...")
        # Format the prompt for the teacher
        prompt = TEACHER_PROMPT_TEMPLATE.format(query=query)
        try:
            # Call the Teacher model to get its high-quality reasoning
            response = sampling_client.sample(
                prompt=prompt,
                max_new_tokens=1024, # Allow for detailed reasoning
                temperature=0.6,    # Low-ish temp for factual, structured output
                stop_sequences=["---"] # Stop at the end of the response
            ).result() #.result() blocks until completion
            # The generated data is a pair of (input, teacher_output)
            # This is the training data for the Student model.
            data_pair = {
                "input": query,
                "output": response.generation.strip()
            }
            f.write(json.dumps(data_pair) + '\n')
        except Exception as e:
            print(f"Error sampling from model for query: {query}\n{e}")
print(f"Done. {len(QUERIES)} distillation examples generated.")
