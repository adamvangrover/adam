import tinker
import os
import json
import asyncio
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
TEACHER_MODEL = "Qwen/Qwen3-235B-A22B"
OUTPUT_FILE = "../data/distill_behavioral.jsonl"
PROMPT_FILE = "SYSTEM_PROMPT_BEHAVIORAL_ECON.md"
QUERIES = [
    "Company X missed earnings by 5% but raised guidance. Analyze market reaction.",
    "Review this trading signal: Buy TSLA on RSI dip.",
    "The CEO of Company Y just gave a very bullish interview on TV.",
    "This high-yield bond (CCC) is trading at 50 cents on the dollar. What's the play?",
    "S&P 500 just hit an all-time high. Is it time to go risk-on?"
]

async def main():
    """
    Asynchronously generates the synthetic dataset for Stage 2 distillation
    using the 'Mentor' model.
    """
    try:
        with open(PROMPT_FILE, 'r') as f:
            teacher_prompt_template = f.read()
    except FileNotFoundError:
        print(f"Error: Could not find {PROMPT_FILE}. Exiting.")
        exit(1)

    print(f"Initializing Teacher Model ({TEACHER_MODEL}) for data generation...")

    service_client = tinker.ServiceClient()
    sampling_client = await service_client.create_sampling_client_async(
        model_name=TEACHER_MODEL
    )

    print(f"Generating synthetic distillation data to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    async def generate_sample(query):
        prompt = teacher_prompt_template.format(query=query)
        try:
            future = await sampling_client.sample_async(
                prompt=prompt,
                max_new_tokens=1024,
                temperature=0.6,
                stop_sequences=["---"]
            )
            response = await future
            return {
                "input": query,
                "output": response.generation.strip()
            }
        except Exception as e:
            print(f"Error sampling from model for query: {query}\\n{e}")
            return None

    tasks = [generate_sample(query) for query in QUERIES]
    results = await asyncio.gather(*tasks)

    with open(OUTPUT_FILE, 'w') as f:
        for data_pair in results:
            if data_pair:
                f.write(json.dumps(data_pair) + '\\n')

    print(f"Done. {len(results)} distillation examples generated.")

if __name__ == "__main__":
    asyncio.run(main())
