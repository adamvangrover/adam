import asyncio
import json
import os

try:
    from datasets import load_dataset as _load_dataset_lib
except ImportError:
    _load_dataset_lib = None

async def load_structured_repository(file_path):
    print(f'Loading Future-Aligned Data from {file_path}...')

    # Determine if we can use the library
    use_lib = _load_dataset_lib

    if use_lib:
        try:
            dataset = use_lib('json', data_files=file_path, split='train', streaming=True)
            # streaming=True returns an IterableDataset
            for example in dataset:
                yield format_for_lora(example)
                # Yield control to event loop
                await asyncio.sleep(0)
        except Exception as e:
             print(f"Error using datasets library: {e}. Falling back to manual read.")
             use_lib = None # Force fallback

    if not use_lib:
        print("Using Manual JSONL loader.")
        # Mock loading from file directly
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            example = json.loads(line)
                            yield format_for_lora(example)
                            await asyncio.sleep(0)
                        except json.JSONDecodeError:
                            continue
        else:
            print(f"File {file_path} not found.")

def format_for_lora(example):
    # Formats JSONL into Tokenizer-ready string
    if 'messages' in example and len(example['messages']) >= 3:
        return {
            'input_text': f"<|system|>\n{example['messages'][0]['content']}\n<|user|>\n{example['messages'][1]['content']}\n<|assistant|>\n",
            'target_text': example['messages'][2]['content']
        }
    return example

async def main():
    # Check if file exists, if not warn
    if not os.path.exists('adam_v22_instruction_tuning.jsonl'):
        print("File 'adam_v22_instruction_tuning.jsonl' not found. Run the full pipeline first.")
        return

    count = 0
    async for training_sample in load_structured_repository('adam_v22_instruction_tuning.jsonl'):
        count += 1
        if count <= 1:
            print(f"Sample loaded keys: {training_sample.keys()}")
    print(f"Async loading complete. Loaded {count} samples.")

if __name__ == '__main__':
    asyncio.run(main())
