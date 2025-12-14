import json
import logging
from typing import List, Dict, Any
from core.system.memory_manager import VectorMemoryManager

logger = logging.getLogger(__name__)

class FineTuningDriver:
    """
    Extracts high-quality interaction pairs from memory for model fine-tuning.
    """
    def __init__(self, output_file: str = "data/training/finetune_dataset.jsonl"):
        self.memory_manager = VectorMemoryManager()
        self.output_file = output_file

    def generate_dataset(self):
        """
        Scans memory for completed analyses and formats them as prompt-completion pairs.
        """
        history = self.memory_manager.load_history()
        dataset = []

        for entry in history:
            # We assume the analysis summary is a good "completion" for a task "Analyze {company_id}"
            # In a real system, we'd use the original prompt.
            # Here we synthesize a prompt.

            prompt = f"Perform a fundamental analysis of {entry['company_id']}."
            completion = entry['analysis_summary']

            # Quality Filter: Skip very short analyses (likely failures)
            if len(completion) < 100:
                continue

            dataset.append({
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "source": "Adam_Memory",
                    "timestamp": entry['timestamp'],
                    "company": entry['company_id']
                }
            })

        self._save_dataset(dataset)
        logger.info(f"Generated {len(dataset)} training examples.")

    def _save_dataset(self, dataset: List[Dict[str, Any]]):
        import os
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        with open(self.output_file, 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + "\n")
