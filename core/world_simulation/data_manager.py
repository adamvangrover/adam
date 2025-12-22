# core/world_simulation/data_manager.py

import json
import pandas as pd
from typing import List, Dict


class DataManager:
    def __init__(self, output_dir: str = "data/world_simulation"):
        self.output_dir = output_dir

    def save_run_data(self, run_id: int, data: List[Dict]):
        """Saves the data for a single simulation run to a JSONL file."""
        filepath = f"{self.output_dir}/run_{run_id}.jsonl"
        with open(filepath, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")

    def load_run_data(self, run_id: int) -> pd.DataFrame:
        """Loads the data for a single simulation run into a pandas DataFrame."""
        filepath = f"{self.output_dir}/run_{run_id}.jsonl"
        return pd.read_json(filepath, lines=True)

    def load_all_data(self, num_runs: int) -> pd.DataFrame:
        """Loads the data for all simulation runs into a single pandas DataFrame."""
        all_data = []
        for i in range(num_runs):
            run_data = self.load_run_data(i)
            run_data["run_id"] = i
            all_data.append(run_data)
        return pd.concat(all_data, ignore_index=True)
