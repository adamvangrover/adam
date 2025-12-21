# scripts/run_llm_driven_simulation.py

import argparse

from core.llm.engines.openai_llm_engine import OpenAILLMEngine
from core.world_simulation.config import load_config
from core.world_simulation.data_manager import DataManager
from core.world_simulation.llm_driven_sim import LLMDrivenSim


def main():
    parser = argparse.ArgumentParser(description="Run the LLM-driven world simulation.")
    parser.add_argument(
        "--config",
        type=str,
        default="config/world_simulation/default.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    llm_engine = OpenAILLMEngine(
        api_key="YOUR_API_KEY",  # Replace with your API key
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
    )
    data_manager = DataManager()
    simulation = LLMDrivenSim(config, llm_engine)

    for i in range(config.simulation.runs):
        print(f"Running simulation {i+1}/{config.simulation.runs}...")
        history = simulation.run_simulation()
        data_manager.save_run_data(i, history)
        print(f"Simulation {i+1} complete.")

if __name__ == "__main__":
    main()
