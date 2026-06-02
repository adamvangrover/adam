"""
Purpose: Utility for extracting strings dynamically out of execution files.
Dependencies: json, os
Outputs: load_prompt function.
"""

import json
import os

def load_prompt(prompt_name: str, file_path: str = "assets/prompts.json") -> str:
    """
    Loads a text prompt or heuristic rule from an external JSON file
    to prevent context window saturation in language models reading Python files.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Prompt file {file_path} not found.")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = data.get("prompts", {})
    prompt = prompts.get(prompt_name)

    if not prompt:
        raise KeyError(f"Prompt '{prompt_name}' not found in {file_path}.")

    return prompt
