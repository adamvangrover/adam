import json
from pydantic import BaseModel
from typing import Dict, Any
from json_logic import jsonLogic
import os

class Kernel:
    def __init__(self, logic_rules_path: str = "kernel/logic_rules.json"):
        if not os.path.exists(logic_rules_path):
             logic_rules_path = os.path.join(os.path.dirname(__file__), "logic_rules.json")
        with open(logic_rules_path, "r") as f:
            self.logic_rules = json.load(f)

    def evaluate_output(self, llm_output: BaseModel) -> Dict[str, Any]:
        """
        The LLM's raw output is strictly validated by Pydantic models, flattened into a standardized dictionary representing the context, and evaluated by a Python JSONLogic interpreter (json-logic-qubit) against logic_rules.json to mechanically override LLM-suggested steps for DAG routing.
        """
        context_dict = llm_output.model_dump()
        overrides = {}
        for rule_id, rule_def in self.logic_rules.get("rules", {}).items():
            condition = rule_def.get("condition", {})
            action = rule_def.get("action")
            if jsonLogic(condition, context_dict):
                overrides["action"] = action
                overrides["error_code"] = rule_def.get("error_code")
                return overrides
        return overrides

    def log_state(self, state):
        with open("state.jsonl", "a") as f:
            f.write(json.dumps(state) + "\n")
