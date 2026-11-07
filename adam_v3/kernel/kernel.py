import functools
import json
from pydantic import BaseModel
from typing import Dict, Any
from json_logic import jsonLogic
import os
from pathlib import Path
from typing import Any, Dict, Optional

from json_logic import jsonLogic
from pydantic import BaseModel

from scripts.sre_eval_harness import AsyncProvOBatchLedger, KernelRPCEnvelope


@functools.lru_cache(maxsize=16)
def _load_logic_rules_cached(resolved_path: str) -> Dict[str, Any]:
    with open(resolved_path, "r", encoding="utf-8") as f:
        return json.load(f)


class Kernel:
    def __init__(self, logic_rules_path: str = "kernel/logic_rules.json"):
        if not os.path.exists(logic_rules_path):
             logic_rules_path = os.path.join(os.path.dirname(__file__), "logic_rules.json")
        with open(logic_rules_path, "r") as f:
            self.logic_rules = json.load(f)
            repo_candidate = Path(__file__).resolve().parent.parent.parent / logic_rules_path
            if repo_candidate.exists():
                logic_rules_path = str(repo_candidate)
            else:
                logic_rules_path = os.path.join(os.path.dirname(__file__), "logic_rules.json")
        self.logic_rules_path = os.path.abspath(logic_rules_path)
        self.logic_rules = _load_logic_rules_cached(self.logic_rules_path)

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
    def log_state(self, state: Dict[str, Any], ledger_path: str = "state.jsonl") -> None:
        with open(ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n")

    async def commit_rpc_envelope(
        self,
        envelope: KernelRPCEnvelope,
        ledger: AsyncProvOBatchLedger,
    ) -> str:
        """
        Commits a validated `KernelRPCEnvelope` through the hardened W3C PROV-O batch ledger
        with multiplicative `Layer 0 x Layer 1` safety gate enforcement.
        """
        return await ledger.commit_event(envelope)

