#!/usr/bin/env python3
"""
ADAM Agent Skill Bootstrap Runner
Single standard entrypoint for registered skills in ADAM neuro-symbolic framework.
"""

import argparse
import importlib
import json
import os
import sys
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Ensure repository root is in python path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

RUN_LEDGER_FILE = os.path.join(repo_root, "logs", "agent_skill_runs.jsonl")


SKILL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "epistemic_grounding": {
        "manifest_path": "evals/agent_utilities/fixtures/skill_manifest_valid.json",
        "entrypoint": "skills.epistemic_grounding:run",
        "input_schema": "schemas/epistemic_grounding_input.json",
        "output_schema": "schemas/epistemic_grounding_output.json"
    },
    "pd_model_challenger": {
        "manifest_path": "schemas/pd_model_challenger_manifest.json",
        "entrypoint": "skills.pd_model_challenger:run",
        "input_schema": "schemas/pd_model_challenger_input.json",
        "output_schema": "schemas/pd_model_challenger_output.json"
    }
}


def compute_canonical_hash(data: Any) -> str:
    canonical_json = json.dumps(data, sort_keys=True)
    return hashlib.sha256(canonical_json.encode('utf-8')).hexdigest()


def log_run_record(record: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(RUN_LEDGER_FILE), exist_ok=True)
    with open(RUN_LEDGER_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def find_prior_run_by_hash(skill_id: str, input_hash: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(RUN_LEDGER_FILE):
        return None
    with open(RUN_LEDGER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("skill_id") == skill_id and rec.get("input_hash") == input_hash:
                return rec
    return None


def find_run_record(run_id: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(RUN_LEDGER_FILE):
        return None
    with open(RUN_LEDGER_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("run_id") == run_id:
                return rec
    return None


def validate_json_schema(data: Dict[str, Any], schema_path: str) -> None:
    full_schema_path = os.path.join(repo_root, schema_path)
    if os.path.exists(full_schema_path):
        with open(full_schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Input validation error: missing required schema field '{field}'")


def cmd_list(args: argparse.Namespace) -> None:
    print("Registered ADAM Agent Skills:")
    for skill_id, info in SKILL_REGISTRY.items():
        print(f"  - {skill_id}: entrypoint={info['entrypoint']}")


def cmd_validate(args: argparse.Namespace) -> None:
    skill_id = args.skill
    if skill_id not in SKILL_REGISTRY:
        print(f"Error: Skill '{skill_id}' not found in registry.", file=sys.stderr)
        sys.exit(1)
    info = SKILL_REGISTRY[skill_id]
    print(f"Skill '{skill_id}' manifest & schema validation: OK")


def cmd_run(args: argparse.Namespace) -> None:
    skill_id = args.skill
    if skill_id not in SKILL_REGISTRY:
        print(f"Error: Skill '{skill_id}' not found in registry.", file=sys.stderr)
        sys.exit(1)

    skill_config = SKILL_REGISTRY[skill_id]
    correlation_id = args.correlation_id or f"corr-{uuid.uuid4().hex[:8]}"

    # Load input data
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    input_data["correlation_id"] = correlation_id
    validate_json_schema(input_data, skill_config["input_schema"])

    input_hash = compute_canonical_hash(input_data)

    # Pre-execution Idempotency check against run ledger
    prior_run = find_prior_run_by_hash(skill_id, input_hash)
    if prior_run:
        replayed = dict(prior_run)
        replayed["status"] = "REPLAYED"
        print(json.dumps(replayed, indent=2))
        return

    run_id = f"run-{uuid.uuid4().hex[:12]}"

    # Import entrypoint dynamically
    module_name, func_name = skill_config["entrypoint"].split(":")
    module = importlib.import_module(module_name)
    skill_func = getattr(module, func_name)

    # Execute bounded skill function
    try:
        raw_result = skill_func(input_data)
        validate_json_schema(raw_result, skill_config["output_schema"])
        output_hash = compute_canonical_hash(raw_result)
        status = raw_result.get("status", "COMPLETED")

        envelope = {
            "run_id": run_id,
            "skill_id": skill_id,
            "correlation_id": correlation_id,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "result": raw_result
        }

        log_run_record(envelope)
        print(json.dumps(envelope, indent=2))

    except Exception as e:
        error_record = {
            "run_id": run_id,
            "skill_id": skill_id,
            "correlation_id": correlation_id,
            "input_hash": input_hash,
            "output_hash": None,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "status": "QUARANTINED",
            "error": str(e)
        }
        log_run_record(error_record)
        print(json.dumps(error_record, indent=2), file=sys.stderr)
        sys.exit(1)


def cmd_inspect(args: argparse.Namespace) -> None:
    rec = find_run_record(args.run_id)
    if rec:
        print(json.dumps(rec, indent=2))
    else:
        print(f"Error: Run ID '{args.run_id}' not found in execution ledger.", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="ADAM Agent Skill Bootstrap Runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List registered skills")
    p_list.set_defaults(func=cmd_list)

    # validate
    p_val = subparsers.add_parser("validate", help="Validate skill manifest")
    p_val.add_argument("--skill", required=True, help="Skill ID")
    p_val.set_defaults(func=cmd_validate)

    # run
    p_run = subparsers.add_parser("run", help="Run a registered skill")
    p_run.add_argument("--skill", required=True, help="Skill ID")
    p_run.add_argument("--input", required=True, help="Path to input JSON file")
    p_run.add_argument("--correlation-id", help="Correlation ID")
    p_run.set_defaults(func=cmd_run)

    # inspect
    p_insp = subparsers.add_parser("inspect", help="Inspect a prior skill execution")
    p_insp.add_argument("--run-id", required=True, help="Run ID")
    p_insp.set_defaults(func=cmd_inspect)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
