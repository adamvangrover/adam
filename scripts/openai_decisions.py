#!/usr/bin/env python3
"""
CLI runner for evaluating inputs using the OpenAI Decisions API (`POST /v1/decisions`).
Usage:
    python scripts/openai_decisions.py --input "Export fails in Safari but works in Chrome." --questions-json '[...]'
    python scripts/openai_decisions.py --request-file request.json --simulate
"""

import argparse
import json
import sys
from typing import Any

from core.llm.engines.openai_decisions_engine import (
    DecisionsRequest,
    OpenAIDecisionsEngine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluations against OpenAI Decisions API (gpt-6-luna)."
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input text string or JSON string representing input content/messages.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to file containing input text or JSON content.",
    )
    parser.add_argument(
        "--questions-json",
        type=str,
        help="JSON string containing questions array.",
    )
    parser.add_argument(
        "--questions-file",
        type=str,
        help="Path to JSON file containing questions array.",
    )
    parser.add_argument(
        "--request-file",
        type=str,
        help="Path to complete JSON file formatted as DecisionsRequest.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-6-luna",
        help="Model name to evaluate (default: gpt-6-luna).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key. If omitted, uses OPENAI_API_KEY environment variable or simulation mode.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Force simulation mode without making live network requests.",
    )
    return parser.parse_args()


def build_request(args: argparse.Namespace) -> DecisionsRequest:
    if args.request_file:
        with open(args.request_file, encoding="utf-8") as f:
            data = json.load(f)
            return DecisionsRequest.model_validate(data)

    input_val: Any = None
    if args.input_file:
        with open(args.input_file, encoding="utf-8") as f:
            content = f.read().strip()
            try:
                input_val = json.loads(content)
            except json.JSONDecodeError:
                input_val = content
    elif args.input:
        try:
            input_val = json.loads(args.input)
        except json.JSONDecodeError:
            input_val = args.input
    else:
        raise ValueError("Must specify either --request-file, --input, or --input-file.")

    questions_val = None
    if args.questions_file:
        with open(args.questions_file, encoding="utf-8") as f:
            questions_val = json.load(f)
    elif args.questions_json:
        questions_val = json.loads(args.questions_json)
    else:
        raise ValueError("Must specify either --request-file, --questions-json, or --questions-file.")

    payload = {
        "model": args.model,
        "input": input_val,
        "questions": questions_val,
    }
    return DecisionsRequest.model_validate(payload)


def main() -> None:
    args = parse_args()
    try:
        req = build_request(args)
    except Exception as e:
        print(f"Error constructing DecisionsRequest: {e}", file=sys.stderr)
        sys.exit(1)

    engine = OpenAIDecisionsEngine(
        api_key=args.api_key,
        default_model=args.model,
        simulation_mode=args.simulate,
    )

    try:
        resp = engine.evaluate_decisions(req)
        print(json.dumps(resp.model_dump(exclude_none=True), indent=2))
    except Exception as e:
        print(f"Error executing decisions evaluation: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
