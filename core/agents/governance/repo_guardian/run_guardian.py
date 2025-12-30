#!/usr/bin/env python3
"""
CLI entry point for the RepoGuardian Agent.
Allows running the agent against a local git repo or a simulated PR.
"""
import asyncio
import argparse
import sys
import json
import logging
from typing import Dict, Any

# Ensure we can import core
import os
sys.path.append(os.getcwd())

from core.agents.governance.repo_guardian.agent import RepoGuardianAgent
from core.agents.governance.repo_guardian.schemas import PullRequest, FileDiff

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def main():
    parser = argparse.ArgumentParser(description="Run RepoGuardian Agent")
    parser.add_argument("--diff-file", help="Path to a file containing the diff to review")
    parser.add_argument("--pr-json", help="Path to a JSON file containing the PullRequest object")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")

    args = parser.parse_args()

    agent = RepoGuardianAgent(config={"agent_id": "cli-guardian"})

    if args.pr_json:
        try:
            with open(args.pr_json, 'r') as f:
                data = json.load(f)
            pr = PullRequest(**data)
            logging.info(f"Loaded PR {pr.pr_id} from {args.pr_json}")
        except Exception as e:
            logging.error(f"Failed to load PR JSON: {e}")
            return
    elif args.diff_file:
        # Construct a synthetic PR from a raw diff file
        try:
            with open(args.diff_file, 'r') as f:
                diff_content = f.read()

            pr = PullRequest(
                pr_id="CLI-DIFF",
                author="CLI User",
                title="Manual Diff Review",
                description="Review triggered via CLI diff file.",
                files=[FileDiff(filepath="diff_input", change_type="modify", diff_content=diff_content)]
            )
        except Exception as e:
            logging.error(f"Failed to read diff file: {e}")
            return
    elif args.interactive:
        print("Interactive Mode not fully implemented yet.")
        return
    else:
        # Default test mode
        pr = PullRequest(
            pr_id="TEST-DEFAULT",
            author="System",
            title="Self-Test",
            description="Testing RepoGuardian functionality.",
            files=[
                FileDiff(
                    filepath="test_file.py",
                    change_type="add",
                    diff_content="def hello():\n    print('world')\n"
                )
            ]
        )

    print(f"Running review for PR: {pr.title}...")
    decision = await agent.execute(pr=pr)

    print("\n" + "="*50)
    print(f"DECISION: {decision.status.upper()}")
    print(f"SCORE: {decision.score}/100")
    print("-" * 50)
    print("SUMMARY:")
    print(decision.summary)
    print("-" * 50)
    print("COMMENTS:")
    for comment in decision.comments:
        print(f"[{comment.severity.upper()}] {comment.filepath}:{comment.line_number or '?'} - {comment.message}")
        if comment.suggestion:
            print(f"  Suggestion: {comment.suggestion}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(main())
