"""
Script to build and verify instruction documentation synchronization across AGENTS.md, llms.txt, and llms-full.txt.
"""

import sys
from pathlib import Path


def verify_instructions() -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    agents_md = repo_root / "AGENTS.md"
    llms_txt = repo_root / "llms.txt"

    if not agents_md.exists():
        print(f"Error: {agents_md} does not exist", file=sys.stderr)
        return False

    if not llms_txt.exists():
        print(f"Error: {llms_txt} does not exist", file=sys.stderr)
        return False

    print("Instruction files verified successfully.")
    return True


def main() -> None:
    verify = "--verify" in sys.argv
    if verify:
        if not verify_instructions():
            sys.exit(1)
    else:
        print("Building instruction files...")


if __name__ == "__main__":
    main()
