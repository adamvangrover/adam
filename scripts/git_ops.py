import sys
import os
import subprocess
import json

# Ensure core is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from core.utils.system_logger import SystemLogger

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/git_ops.py <git_command> [args...]")
        sys.exit(1)

    git_command = sys.argv[1]
    args = sys.argv[2:]

    logger = SystemLogger()

    # Determine tag
    tag = "GIT_OP"
    if git_command in ["push"]:
        tag = "PUSH"
    elif git_command in ["pull"]:
        tag = "PULL"
    elif git_command in ["branch", "checkout", "switch"]:
        tag = "BRANCH"
    elif git_command in ["commit"]:
        tag = "CREATION" # Commit is creation of a snapshot
    elif git_command in ["rm"]:
        tag = "DELETION"
    elif git_command in ["add"]:
        tag = "CREATION"

    # Log event
    logger.log_event(tag, {
        "command": git_command,
        "args": args,
        "cwd": os.getcwd()
    })

    # Execute git
    try:
        subprocess.check_call(["git", git_command] + args)
    except subprocess.CalledProcessError as e:
        logger.log_event("GIT_ERROR", {
            "command": git_command,
            "error": str(e)
        })
        sys.exit(e.returncode)
    except FileNotFoundError:
        print("Error: 'git' command not found.")
        sys.exit(1)

if __name__ == "__main__":
    main()
