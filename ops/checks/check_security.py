import shutil
import subprocess
import sys


def run_security(paths):
    bandit_cmd = shutil.which("bandit")
    if not bandit_cmd:
        print("bandit not found. Please install it with 'pip install bandit'.")
        return False

    # -r: recursive
    # -ll: severity level (medium) - avoiding Low which has too many false positives usually
    cmd = [bandit_cmd, "-r", "-ll", "-x", "tests,venv,env,.venv,node_modules"] + paths
    print(f"Running security check: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["core", "services", "scripts", "ops"]
    if run_security(paths):
        print("Security check passed.")
        sys.exit(0)
    else:
        print("Security check failed.")
        sys.exit(1)
