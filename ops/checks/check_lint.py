import subprocess
import sys
import shutil

def run_lint(paths):
    flake8_cmd = shutil.which("flake8")
    if not flake8_cmd:
        print("flake8 not found. Please install it with 'pip install flake8'.")
        return False

    cmd = [flake8_cmd] + paths
    print(f"Running lint: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["core", "services", "scripts", "tests", "ops"]
    if run_lint(paths):
        print("Lint check passed.")
        sys.exit(0)
    else:
        print("Lint check failed.")
        sys.exit(1)
