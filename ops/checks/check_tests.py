import subprocess
import sys

def run_tests(paths):
    # Use sys.executable to ensure we use the same python environment
    cmd = [sys.executable, "-m", "pytest"] + paths
    print(f"Running tests: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["tests/"]
    if run_tests(paths):
        print("Tests passed.")
        sys.exit(0)
    else:
        print("Tests failed.")
        sys.exit(1)
