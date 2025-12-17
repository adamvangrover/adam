import subprocess
import sys
import shutil

def run_tests(paths):
    pytest_cmd = shutil.which("pytest")
    if not pytest_cmd:
        print("pytest not found. Please install it with 'pip install pytest'.")
        return False

    cmd = [pytest_cmd] + paths
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
