import subprocess
import sys
import shutil

def run_types(paths):
    mypy_cmd = shutil.which("mypy")
    if not mypy_cmd:
        print("mypy not found. Please install it with 'pip install mypy'.")
        return False

    # Use --no-incremental to avoid generating .mypy_cache artifacts that clutter the repo
    cmd = [mypy_cmd, "--ignore-missing-imports", "--follow-imports=skip", "--install-types", "--non-interactive", "--no-incremental"] + paths
    print(f"Running type check: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["core", "services", "scripts", "ops"]
    if run_types(paths):
        print("Type check passed.")
        sys.exit(0)
    else:
        print("Type check failed.")
        sys.exit(1)
