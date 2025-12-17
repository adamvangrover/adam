import sys
import compileall
import os

def check_syntax(paths):
    print(f"Checking syntax for: {paths}")
    success = True
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Path {path} does not exist.")
            continue
        # compile_dir returns True if success
        if not compileall.compile_dir(path, quiet=1, force=True):
            print(f"Syntax error found in {path}")
            success = False
    return success

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else ["core", "services", "scripts", "tests", "ops"]
    if check_syntax(paths):
        print("Syntax check passed.")
        sys.exit(0)
    else:
        print("Syntax check failed.")
        sys.exit(1)
