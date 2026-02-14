import subprocess
import time
import sys
import os

def main():
    print("=== Simulating Runtime Seed Prompt Injection ===")
    print("This script demonstrates how to inject a seed prompt into the running Adam system.")
    print("Note: This simulation runs the interactive console and pipes input to it.")

    # Simulate a user session
    # We send "exit" at the end to close gracefully
    input_sequence = [
        "Analyze AAPL",
        "/seed You are a skeptical auditor looking for fraud.",
        "Analyze AAPL again",
        "exit"
    ]

    # Ensure PYTHONPATH includes current directory so "core" can be imported
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()

    process = subprocess.Popen(
        [sys.executable, "core/main.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=0  # Unbuffered
    )

    try:
        # We need to read output while writing input to avoid deadlock if buffer fills up
        # However, for simple test, communicate is easier if the process finishes.
        # But communicate sends all input at once. We want step-by-step.

        # Let's try communicate with the full input string
        full_input = "\n".join(input_sequence) + "\n"
        print(f"Sending Input:\n{full_input}")

        stdout, stderr = process.communicate(input=full_input, timeout=15)

        print("\n[SYSTEM OUTPUT LOG]")
        print(stdout)

        if stderr:
            print("\n[SYSTEM ERROR LOG]")
            print(stderr)

        if "Runtime Seed Injected" in stdout:
            print("\n[SUCCESS] Runtime Seed Injection Verified.")
        else:
            print("\n[FAILURE] Runtime Seed Injection NOT detected in output.")

    except subprocess.TimeoutExpired:
        print("Timeout Expired. Killing process.")
        process.kill()
        stdout, stderr = process.communicate()
        print(stdout)
        print(stderr)
    except Exception as e:
        print(f"Error running simulation: {e}")
        process.kill()

if __name__ == "__main__":
    main()
