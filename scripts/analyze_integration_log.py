import json
import os
import sys

# Add the current directory to sys.path to ensure we can import from report_generation
# assuming report_generation.py is in the same directory (scripts/)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

try:
    from report_generation import generate_integration_report
except ImportError:
    # If running from root, maybe we need to adjust import
    sys.path.append(os.path.join(os.getcwd(), 'scripts'))
    from report_generation import generate_integration_report

def main():
    # Define paths relative to the repo root
    # If run from scripts/, repo_root is ..
    # If run from root, repo_root is .

    # Let's trust absolute paths based on __file__
    repo_root = os.path.dirname(current_dir)
    log_path = os.path.join(repo_root, 'data', 'v23_integration_log.json')
    output_dir = os.path.join(repo_root, 'docs', 'reports')
    output_path = os.path.join(output_dir, 'v23_integration_log_report.md')

    print(f"Reading log from: {log_path}")

    if not os.path.exists(log_path):
        print(f"Error: Log file not found at {log_path}")
        return

    try:
        with open(log_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Generate the report
    report = generate_integration_report(data)

    # Print to console
    print("Generated Report Preview:")
    print("=========================")
    print(report)
    print("=========================")

    # Save to file
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"Report successfully saved to {output_path}")
    except OSError as e:
        print(f"Error writing report to file: {e}")

if __name__ == "__main__":
    main()
