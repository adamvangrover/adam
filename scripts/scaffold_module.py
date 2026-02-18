import argparse
import os
import sys

# Constants
TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
AGENT_OUTPUT_DIR = "core/agents"
REPORT_OUTPUT_DIR = "showcase/reports"

def load_template(filename):
    path = os.path.join(TEMPLATE_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: Template {filename} not found in {TEMPLATE_DIR}")
        sys.exit(1)
    with open(path, 'r') as f:
        return f.read()

def scaffold_agent(name, role, description):
    content = load_template("agent_template.py")

    # Simple replace
    content = content.replace("{{agent_class_name}}", name)
    content = content.replace("{{agent_role}}", role)
    content = content.replace("{{agent_description}}", description)

    # Snake case filename
    filename = "".join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_') + ".py"
    output_path = os.path.join(AGENT_OUTPUT_DIR, filename)

    os.makedirs(AGENT_OUTPUT_DIR, exist_ok=True)

    if os.path.exists(output_path):
        print(f"Error: File {output_path} already exists.")
        sys.exit(1)

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Successfully created agent: {output_path}")

def scaffold_report(title, date, summary):
    content = load_template("report_template.html")

    content = content.replace("{{report_title}}", title)
    content = content.replace("{{report_date}}", date)
    content = content.replace("{{executive_summary}}", summary)
    content = content.replace("{{report_content}}", "<p>Generated content goes here.</p>")

    filename = title.lower().replace(" ", "_") + ".html"
    output_path = os.path.join(REPORT_OUTPUT_DIR, filename)

    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)

    if os.path.exists(output_path):
        print(f"Error: File {output_path} already exists.")
        sys.exit(1)

    with open(output_path, 'w') as f:
        f.write(content)

    print(f"Successfully created report: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Scaffold new modules using templates.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Agent Command
    agent_parser = subparsers.add_parser("agent", help="Create a new agent")
    agent_parser.add_argument("--name", required=True, help="Class name of the agent (e.g. FinanceBot)")
    agent_parser.add_argument("--role", default="Assistant", help="Role of the agent")
    agent_parser.add_argument("--desc", default="A new agent.", help="Description")

    # Report Command
    report_parser = subparsers.add_parser("report", help="Create a new report")
    report_parser.add_argument("--title", required=True, help="Report title")
    report_parser.add_argument("--date", required=True, help="Report date")
    report_parser.add_argument("--summary", default="Summary...", help="Executive summary")

    args = parser.parse_args()

    if args.command == "agent":
        scaffold_agent(args.name, args.role, args.desc)
    elif args.command == "report":
        scaffold_report(args.title, args.date, args.summary)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
