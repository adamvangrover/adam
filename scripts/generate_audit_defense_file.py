import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Any

AUDIT_DIR = "core/libraries_and_archives/audit_trails"
REPORT_DIR = "core/libraries_and_archives/reports"
REPORT_FILE = os.path.join(REPORT_DIR, "Audit_Defense_File.md")

def load_audit_logs(directory: str) -> List[Dict[str, Any]]:
    logs = []
    pattern = os.path.join(directory, "*.json")
    for filepath in glob.glob(pattern):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Handle both list of logs and single PROV object
                if isinstance(data, list):
                    logs.extend(data)
                else:
                    logs.append(data)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
    return logs

def generate_markdown_report(logs: List[Dict[str, Any]]) -> str:
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    md = f"# Audit Defense File\n\n"
    md += f"**Generated:** {timestamp}\n"
    md += f"**Total Records:** {len(logs)}\n\n"

    md += "## Executive Summary\n"
    md += "This document serves as an automated audit trail for AI-driven credit risk decisions. "
    md += "It aggregates provenance logs, compliance checks, and agent reasoning traces.\n\n"

    # Stats
    agents = {}
    outcomes = {}

    for log in logs:
        # Normalize fields
        agent = log.get('agent_id', log.get('agent', 'Unknown'))
        outcome = log.get('outcome', {})
        if isinstance(outcome, dict):
             status = outcome.get('status', 'Unknown')
        else:
             status = str(outcome)

        agents[agent] = agents.get(agent, 0) + 1
        outcomes[status] = outcomes.get(status, 0) + 1

    md += "### Activity Statistics\n"
    md += "| Metric | Count |\n|---|---|\n"
    for agent, count in agents.items():
        md += f"| Agent: {agent} | {count} |\n"
    for status, count in outcomes.items():
        md += f"| Status: {status} | {count} |\n"

    md += "\n## Detailed Decision Logs\n"

    # Sort by timestamp if available
    def get_time(item):
        return item.get('timestamp', '')

    logs.sort(key=get_time, reverse=True)

    for log in logs:
        ts = log.get('timestamp', 'N/A')
        record_id = log.get('record_id', 'N/A')
        agent = log.get('agent_id', 'Unknown')
        activity = log.get('activity_type', log.get('event_type', 'Action'))

        md += f"### Record: {record_id}\n"
        md += f"- **Timestamp:** {ts}\n"
        md += f"- **Agent:** {agent}\n"
        md += f"- **Activity:** {activity}\n"

        # Details
        details = log.get('entity_context', log.get('details', {}))
        md += f"- **Context/Details:**\n"
        md += "```json\n"
        md += json.dumps(details, indent=2)
        md += "\n```\n"

        # Outcome
        outcome = log.get('outcome', {})

        # Specialized Formatting for Risk Agents
        if isinstance(outcome, dict):
            if "sensitivity_table_markdown" in outcome:
                md += "- **Sensitivity Analysis (Black Swan):**\n\n"
                md += outcome["sensitivity_table_markdown"] + "\n\n"

            if "simulation_results" in outcome:
                results = outcome.get("simulation_results", {})
                if isinstance(results, dict) and "results" in results:
                    metrics = results.get("results", {})
                    md += "- **Quantum Simulation Results:**\n"
                    md += f"  - **Expected Value:** {metrics.get('expected_value'):.2f}\n"
                    md += f"  - **VaR 99%:** {metrics.get('var_99'):.2f}\n"
                    md += f"  - **Confidence Interval:** {metrics.get('confidence_interval')}\n\n"

        md += f"- **Outcome (Raw):**\n"
        md += "```json\n"
        md += json.dumps(outcome, indent=2)
        md += "\n```\n"

        md += "---\n"

    return md

def main():
    print(f"Scanning {AUDIT_DIR}...")
    logs = load_audit_logs(AUDIT_DIR)

    if not logs:
        print("No audit logs found.")
        # Create dummy log for demonstration if empty
        logs = [{
            "record_id": "DUMMY-001",
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": "SNCAnalystAgent",
            "activity_type": "ComplianceCheck",
            "details": {"passed": False, "violations": ["Leverage Ratio > 3.0"]},
            "outcome": "Downgrade to Substandard"
        }]

    report = generate_markdown_report(logs)

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        f.write(report)

    print(f"Report generated at {REPORT_FILE}")

if __name__ == "__main__":
    main()
