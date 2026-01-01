from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import uuid

class HMMParser:
    """
    Parses and generates Human-Machine Markdown (HMM) Protocol messages.
    Follows the specification in the Agentic Convergence Whitepaper (Appendix A.2).
    """

    @staticmethod
    def parse_request(text: str) -> Dict[str, Any]:
        """
        Parses an HMM Request text block into a dictionary.

        Expected Format:
        HMM INTERVENTION REQUEST
         * Request ID: <id>
         * Action: <action>
         * Target: <target>
         * Justification: <text>
         * Parameters:
           * key: value
        """
        lines = text.strip().split('\n')
        data = {"parameters": {}}
        mode = "header"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "HMM INTERVENTION REQUEST" in line:
                continue

            if line.startswith("* Parameters:"):
                mode = "params"
                continue

            if mode == "header" and line.startswith("* "):
                key_val = line[2:].split(':', 1)
                if len(key_val) == 2:
                    key = key_val[0].strip().lower().replace(" ", "_")
                    value = key_val[1].strip()
                    data[key] = value

            elif mode == "params" and line.startswith("* "):
                key_val = line[2:].split(':', 1)
                if len(key_val) == 2:
                    key = key_val[0].strip()
                    value = key_val[1].strip()
                    # Try to infer type
                    if value.lower() == "true": value = True
                    elif value.lower() == "false": value = False
                    elif value.replace(",","").replace(".","").isdigit():
                         if "." in value: value = float(value.replace(",",""))
                         else: value = int(value.replace(",",""))
                    data["parameters"][key] = value

        return data

    @staticmethod
    def generate_log(action_taken: str,
                     impact_analysis: Dict[str, Any],
                     audit_link: str,
                     log_id: Optional[str] = None) -> str:
        """
        Generates an HMM Action Log formatted string.
        """
        if not log_id:
            log_id = f"{datetime.now().strftime('%Y-%m-%d')}-AG-{uuid.uuid4().hex[:6].upper()}"

        lines = [
            "HMM ACTION LOG",
            f" * Log ID: {log_id}",
            f" * Action Taken: {action_taken}",
            " * Impact Analysis:"
        ]

        for k, v in impact_analysis.items():
            lines.append(f"   * {k}: {v}")

        lines.append(f" * Audit Trail: Linked to {audit_link}")

        return "\n".join(lines)

    @staticmethod
    def generate_request(action: str, target: str, justification: str, parameters: Dict[str, Any], request_id: Optional[str] = None) -> str:
        """
        Generates an HMM Intervention Request formatted string.
        Useful for agents simulating a request or for UI generation.
        """
        if not request_id:
            request_id = f"{datetime.now().strftime('%Y-%m-%d')}-CR-{uuid.uuid4().hex[:6].upper()}"

        lines = [
            "HMM INTERVENTION REQUEST",
            f" * Request ID: {request_id}",
            f" * Action: {action}",
            f" * Target: {target}",
            f" * Justification: {justification}",
            " * Parameters:"
        ]

        for k, v in parameters.items():
            lines.append(f"   * {k}: {v}")

        return "\n".join(lines)
