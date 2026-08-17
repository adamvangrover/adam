"""
Certificate generation for the ADAM Evaluation Harness.
"""
import uuid
from datetime import datetime
import json

def generate_certificate(evaluator_results: dict, output_path: str) -> dict:
    """
    Generates the final certification object.
    """
    cert = {
        "certificate_id": str(uuid.uuid4()),
        "evaluation_timestamp": datetime.utcnow().isoformat(),
        "certification_status": evaluator_results.get("certification", "FAIL"),
        "critical_failures": evaluator_results.get("critical_failures", 0),
        "metrics": evaluator_results
    }
    with open(output_path, "w") as f:
        json.dump(cert, f, indent=2)
    return cert
