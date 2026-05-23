import json
import jsonschema
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ProvenanceHeader(BaseModel):
    git_commit_hash: str = Field(..., description="Git commit hash of the environment")
    timestamp: str = Field(..., description="ISO 8601 timestamp of execution")
    content_hash: str = Field(..., description="SHA-256 hash of the generated content")
    jsonLogic_version: str = Field(..., description="Version of the jsonLogic schema used")
    confidence_score: float = Field(..., description="Agent conviction score (0.0 to 1.0)")
    derivation_path: str = Field(..., description="Path indicating how the conclusion was reached")

class GovernanceError(Exception):
    """Raised when an inference fails governance validation (e.g. invalid schema, missing provenance, poisoned data)."""
    pass

class GovernanceGatekeeper:
    def __init__(self, schema: Dict[str, Any]):
        """
        Initializes the gatekeeper with a specific JSON schema constraint.
        """
        self.schema = schema

    def validate_inference(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates LLM probabilistic inferences natively using jsonschema.
        Ensures the presence of a valid ProvenanceHeader.
        Raises GovernanceError if validation fails or data is poisoned.
        """

        if "provenance_trace" not in inference_output:
            raise GovernanceError("Missing 'provenance_trace' containing the ProvenanceHeader.")

        try:
            # Pydantic validation for the header
            header = ProvenanceHeader(**inference_output["provenance_trace"])
        except Exception as e:
            raise GovernanceError(f"Invalid ProvenanceHeader: {e}")

        # Optional: Poison check based on confidence score boundary issues
        if header.confidence_score < 0.0 or header.confidence_score > 1.0:
             raise GovernanceError("Poisoned data detected: confidence score out of bounds.")

        # Extract the actual data payload to validate against the schema
        payload = inference_output.get("data", {})

        if not payload:
            raise GovernanceError("Missing 'data' payload in inference output.")

        try:
            jsonschema.validate(instance=payload, schema=self.schema)
        except jsonschema.exceptions.ValidationError as e:
            raise GovernanceError(f"Schema validation failed: {e.message}")

        return inference_output
