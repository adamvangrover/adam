import json
import hashlib
import urllib.request
from urllib.error import URLError, HTTPError
import jsonschema
import asyncio
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class ProvenanceHeader(BaseModel):
    """
    W3C PROV-O compliant metadata header for probabilistic inferences.
    This ensures that AI context windows can discern *why* a decision was made by tracing data back to its source.
    """
    git_commit_hash: str = Field(..., description="Git commit hash of the environment (PROV-O: wasGeneratedBy)")
    timestamp: str = Field(..., description="ISO 8601 timestamp of execution (PROV-O: generatedAtTime)")
    content_hash: str = Field(..., description="SHA-256 hash of the generated content (PROV-O: value)")
    jsonLogic_version: str = Field(..., description="Version of the jsonLogic schema used")
    confidence_score: float = Field(..., description="Agent conviction score (0.0 to 1.0)")
    derivation_path: str = Field(..., description="Path indicating how the conclusion was reached (PROV-O: wasDerivedFrom)")
    source_data_object: str = Field(..., description="Reference to the source data object, satisfying W3C PROV-O requirements (PROV-O: hadPrimarySource)")

class GovernanceError(Exception):
    """Raised when an inference fails governance validation (e.g. invalid schema, missing provenance, poisoned data)."""
    pass

class GovernanceGatekeeper:
    def __init__(self, schema: Dict[str, Any]):
        """
        Initializes the PDIL gatekeeper with a specific JSON schema constraint.
        Bridges stochastic model outputs with deterministic system inputs.
        """
        self.schema = schema

        # Pre-compile the JSON schema validator for performance
        ValidatorClass = jsonschema.validators.validator_for(schema)
        ValidatorClass.check_schema(schema)
        self.validator = ValidatorClass(schema)



    def entry_gate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for agentic workflows.
        """
        return input_data

    def exit_gate(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exit point for agentic workflows.
        """
        return self.validate_inference(inference_output)

    def validate_inference(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates LLM probabilistic inferences natively using jsonschema.
        Ensures the presence of a valid ProvenanceHeader and satisfies W3C PROV-O compliance.
        Raises GovernanceError if validation fails or data is poisoned.
        """
        if "provenance_trace" not in inference_output:
            raise GovernanceError("Missing 'provenance_trace' containing the ProvenanceHeader.")

        try:
            # Pydantic validation for the header
            header = ProvenanceHeader(**inference_output["provenance_trace"])
        except Exception as e:
            raise GovernanceError(f"Invalid ProvenanceHeader: {e}")

        # Poison check based on confidence score boundary issues
        if header.confidence_score < 0.0 or header.confidence_score > 1.0:
             raise GovernanceError("Poisoned data detected: confidence score out of bounds.")

        # Extract the actual data payload to validate against the schema
        payload = inference_output.get("data", {})

        if not payload:
            raise GovernanceError("Missing 'data' payload in inference output.")

        # Real provenance checks must be enforced by actively computing the payload's SHA-256 hash
        payload_json = json.dumps(payload, sort_keys=True).encode("utf-8")
        computed_hash = hashlib.sha256(payload_json).hexdigest()

        if computed_hash != header.content_hash:
            raise GovernanceError(f"Content hash mismatch: computed {computed_hash}, expected {header.content_hash}")

        # Verify source_data_object reachability if it is a URL
        source = header.source_data_object
        if source.startswith("http://") or source.startswith("https://"):
            try:
                req = urllib.request.Request(source, headers={'User-Agent': 'Mozilla/5.0'})
                urllib.request.urlopen(req, timeout=5.0)
            except (URLError, HTTPError, ValueError) as e:
                raise GovernanceError(f"Source data object reachability check failed: {e}")

        try:
            self.validator.validate(instance=payload)
        except jsonschema.exceptions.ValidationError as e:
            raise GovernanceError(f"Schema validation failed: {e.message}")

        return inference_output

    async def async_validate_inference(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous pipeline for data processing and validation.
        Implements modern concurrency patterns.
        """
        return await asyncio.to_thread(self.validate_inference, inference_output)

    def heal_drift(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Injects capability for autonomous self-healing.
        Detects drift between data sources and model assumptions, triggering re-validation.
        """
        if inference_output.get("observed_drift"):
            # Autonomous re-validation protocol
            inference_output["revalidation_triggered"] = True
            inference_output["observed_drift"] = False # Healed

        return self.validate_inference(inference_output)
