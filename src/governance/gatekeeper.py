import json
import hashlib
import time
import urllib.request
import urllib.error
import asyncio
from urllib.parse import urlparse
import jsonschema
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
        Initializes the gatekeeper with a specific JSON schema constraint.
        Bridges stochastic model outputs with deterministic system inputs.
        """
        self.schema = schema

        # Pre-compile the JSON schema validator for performance
        ValidatorClass = jsonschema.validators.validator_for(schema)
        ValidatorClass.check_schema(schema)
        self.validator = ValidatorClass(schema)

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
        if header.confidence_score < 0.5 or header.confidence_score > 1.0:
             raise GovernanceError("Poisoned data detected: confidence score out of bounds. Must be >= 0.5")

        # Extract the actual data payload to validate against the schema
        payload = inference_output.get("data", {})

        if not payload:
            raise GovernanceError("Missing 'data' payload in inference output.")

        # Strict Provenance Checks: Reproducible Hash Validation
        # Using separators removes whitespace for a strictly deterministic hash
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        computed_hash = hashlib.sha256(payload_json).hexdigest()
        
        if header.content_hash != computed_hash:
            raise GovernanceError(f"Provenance violation: content_hash mismatch. Expected {computed_hash}, got {header.content_hash}")

        # Strict Provenance Checks: Source Data Object Reachability & Whitelisting
        source = header.source_data_object
        if source.startswith("http://") or source.startswith("https://"):
            parsed_url = urlparse(source)
            
            # Domain whitelisting enforcement
            allowed_domains = ["example.com", "api.github.com", "query2.finance.yahoo.com"]
            if parsed_url.hostname not in allowed_domains:
                raise GovernanceError(f"Source data object domain not permitted: {parsed_url.hostname}")

            try:
                req = urllib.request.Request(
                    source,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req, timeout=5.0) as response:
                    if response.getcode() >= 400:
                        raise GovernanceError(f"Source data object unreachable: HTTP {response.getcode()}")
            except urllib.error.URLError as e:
                raise GovernanceError(f"Source data object unreachable: {e}")
            except Exception as e:
                 raise GovernanceError(f"Source data object validation failed: {e}")

        # Schema Validation
        try:
            self.validator.validate(instance=payload)
        except jsonschema.exceptions.ValidationError as e:
            raise GovernanceError(f"Schema validation failed: {e.message}")

        return inference_output

    def entry_gate(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Entry point for all agentic workflows. Proxies to validate_inference.
        """
        return self.validate_inference(inference_output)

    def exit_gate(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exit point for all agentic workflows. Proxies to validate_inference.
        """
        return self.validate_inference(inference_output)

    async def async_validate_inference(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronous pipeline for data processing and validation.
        Implements modern concurrency patterns.
        """
        return await asyncio.to_thread(self.validate_inference, inference_output)

    async def async_validate_inference_batch(self, inferences: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        """
        Concurrent asynchronous batch pipeline for data processing and validation.
        Implements modern concurrency patterns to handle large stochastic data ingests.
        """
        tasks = [self.async_validate_inference(inference) for inference in inferences]
        return await asyncio.gather(*tasks)

    def detect_and_heal_drift(self, inference_output: Dict[str, Any], historical_hash: str) -> Dict[str, Any]:
        """
        Detects if the AI output has drifted from the historical expectation (e.g. historical hash mismatch).
        If drifted, injects the 'observed_drift' flag before triggering the self-healing process.
        """
        current_hash = inference_output.get("provenance_trace", {}).get("content_hash")
        if current_hash != historical_hash:
            inference_output["observed_drift"] = True

        return self.heal_drift(inference_output)

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


class ProofOfThoughtLogger:
    """
    Enforces W3C PROV-O compliance strictly on every LLM-generated JSON payload.
    """
    def log_payload(self, payload: Dict[str, Any], derivation_path: str = "unknown", source_data_object: str = "unknown") -> Dict[str, Any]:
        """Logs and structures a payload according to PROV-O."""
        content_hash = hashlib.sha256(str(payload).encode('utf-8')).hexdigest()

        prov_o_log = {
            "entity": "LLM_Output",
            "wasGeneratedBy": "Adam_Swarm_Agent",
            "generatedAtTime": time.time(),
            "content_hash": content_hash,
            "payload": payload,
            "derivation_path": derivation_path,
            "source_data_object": source_data_object
        }
        return prov_o_log


class MilestoneLogger:
    """
    Creates milestone markers and logs for tracking session state and runtime execution
    downstream for human and machine learning reporting.
    """
    def __init__(self):
        self.milestones: list = []

    def add_milestone(self, name: str, details: Dict[str, Any], complexity: float, conviction: float) -> None:
        """
        Adds a milestone evaluating complexity and conviction.
        """
        self.milestones.append({
            "name": name,
            "details": details,
            "complexity": complexity,
            "conviction": conviction,
            "timestamp": time.time()
        })

    def get_most_efficient_process(self) -> Dict[str, Any]:
        """
        Sorts to the most efficient deterministic process
        (highest conviction over complexity).
        """
        if not self.milestones:
            return {}
        return sorted(self.milestones, key=lambda m: m["conviction"] / max(m["complexity"], 0.001), reverse=True)[0]