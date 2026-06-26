import json
import hashlib
import asyncio
import time
import urllib.request
import urllib.error
import asyncio
import socket
import ipaddress
from urllib.parse import urlparse
import jsonschema
from typing import Dict, Any, Optional
from src.pdil.models import ProvenanceHeader

class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

from json_logic import jsonLogic

class GovernanceError(Exception):
    """Raised when an inference fails governance validation."""
    pass

class JsonLogicGovernanceGatekeeper:
    def __init__(self, rules: Dict[str, Any]):
        """
        Initializes the gatekeeper with a specific jsonLogic ruleset constraint.
        Bridges stochastic model outputs with deterministic system inputs using jsonLogic.
        """
        self.rules = rules

    def validate_inference(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validates LLM probabilistic inferences using jsonLogic.
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

        # Extract the actual data payload to validate against the jsonLogic rules
        payload = inference_output.get("data", {})

        if not payload:
            raise GovernanceError("Missing 'data' payload in inference output.")

        # Strict Provenance Checks: Reproducible Hash Validation
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        computed_hash = hashlib.sha256(payload_json).hexdigest()

        if header.content_hash != computed_hash:
            raise GovernanceError(f"Provenance violation: content_hash mismatch. Expected {computed_hash}, got {header.content_hash}")

        # Schema Validation with jsonLogic
        is_valid = jsonLogic(self.rules, payload)
        if not is_valid:
            raise GovernanceError("jsonLogic validation failed: payload does not satisfy rules.")

        return inference_output

    def entry_gate(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_inference(inference_output)

    def exit_gate(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_inference(inference_output)

    async def async_validate_inference(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self.validate_inference, inference_output)

    async def async_validate_inference_batch(self, inferences: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        tasks = [self.async_validate_inference(inference) for inference in inferences]
        return await asyncio.gather(*tasks)

class SecurityGovernanceGatekeeper:
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
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        computed_hash = hashlib.sha256(payload_json).hexdigest()

        if header.content_hash != computed_hash:
            raise GovernanceError(f"Provenance violation: content_hash mismatch. Expected {computed_hash}, got {header.content_hash}")

        # Strict Provenance Checks: Source Data Object Reachability & Whitelisting
        source = header.source_data_object
        if source.startswith("https://"):
            parsed_url = urlparse(source)

            allowed_domains = ["example.com", "api.github.com", "query2.finance.yahoo.com"]
            if parsed_url.hostname not in allowed_domains:
                raise GovernanceError(f"Source data object domain not permitted: {parsed_url.hostname}")

            try:
                ip = socket.gethostbyname(parsed_url.hostname)
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private or ip_obj.is_loopback:
                    raise GovernanceError(f"Source data object resolves to a private IP: {ip}")
            except Exception as e:
                raise GovernanceError(f"IP resolution failed or private IP detected: {e}")

            try:
                opener = urllib.request.build_opener(NoRedirectHandler())
                req = urllib.request.Request(
                    source,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with opener.open(req, timeout=5.0) as response:
                    if response.getcode() >= 400:
                        raise GovernanceError(f"Source data object unreachable: HTTP {response.getcode()}")
            except urllib.error.URLError as e:
                raise GovernanceError(f"Source data object unreachable: {e}")
            except Exception as e:
                 raise GovernanceError(f"Source data object validation failed: {e}")
        elif source.startswith("http://"):
            raise GovernanceError("HTTP is not allowed for source data objects, use HTTPS.")

        # Schema Validation
        try:
            self.validator.validate(instance=payload)
        except jsonschema.exceptions.ValidationError as e:
            raise GovernanceError(f"Schema validation failed: {e.message}")

        return inference_output

    def entry_gate(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_inference(inference_output)

    def exit_gate(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_inference(inference_output)

    async def async_validate_inference(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        return await asyncio.to_thread(self.validate_inference, inference_output)

    async def async_validate_inference_batch(self, inferences: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        tasks = [self.async_validate_inference(inference) for inference in inferences]
        return await asyncio.gather(*tasks)


class DriftIntelligenceLayer:
    def __init__(self, gatekeeper: SecurityGovernanceGatekeeper):
        self.gatekeeper = gatekeeper

    def detect_and_heal_drift(self, inference_output: Dict[str, Any], historical_hash: str) -> Dict[str, Any]:
        current_hash = inference_output.get("provenance_trace", {}).get("content_hash")
        if current_hash != historical_hash:
            inference_output["observed_drift"] = True

        return self.heal_drift(inference_output)

    def heal_drift(self, inference_output: Dict[str, Any]) -> Dict[str, Any]:
        if inference_output.get("observed_drift"):
            inference_output["revalidation_triggered"] = True
            inference_output["observed_drift"] = False

        return self.gatekeeper.validate_inference(inference_output)


class ProofOfThoughtLogger:
    def log_payload(self, payload: Dict[str, Any], derivation_path: str = "unknown", source_data_object: str = "unknown") -> Dict[str, Any]:
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
    def __init__(self):
        self.milestones: list = []

    def add_milestone(self, name: str, details: Dict[str, Any], complexity: float, conviction: float) -> None:
        self.milestones.append({
            "name": name,
            "details": details,
            "complexity": complexity,
            "conviction": conviction,
            "timestamp": time.time()
        })

    def get_most_efficient_process(self) -> Dict[str, Any]:
        if not self.milestones:
            return {}
        return sorted(self.milestones, key=lambda m: m["conviction"] / max(m["complexity"], 0.001), reverse=True)[0]
