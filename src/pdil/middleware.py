import json
import hashlib
import asyncio
import time
import urllib.request
import urllib.error
import asyncio
import socket
import ipaddress
import ssl
from urllib.parse import urlparse
import jsonschema
from typing import Dict, Any, Optional
from src.pdil.models import ProvenanceHeader

class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

from json_logic import jsonLogic

class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None

class GovernanceError(Exception):
    """Raised when an inference fails governance validation."""
    def __init__(self, message: str, provenance: Dict[str, Any] = None):
        if provenance:
            message = f"{message} | Erroneous input provenance: {provenance}"
        super().__init__(message)
        self.provenance = provenance

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
            raise GovernanceError("Missing 'provenance_trace' containing the ProvenanceHeader.", provenance={})

        try:
            # Pydantic validation for the header
            header = ProvenanceHeader(**inference_output["provenance_trace"])
        except Exception as e:
            raise GovernanceError(f"Invalid ProvenanceHeader: {e}", provenance=inference_output.get("provenance_trace"))

        # Poison check based on confidence score boundary issues
        if header.confidence_score < 0.5 or header.confidence_score > 1.0:
             raise GovernanceError("Poisoned data detected: confidence score out of bounds. Must be >= 0.5", provenance=inference_output.get("provenance_trace"))

        # Extract the actual data payload to validate against the jsonLogic rules
        payload = inference_output.get("data", {})

        if not payload:
            raise GovernanceError("Missing 'data' payload in inference output.", provenance=inference_output.get("provenance_trace"))

        # Strict Provenance Checks: Reproducible Hash Validation
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        computed_hash = hashlib.sha256(payload_json).hexdigest()

        if header.content_hash != computed_hash:
            raise GovernanceError(f"Provenance violation: content_hash mismatch. Expected {computed_hash}, got {header.content_hash}", provenance=inference_output.get("provenance_trace"))

        # Schema Validation with jsonLogic
        is_valid = jsonLogic(self.rules, payload)
        if not is_valid:
            raise GovernanceError("jsonLogic validation failed: payload does not satisfy rules.", provenance=inference_output.get("provenance_trace"))

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
            raise GovernanceError("Missing 'provenance_trace' containing the ProvenanceHeader.", provenance={})

        try:
            # Pydantic validation for the header
            header = ProvenanceHeader(**inference_output["provenance_trace"])
        except Exception as e:
            raise GovernanceError(f"Invalid ProvenanceHeader: {e}", provenance=inference_output.get("provenance_trace"))

        # Poison check based on confidence score boundary issues
        if header.confidence_score < 0.5 or header.confidence_score > 1.0:
             raise GovernanceError("Poisoned data detected: confidence score out of bounds. Must be >= 0.5", provenance=inference_output.get("provenance_trace"))

        # Extract the actual data payload to validate against the schema
        payload = inference_output.get("data", {})

        if not payload:
            raise GovernanceError("Missing 'data' payload in inference output.", provenance=inference_output.get("provenance_trace"))

        # Strict Provenance Checks: Reproducible Hash Validation
        payload_json = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
        computed_hash = hashlib.sha256(payload_json).hexdigest()

        if header.content_hash != computed_hash:
            raise GovernanceError(f"Provenance violation: content_hash mismatch. Expected {computed_hash}, got {header.content_hash}", provenance=inference_output.get("provenance_trace"))

        # Strict Provenance Checks: Source Data Object Reachability & Whitelisting
        source = header.source_data_object
        if source.startswith("https://"):
            parsed_url = urlparse(source)

            allowed_domains = ["example.com", "api.github.com", "query2.finance.yahoo.com"]
            if parsed_url.hostname not in allowed_domains:
                raise GovernanceError(f"Source data object domain not permitted: {parsed_url.hostname}", provenance=inference_output.get("provenance_trace"))

            try:
                ip = socket.gethostbyname(parsed_url.hostname)
                ip_obj = ipaddress.ip_address(ip)
                if ip_obj.is_private or ip_obj.is_loopback:
                    raise GovernanceError(f"Source data object resolves to a private IP: {ip}", provenance=inference_output.get("provenance_trace"))
            except Exception as e:
                raise GovernanceError(f"IP resolution failed or private IP detected: {e}", provenance=inference_output.get("provenance_trace"))

            try:
                req = urllib.request.Request(
                    source,
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                context = ssl.create_default_context()
                opener = urllib.request.build_opener(
                    NoRedirectHandler(),
                    urllib.request.HTTPSHandler(context=context)
                )
                with opener.open(req, timeout=5.0) as response:  # nosec B310
                    if response.getcode() >= 400:
                        raise GovernanceError(f"Source data object unreachable: HTTP {response.getcode()}", provenance=inference_output.get("provenance_trace"))
            except urllib.error.URLError as e:
                raise GovernanceError(f"Source data object unreachable: {e}", provenance=inference_output.get("provenance_trace"))
            except Exception as e:
                 raise GovernanceError(f"Source data object validation failed: {e}", provenance=inference_output.get("provenance_trace"))
        elif source.startswith("http://"):
            raise GovernanceError("HTTP is not allowed for source data objects, use HTTPS.", provenance=inference_output.get("provenance_trace"))

        # Schema Validation
        try:
            self.validator.validate(instance=payload)
        except jsonschema.exceptions.ValidationError as e:
            raise GovernanceError(f"Schema validation failed: {e.message}", provenance=inference_output.get("provenance_trace"))

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

class RustPDILBridgeGatekeeper:
    """
    Python wrapper bridging RustPDILGatekeeper with zero-copy JCS evaluation
    and pure-Python fallback if rust_ext is unavailable.
    """
    def __init__(self, min_conviction: float = 0.85, max_vpin: float = 0.82, min_dscr: float = 1.05):
        self.min_conviction = min_conviction
        self.max_vpin = max_vpin
        self.min_dscr = min_dscr
        self._rust_gate = None
        try:
            import rust_ext  # type: ignore[import-not-found]
            if hasattr(rust_ext, "RustPDILGatekeeper"):
                self._rust_gate = rust_ext.RustPDILGatekeeper(min_conviction, max_vpin, min_dscr)
        except Exception:
            self._rust_gate = None

    def evaluate(
        self,
        raw_json_bytes: bytes,
        jev_conviction: float,
        vpin_toxicity: float,
        stressed_dscr: float,
        cedar_forbid_triggered: bool,
        raw_quality_score: float,
    ) -> tuple[int, float, str, str]:
        if self._rust_gate is not None:
            return self._rust_gate.evaluate_gate(
                raw_json_bytes,
                jev_conviction,
                vpin_toxicity,
                stressed_dscr,
                cedar_forbid_triggered,
                raw_quality_score,
            )

        # Pure-Python fallback matching rust_ext logic
        parsed = json.loads(raw_json_bytes.decode("utf-8"))
        canonical_str = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        sha256_hex = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        if cedar_forbid_triggered:
            return 0, 0.0, sha256_hex, "BLOCK_CEDAR_FORBID"
        if vpin_toxicity > self.max_vpin:
            return 0, 0.0, sha256_hex, "BLOCK_CBF_VPIN_TOXICITY"
        if stressed_dscr < self.min_dscr:
            return 0, 0.0, sha256_hex, "BLOCK_DSCR_FLOOR_BREACH"
        if jev_conviction < self.min_conviction:
            return 0, 0.0, sha256_hex, "ABSTAIN_JEV_LOW_CONVICTION"

        final_score = 1.0 * min(5.0, max(0.0, raw_quality_score))
        if final_score >= 4.0:
            route = "COMMIT_CANDIDATE"
        elif final_score >= 3.0:
            route = "HUMAN_ESCALATE"
        else:
            route = "BLOCK_AND_LOG"
        return 1, final_score, sha256_hex, route


__all__ = [
    "GovernanceError",
    "JsonLogicGovernanceGatekeeper",
    "SecurityGovernanceGatekeeper",
    "DriftIntelligenceLayer",
    "ProofOfThoughtLogger",
    "MilestoneLogger",
    "RustPDILBridgeGatekeeper",
]
