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
try:
    import jsonschema
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    class _FallbackValidationError(Exception):
        def __init__(self, message: str):
            super().__init__(message)
            self.message = message

    class _FallbackExceptions:
        ValidationError = _FallbackValidationError

    class _FallbackValidator:
        def __init__(self, schema: Dict[str, Any]):
            self.schema = schema

        @classmethod
        def check_schema(cls, schema: Dict[str, Any]) -> None:
            if not isinstance(schema, dict):
                raise ValueError("Schema must be a dict")

        def validate(self, instance: Any) -> None:
            if self.schema.get("type") == "object" and not isinstance(instance, dict):
                raise _FallbackValidationError("Expected object instance")
            for req in self.schema.get("required", []):
                if isinstance(instance, dict) and req not in instance:
                    raise _FallbackValidationError(f"Missing required property: {req}")

    class _FallbackValidators:
        @staticmethod
        def validator_for(schema: Dict[str, Any]):
            return _FallbackValidator

    class _FallbackJsonSchema:
        exceptions = _FallbackExceptions
        validators = _FallbackValidators

    jsonschema = _FallbackJsonSchema()


class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


try:
    from json_logic import jsonLogic
except ImportError:
    def _resolve_var(path: Any, data: Any, default: Any = None) -> Any:
        if path is None or path == "":
            return data
        parts = str(path).split(".")
        curr = data
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            elif isinstance(curr, list):
                try:
                    curr = curr[int(part)]
                except (ValueError, IndexError):
                    return default
            else:
                return default
        return curr

    def _to_number(val: Any) -> float:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return float(val)
        return float(val)

    def jsonLogic(tests: Any, data: Any = None) -> Any:
        """Deterministic fallback jsonLogic evaluator for minimal Python environments."""
        if tests is None or not isinstance(tests, dict):
            return tests
        if data is None:
            data = {}
        if not tests:
            return True

        op = next(iter(tests))
        values = tests[op]
        if not isinstance(values, list):
            values = [values]

        if op == "var":
            default = jsonLogic(values[1], data) if len(values) > 1 else None
            path = jsonLogic(values[0], data) if values else ""
            return _resolve_var(path, data, default)

        if op in ("if", "?:"):
            for i in range(0, len(values) - 1, 2):
                if jsonLogic(values[i], data):
                    return jsonLogic(values[i + 1], data)
            if len(values) % 2 == 1:
                return jsonLogic(values[-1], data)
            return None

        if op == "and":
            res: Any = True
            for v in values:
                res = jsonLogic(v, data)
                if not res:
                    return res
            return res

        if op == "or":
            res = False
            for v in values:
                res = jsonLogic(v, data)
                if res:
                    return res
            return res

        evals = [jsonLogic(v, data) for v in values]

        if op == "!":
            return not bool(evals[0]) if evals else True
        if op == "!!":
            return bool(evals[0]) if evals else False
        if op == "==":
            return evals[0] == evals[1]
        if op == "===":
            return evals[0] == evals[1] and type(evals[0]) is type(evals[1])
        if op == "!=":
            return evals[0] != evals[1]
        if op == "!==":
            return not (evals[0] == evals[1] and type(evals[0]) is type(evals[1]))
        if op == "<":
            return all(evals[i] < evals[i + 1] for i in range(len(evals) - 1))
        if op == "<=":
            return all(evals[i] <= evals[i + 1] for i in range(len(evals) - 1))
        if op == ">":
            return all(evals[i] > evals[i + 1] for i in range(len(evals) - 1))
        if op == ">=":
            return all(evals[i] >= evals[i + 1] for i in range(len(evals) - 1))
        if op == "+":
            return sum(_to_number(v) for v in evals)
        if op == "-":
            if len(evals) == 1:
                return -_to_number(evals[0])
            return _to_number(evals[0]) - _to_number(evals[1])
        if op == "*":
            prod = 1.0
            for v in evals:
                prod *= _to_number(v)
            return prod
        if op == "/":
            return _to_number(evals[0]) / _to_number(evals[1])
        if op == "%":
            return _to_number(evals[0]) % _to_number(evals[1])
        if op == "min":
            return min(_to_number(v) for v in evals)
        if op == "max":
            return max(_to_number(v) for v in evals)
        if op == "in":
            container = evals[1]
            return evals[0] in container if container is not None else False

        raise ValueError(f"Unsupported jsonLogic operator: {op}")

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

__all__ = [
    "GovernanceError",
    "JsonLogicGovernanceGatekeeper",
    "SecurityGovernanceGatekeeper",
    "DriftIntelligenceLayer",
    "ProofOfThoughtLogger",
    "MilestoneLogger",
]
