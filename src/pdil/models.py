# Updated by Jules to meet professional engineering standards.

import json
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

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

    def hash_state(self) -> str:
        """
        Generates an immutable SHA-256 hash of the git_commit_hash and jsonLogic_version.
        """
        data = f"{self.git_commit_hash}:{self.jsonLogic_version}".encode('utf-8')
        return hashlib.sha256(data).hexdigest()

def build_provenance_header(
    data: Dict[str, Any],
    git_commit_hash: str = "HEAD",
    timestamp: Optional[str] = None,
    jsonLogic_version: str = "1.0.0",
    confidence_score: float = 1.0,
    derivation_path: str = "direct_inference",
    source_data_object: str = "https://example.com/source_data",
) -> Dict[str, Any]:
    """
    Constructs a W3C PROV-O compliant ProvenanceHeader dict from data.
    Computes the SHA-256 hash automatically from canonical JSON serialization.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    payload_json = json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')
    computed_hash = hashlib.sha256(payload_json).hexdigest()

    header = ProvenanceHeader(
        git_commit_hash=git_commit_hash,
        timestamp=timestamp,
        content_hash=computed_hash,
        jsonLogic_version=jsonLogic_version,
        confidence_score=confidence_score,
        derivation_path=derivation_path,
        source_data_object=source_data_object,
    )
    return header.model_dump()

__all__ = ["ProvenanceHeader", "build_provenance_header"]
