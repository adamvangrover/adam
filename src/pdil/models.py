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

__all__ = ["ProvenanceHeader"]
