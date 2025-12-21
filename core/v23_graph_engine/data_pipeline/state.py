from typing import TypedDict, Optional, Any, Dict, List
from dataclasses import dataclass


class DataIngestionState(TypedDict):
    """
    Represents the state of a single file as it moves through the
    Sequential Agent Data Pipeline (ADK Pattern #2).
    """
    file_path: str
    raw_content: Optional[str]
    cleaned_content: Optional[Any]  # Str or Dict or List
    artifact_type: Optional[str]
    metadata: Dict[str, Any]

    # Verification State
    verification_status: str  # "pending", "verified", "rejected"
    verification_errors: List[str]

    # Final Output
    formatted_artifact: Optional[Dict[str, Any]]  # GoldStandardArtifact as dict

    # Pipeline Control
    pipeline_status: str  # "processing", "success", "failed"
    error_message: Optional[str]
