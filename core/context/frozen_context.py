import hashlib
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class FrozenContext(BaseModel):
    context_id: str
    created_at: str
    source_ids: List[str] = Field(default_factory=list)
    document_hashes: List[str] = Field(default_factory=list)
    market_snapshot_id: str
    portfolio_snapshot_id: str
    policy_version: str
    model_version: str
    code_revision: str
    retrieval_manifest: Dict[str, Any] = Field(default_factory=dict)
    feature_manifest: Dict[str, Any] = Field(default_factory=dict)
    expires_at: Optional[str] = None

    @property
    def context_hash(self) -> str:
        """
        Cryptographically binds the execution state.
        This provides zero-context-drift guarantees.
        """
        canonical_state = {
            "source_ids": sorted(self.source_ids),
            "document_hashes": sorted(self.document_hashes),
            "market_snapshot_id": self.market_snapshot_id,
            "portfolio_snapshot_id": self.portfolio_snapshot_id,
            "policy_version": self.policy_version,
            "code_revision": self.code_revision
        }
        state_str = json.dumps(canonical_state, sort_keys=True)
        return hashlib.sha256(state_str.encode('utf-8')).hexdigest()

    def verify_hash(self, provided_hash: str) -> bool:
        return self.context_hash == provided_hash
