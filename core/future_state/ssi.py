from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import uuid
import hashlib

class VerifiableCredential(BaseModel):
    """
    A W3C-style Verifiable Credential.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    issuer: str
    subject_id: str
    claims: Dict[str, str]
    proof: str = "mock_zkp_signature"

    def verify(self) -> bool:
        # Mock verification logic
        return self.proof == "mock_zkp_signature"

class SoulboundToken(BaseModel):
    """
    Non-transferable reputation token.
    """
    token_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    owner_did: str
    reputation_score: int = 10

class DigitalIdentity(BaseModel):
    """
    Self-Sovereign Identity (DID) container.
    """
    did: str = Field(default_factory=lambda: f"did:adam:{uuid.uuid4()}")
    credentials: List[VerifiableCredential] = []
    soulbound_tokens: List[SoulboundToken] = []
    is_verified_human: bool = False

    def add_credential(self, cred: VerifiableCredential):
        if cred.subject_id == self.did and cred.verify():
            self.credentials.append(cred)
            if "is_human" in cred.claims and cred.claims["is_human"] == "true":
                self.is_verified_human = True

    def get_reputation(self) -> int:
        return sum(token.reputation_score for token in self.soulbound_tokens)

    def sign_transaction(self, data: str) -> str:
        """Mock signing of a transaction with the DID."""
        return hashlib.sha256(f"{self.did}:{data}".encode()).hexdigest()
