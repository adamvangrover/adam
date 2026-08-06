from typing import Any, Dict, List, Optional
import json
import logging
from src.schemas.provenance import AgentOutput, ProvenanceHeader

logger = logging.getLogger(__name__)

class GovernanceGatekeeper:
    """
    The PDIL (Probabilistic-to-Deterministic Integration Layer) GovernanceGatekeeper.
    Acts as the strict entry/exit interface separating System 1 (LLMs) from System 2 (deterministic engines).
    """

    def __init__(self, approval_threshold: float = 0.85):
        """
        Initializes the Gatekeeper.

        Args:
            approval_threshold: The minimum confidence score required to bypass human review.
        """
        self.approval_threshold = approval_threshold
        # In a real system, this might be a connection to an immutable ledger (e.g., Temporal Event Store)
        self.ledger: List[Dict[str, Any]] = []

    def validate_agent_output(self, output: AgentOutput) -> bool:
        """
        Strictly validates an AgentOutput payload ensuring schema compliance and provenance.
        """
        try:
            # Pydantic validation handles most checks implicitly during instantiation,
            # but we explicitly check grounding here per W3C PROV-O compliance rules.
            if not output.provenance.source_data_reference:
                logger.error("ProvenanceHeader is missing source_data_reference (Grounding failure).")
                return False

            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

    def require_approval(self, output: AgentOutput) -> bool:
        """
        Determines if an agent decision requires human approval based on policy thresholds.

        Returns:
            True if human approval is required, False otherwise.
        """
        if output.confidence < self.approval_threshold:
            logger.warning(
                f"Confidence ({output.confidence}) below threshold ({self.approval_threshold}). "
                "Routing decision to human review."
            )
            return True
        return False

    def register_decision(self, output: AgentOutput, approved: bool = True) -> Dict[str, Any]:
        """
        Records the agent decision to the immutable ledger.

        Args:
            output: The validated agent output.
            approved: Whether the decision was approved (either auto or manual).

        Returns:
            The ledger entry.
        """
        if not self.validate_agent_output(output):
            raise ValueError("Cannot register an invalid agent output.")

        entry = {
            "hash": output.provenance.hash,
            "git_commit": output.provenance.git_commit_hash,
            "logic_version": output.provenance.jsonLogic_version,
            "source_ref": output.provenance.source_data_reference,
            "decision": output.answer,
            "confidence": output.confidence,
            "approved": approved,
            # In a production scenario, we'd record a timestamp here via datetime.now(timezone.utc)
        }

        self.ledger.append(entry)
        logger.info(f"Decision {entry['hash']} registered to ledger.")
        return entry

    def process_agent_request(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        The main interface for the Swarm to push data into the deterministic execution layers.

        Args:
            raw_data: A raw dictionary that should map to an AgentOutput.

        Returns:
            The registered ledger entry if successful, or None if validation/approval fails.
        """
        try:
            output = AgentOutput(**raw_data)
        except Exception as e:
            logger.error(f"Failed to parse AgentOutput: {e}")
            return None

        if not self.validate_agent_output(output):
            return None

        # Check if it needs human review
        needs_review = self.require_approval(output)

        if needs_review:
            # Simulate a "pending" state or quarantine
            logger.info(f"Decision quarantined for human review. Hash: {output.provenance.hash}")
            # For demonstration, we automatically approve here, but in real life it pauses
            # return None or store in a review queue. We'll register as unapproved.
            return self.register_decision(output, approved=False)

        return self.register_decision(output, approved=True)
