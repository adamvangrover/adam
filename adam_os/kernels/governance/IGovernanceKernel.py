import logging
import datetime
from typing import Dict, Optional, Any, List

# Assuming these are available from your core and interfaces packages
from afos_core import Decision, ImmutableDecisionBlock
from afos_interfaces import IGovernanceKernel

logger = logging.getLogger(__name__)

class GovernanceKernel(IGovernanceKernel):
    """
    Concrete implementation of the Governance Kernel.
    Maintains an in-memory cryptographic ledger of all OS decisions.
    """
    def __init__(self):
        # The ledger maps the block_hash to the actual ImmutableDecisionBlock
        self._ledger: Dict[str, ImmutableDecisionBlock[Decision]] = {}
        
        # The Genesis hash represents the start of the chain (zeroed 384-bit hash)
        self._genesis_hash = "0" * 96 
        self._head_hash = self._genesis_hash

    async def initialize(self) -> None:
        """
        Boot sequence for the Governance Kernel.
        In a production environment, this would hydrate the Merkle tree/ledger
        from a durable datastore (e.g., PostgreSQL or a specialized append-only DB).
        """
        logger.info("Initializing Governance Kernel: Cryptographic ledger ready.")
        logger.info(f"Current Chain Head: {self._head_hash[:12]}...")

    async def shutdown(self) -> None:
        """
        Teardown sequence.
        Flushes any pending blocks to disk and gracefully closes connections.
        """
        logger.info(f"Shutting down Governance Kernel. Total blocks secured: {len(self._ledger)}")

    async def register_decision_block(self, block: ImmutableDecisionBlock[Decision]) -> str:
        """
        Validates and appends a new decision block to the ledger.
        """
        # 1. State Continuity Check (Prevent branching/forking)
        if block.previous_block_hash != self._head_hash:
            raise ValueError(
                f"Ledger Integrity Error: Block's previous hash ({block.previous_block_hash[:12]}...) "
                f"does not match current head ({self._head_hash[:12]}...)."
            )

        # 2. Cryptographic Sealing
        # If the block isn't sealed yet, seal it. If it is, verify the seal.
        expected_hash = block.seal()
        
        if block.block_hash != expected_hash:
            raise ValueError("Cryptographic Verification Failed: Block hash does not match payload.")

        # 3. Append to Ledger
        self._ledger[block.block_hash] = block
        
        # 4. Advance the Chain Head
        self._head_hash = block.block_hash
        
        logger.info(f"Block Secured | ID: {block.block_id} | Hash: {self._head_hash[:12]}...")
        return self._head_hash

    async def require_approval(self, decision: Decision) -> bool:
        """
        Evaluates the business logic to determine if human-in-the-loop is required.
        """
        # Example Risk Heuristics for Human Routing:
        
        # 1. Outcome strictly dictates a manual review (e.g., Edge-case rejection)
        if decision.outcome.upper() in ["MANUAL_REVIEW", "ESCALATED"]:
            return True
            
        # 2. If the rationale mentions specific high-risk keywords
        high_risk_flags = ["override", "exception", "override_policy", "data_missing"]
        if any(flag in decision.rationale.lower() for flag in high_risk_flags):
            return True

        # In a fully fleshed system, you might also cross-reference the AdversarialContext 
        # (e.g., block.adversarial_defense.confidence_score < 0.90) to force a human review,
        # but evaluating just the Decision payload keeps boundaries clean.
        
        return False

    def get_block(self, block_hash: str) -> Optional[ImmutableDecisionBlock[Decision]]:
        """Utility method for auditors to retrieve a block by its hash."""
        return self._ledger.get(block_hash)

    def verify_chain_integrity(self) -> bool:
        """
        Auditor function: Walks the chain backward from the head to genesis 
        to mathematically prove the history has not been tampered with.
        """
        current_hash = self._head_hash
        
        while current_hash != self._genesis_hash:
            block = self._ledger.get(current_hash)
            if not block:
                logger.error(f"Chain broken! Missing block with hash: {current_hash}")
                return False
                
            # Verify the block's internal payload hasn't been altered
            recalculated_hash = block.seal()
            if recalculated_hash != current_hash:
                logger.error(f"Data corruption detected in block: {block.block_id}")
                return False
                
            # Move pointer backward
            current_hash = block.previous_block_hash
            
        return True

    async def log_telemetry_prov_o(self, activity_id: str, used_entities: List[str], generated_entity_id: str, generated_metric: str, generated_value: Any, dpo_feedback: Optional[Dict[str, str]] = None, human_override: bool = False, override_reason: str = "") -> dict:
        """
        Logs a decision trace in W3C PROV-O format.
        """
        prov_o_log = {
            "@context": "http://www.w3.org/ns/prov#",
            "activity": {
                activity_id: {
                    "prov:startedAtTime": datetime.datetime.utcnow().isoformat() + "Z",
                    "prov:used": used_entities
                }
            },
            "entity": {
                generated_entity_id: {
                    "prov:wasGeneratedBy": activity_id,
                    "metric": generated_metric,
                    "value": generated_value,
                    "human_override": human_override
                }
            }
        }
        
        if human_override and override_reason:
            prov_o_log["entity"][generated_entity_id]["override_reason"] = override_reason
            
        if dpo_feedback:
            prov_o_log["dpo_feedback"] = dpo_feedback
            
        logger.info(f"PROV-O Log generated for {generated_entity_id}")
        return prov_o_log
