import asyncio
import logging
import json
from datetime import datetime, timezone

# ==================================================================
# CORE & INTERFACES IMPORTS
# ==================================================================
from afos_core import (
    Event, Policy, Decision, Evidence, AssuredDependency, 
    AdversarialContext, ImmutableDecisionBlock
)

# ==================================================================
# KERNEL IMPORTS
# ==================================================================
from knowledge_kernel import KnowledgeKernel
from policy_kernel import PolicyKernel
from simulation_kernel import SimulationKernel
from integration_kernel import IntegrationKernel
from decision_kernel import DecisionKernel
from execution_kernel import ExecutionKernel
from governance_kernel import GovernanceKernel

logger = logging.getLogger("ADAM_OS_RUNTIME")

# ==================================================================
# THE AFOS RUNTIME BOOTSTRAPPER (v30.1)
# ==================================================================
class AdamOSRuntime:
    """
    The central multi-agent financial operating system framework.
    Manages the lifecycle, memory, and cryptographic integrity of all kernels.
    """
    def __init__(self):
        self.state = "OFFLINE"
        
        # Instantiate Kernels
        self.knowledge = KnowledgeKernel()
        self.policy = PolicyKernel()
        self.simulation = SimulationKernel()
        self.integration = IntegrationKernel()
        self.decision = DecisionKernel()
        self.execution = ExecutionKernel()
        self.governance = GovernanceKernel()

    async def boot(self) -> None:
        """
        Executes the strict deterministic boot sequence.
        Dependency order is critical: Memory -> Rules -> Compute -> I/O -> Ledger.
        """
        logger.info("==================================================")
        logger.info("🚀 INITIATING ADAM OS v30.1 BOOT SEQUENCE")
        logger.info("==================================================")
        
        self.state = "BOOTING"
        
        # 1. Foundation: Institutional Memory & State
        await self.knowledge.initialize()
        
        # 2. Logic: Deterministic Business Rules
        await self.policy.initialize()
        
        # 3. Quantitative Engine: Stress Testing & VaR
        await self.simulation.initialize()
        
        # 4. Neuro-Symbolic Engine: AI / Logic Synthesis
        await self.decision.initialize()
        
        # 5. Orchestration: DAG Workflow Engine
        await self.execution.initialize()
        
        # 6. I/O: External Signal Bus
        await self.integration.initialize()
        
        # 7. Integrity: Cryptographic Ledger
        await self.governance.initialize()
        
        self.state = "ONLINE"
        logger.info("==================================================")
        logger.info("✅ ADAM OS v30.1 ONLINE - SIMULATION 0 ACTIVE")
        logger.info("==================================================")

    async def halt(self) -> None:
        """
        Executes a graceful teardown, draining queues and sealing ledgers.
        Reverse order of boot sequence.
        """
        logger.info("==================================================")
        logger.info("🛑 INITIATING ADAM OS HALT SEQUENCE")
        logger.info("==================================================")
        
        self.state = "SHUTTING_DOWN"
        
        await self.integration.shutdown()
        await self.execution.shutdown()
        await self.decision.shutdown()
        await self.simulation.shutdown()
        await self.policy.shutdown()
        await self.knowledge.shutdown()
        
        # Shut down ledger last to ensure all final state changes are recorded
        await self.governance.shutdown()
        
        self.state = "OFFLINE"
        logger.info("✅ ADAM OS OFFLINE. ALL SYSTEMS SECURED.")

# ==================================================================
# FULL LIFECYCLE EXECUTION TEST (SIMULATION 0)
# ==================================================================
if __name__ == "__main__":
    # Configure institutional-grade logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)-15s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    async def run_simulation_zero():
        os_runtime = AdamOSRuntime()
        
        # 1. Boot the OS
        await os_runtime.boot()
        
        try:
            print("\n" + "="*60)
            print("▶️ COMMENCING CORE SYSTEM WORKLOAD (CREDIT RISK DECISIONING)")
            print("="*60 + "\n")
            
            # ---------------------------------------------------------
            # STEP 1: Compile Institutional Policy
            # ---------------------------------------------------------
            logger.info("[STEP 1] Compiling strict underwriting covenant policy...")
            covenant_dsl = json.dumps({
                "and": [
                    {"<=": [{"var": "financials.leverage"}, 4.25]},
                    {">=": [{"var": "financials.interest_coverage"}, 2.5]}
                ]
            })
            active_policy = await os_runtime.policy.compile_policy(covenant_dsl)
            
            # ---------------------------------------------------------
            # STEP 2: Ingest External Telemetry (Market Signal)
            # ---------------------------------------------------------
            logger.info("[STEP 2] Ingesting Q3 Earnings Data via Integration Bus...")
            raw_signal = {
                "event_type": "QuarterlyFinancialUpdate",
                "entity_id": "org_tmt_alpha",
                "financials": {"leverage": 4.10, "interest_coverage": 3.1}
            }
            signal_event = await os_runtime.integration.ingest_signal("cap_iq_feed", raw_signal)
            
            # ---------------------------------------------------------
            # STEP 3: Retrieve Institutional Memory
            # ---------------------------------------------------------
            logger.info("[STEP 3] Fetching semantic and topological risk memory...")
            entity_id = signal_event.payload["sanitized_data"]["entity_id"]
            
            # Retrieve semantic vector data (simulated Qdrant)
            vector_memory = await os_runtime.knowledge.ask_similarity([0.5, -0.2, 0.8, 0.1], limit=1)
            
            # Construct Cryptographic Evidence Arrays
            structured_evidence = Evidence(
                id="evd_struct_01",
                source_uri="sql://core_banking/financials/org_tmt_alpha",
                source="integration_bus",
                hash=signal_event.payload["provenance_hash"],
                data=signal_event.payload["sanitized_data"]
            )
            
            unstructured_evidence = Evidence(
                id="evd_vector_01",
                source_uri="qdrant://vector_store/memos",
                source="knowledge_kernel",
                hash="vec_hash_8899",
                data={"semantic_insight": vector_memory[0]["content"], "sentiment_score": 0.98}
            )

            # ---------------------------------------------------------
            # STEP 4: Neuro-Symbolic Synthesis
            # ---------------------------------------------------------
            logger.info("[STEP 4] Synthesizing Decision Graph via multi-agent consensus...")
            decision = await os_runtime.decision.compute_decision(
                target_id=entity_id,
                policy=active_policy,
                evidence=[structured_evidence, unstructured_evidence]
            )
            
            # ---------------------------------------------------------
            # STEP 5: Cryptographic Sealing (Source of Truth)
            # ---------------------------------------------------------
            logger.info("[STEP 5] Wrapping outcome in Immutable Decision Block & Sealing...")
            
            # Record the state of the system dependencies at execution time
            model_dep = AssuredDependency(
                dependency_id="sentinel_credit_swarm",
                version="v30.1",
                source_uri="github.com/adamvangrover/adam",
                expected_sha384="a4b3c2d1e5f6..."
            )
            
            adv_context = AdversarialContext(
                model_id="sentinel_credit_swarm",
                confidence_score=0.99,
                perturbation_bound=0.005,
                entropy_score=1.2
            )
            
            decision_block = ImmutableDecisionBlock[Decision](
                block_id=f"blk_{decision.id}",
                previous_block_hash=os_runtime.governance._head_hash,
                dependencies=[model_dep],
                adversarial_defense=adv_context,
                payload=decision
            )
            
            final_hash = await os_runtime.governance.register_decision_block(decision_block)
            
            print("\n" + "="*60)
            print(f"💎 SYSTEMIC SOURCE OF TRUTH SECURED")
            print(f"Block Hash: {final_hash}")
            print(f"Outcome:    {decision.outcome}")
            print(f"Rationale:  {decision.rationale}")
            print("="*60 + "\n")
            
        finally:
            # 2. Safely Halt the OS
            await asyncio.sleep(1.0) # Allow async queues to settle
            await os_runtime.halt()

    # Execute the OS runtime loop
    asyncio.run(run_simulation_zero())
