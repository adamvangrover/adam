#!/usr/bin/env python3
import json
import sys
import time

from experimental.nexus_aurora.simulation import NexusOrchestrator


def main():
    print("🚀 NEXUS-ZERO: MAXIMUM-CAPACITY EXECUTION INITIATED")
    print("===================================================")

    start_time = time.time()

    # Initialize Orchestrator
    nexus = NexusOrchestrator()

    # Run a multi-cycle simulation
    # Cycle 1: Basic
    # Cycle 2: Advanced
    # Cycle 3: Maximum Complexity
    try:
        synthesis = nexus.run_simulation(iterations=3)

        end_time = time.time()
        duration = end_time - start_time

        print("\n🧩 NEXUS-ZERO FINAL SYNTHESIS")
        print("=============================")
        print(f"Execution Time: {duration:.4f}s")
        print(f"Total Cycles: {synthesis['cycles_run']}")
        print(f"Compiler Log Size: {synthesis['final_log_size']}")

        print("\n--- Detailed Simulation Log ---")
        print(json.dumps(synthesis['simulation_log'], indent=2))

        print("\n✅ NEXUS-AURORA SYSTEMS NOMINAL.")
        print("Refined constraints held. Speculative universes collapsed safely.")

    except Exception as e:
        print(f"\n❌ CRITICAL FAILURE IN NEXUS-ZERO: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
