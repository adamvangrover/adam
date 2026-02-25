import sys
import os

# Add repo root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.engine.consensus_engine_v2 import ConsensusEngineV2
from core.swarms.memory_matrix import MemoryMatrix

def run():
    print("Initializing Consensus Engine V2...")

    # Initialize Memory
    memory = MemoryMatrix()

    # Seed if empty for demo purposes
    if not memory.memory_store.get("nodes"):
        print("Memory is empty. Seeding with demo data...")
        memory.write_consensus("AI Technology", "AI infrastructure spend is accelerating faster than anticipated.", "TechAnalyst", 0.9)
        memory.write_consensus("Semiconductors", "Supply constraints are easing, but demand remains robust.", "SupplyChainAgent", 0.8)
        memory.write_consensus("Interest Rates", "Fed signaling a pause, creating a favorable environment for growth.", "MacroEconomist", 0.7)
        memory.write_consensus("Energy", "Oil prices stabilizing, reducing input cost volatility.", "CommoditiesAgent", 0.6)
        memory.write_consensus("Consumer Discretionary", "Retail spending is holding up despite inflation concerns.", "ConsumerAgent", 0.5)
        memory.write_consensus("Commercial Real Estate", "Office vacancy rates remain critically high, posing systemic risk.", "RealEstateAgent", 0.8)

    engine = ConsensusEngineV2(memory_matrix=memory)
    output_path = engine.generate_report()

    print(f"Success! Strategic Command Report generated at: {output_path}")

if __name__ == "__main__":
    run()
