import random
import uuid
from typing import List, Dict, Any
from .sovereign import Sovereign, Resource
from .demographics import DemographicsEngine
from .economy import EconomyEngine
from .narrative import NarrativeEngine

class SimulationGenerator:
    def __init__(self, num_sovereigns=10):
        self.num_sovereigns = num_sovereigns
        self.sovereigns: List[Sovereign] = []
        self.world_events: List[Dict] = []
        self.synthetic_data: List[str] = []

    def initialize_world(self):
        ideologies = ["Digital Autocracy", "Market Democracy", "Resource State", "Theocratic Cyber-State"]
        regions = ["NorthAm", "EuroZone", "AsiaPac", "MENA", "LatAm", "SubSahara"]
        resources_list = ["Rare Earths", "Water", "Oil", "Compute", "Semiconductors", "Grain"]

        for i in range(self.num_sovereigns):
            ideology = random.choice(ideologies)
            region = random.choice(regions)

            # Generate resources
            res = []
            for _ in range(random.randint(1, 3)):
                res.append(Resource(name=random.choice(resources_list), amount=random.random()))

            # Generate demographics
            demographics = DemographicsEngine.generate_profile("Advanced Economy" if ideology == "Market Democracy" else "Emerging Market")

            sov = Sovereign(
                id=str(uuid.uuid4()),
                name=f"{region}-{ideology.split(' ')[0]}-{i}",
                region=region,
                ideology=ideology,
                resources=res,
                demographics=demographics
            )
            self.sovereigns.append(sov)

        # Establish Nexus (Relationships)
        for sov in self.sovereigns:
            # Pick random allies/adversaries
            others = [s for s in self.sovereigns if s.id != sov.id]
            if others:
                sov.allies.append(random.choice(others).name)
                sov.adversaries.append(random.choice(others).name)

    def run_simulation_step(self, step_index: int):
        """
        Run one step of the simulation.
        1. Update demographics/stability.
        2. Generate events.
        3. Generate synthetic narrative (LLM training data).
        """
        for sov in self.sovereigns:
            # 1. Update
            DemographicsEngine.simulate_shift(sov.demographics, step_index)
            stability = sov.calculate_stability()

            # 2. Event
            if stability < 0.3:
                self.world_events.append({
                    "step": step_index,
                    "type": "Civil Unrest",
                    "sovereign": sov.name,
                    "details": f"Stability dropped to {stability:.2f}. Protests in {sov.demographics[0].label} sector."
                })
            elif stability > 0.8:
                self.world_events.append({
                    "step": step_index,
                    "type": "Golden Age",
                    "sovereign": sov.name,
                    "details": "High stability achieving cultural dominance."
                })

            # 3. Narrative
            thought = NarrativeEngine.generate_thought(sov)
            self.synthetic_data.append(thought)

    def generate_full_simulation(self, steps=10):
        self.initialize_world()
        for i in range(steps):
            self.run_simulation_step(i)

        return {
            "sovereigns": [s.dict() for s in self.sovereigns],
            "events": self.world_events,
            "synthetic_data": self.synthetic_data
        }
