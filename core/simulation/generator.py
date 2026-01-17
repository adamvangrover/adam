import random
import uuid
from typing import List, Dict, Any
from .sovereign import Sovereign, Resource, Military
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
            num_res = random.randint(2, 4)
            for _ in range(num_res):
                r_name = random.choice(resources_list)
                if not any(r.name == r_name for r in res):
                    res.append(Resource(
                        name=r_name,
                        amount=random.uniform(0.5, 2.0), # >1 means surplus
                        strategic_importance=random.random()
                    ))

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

            # Configure Military
            sov.military.readiness = random.uniform(0.4, 0.9)
            sov.military.kinetic_capability = random.uniform(0.1, 0.9)
            sov.military.cyber_capability = random.uniform(0.1, 0.9)
            sov.military.doctrine = "Expansionist" if "Autocracy" in ideology else "Defensive"

            self.sovereigns.append(sov)

        # Establish Nexus (Relationships)
        for sov in self.sovereigns:
            # Pick random allies/adversaries
            others = [s for s in self.sovereigns if s.id != sov.id]
            if others:
                # 2 Allies, 2 Adversaries
                for _ in range(2):
                    if others:
                        ally = random.choice(others)
                        if ally.name not in sov.allies:
                            sov.allies.append(ally.name)

                for _ in range(2):
                    if others:
                        enemy = random.choice(others)
                        if enemy.name not in sov.adversaries:
                            sov.adversaries.append(enemy.name)

    def run_simulation_step(self, step_index: int):
        """
        Run one step of the simulation.
        1. Global Market Update (Prices).
        2. Individual Sovereign Updates (Demographics, Economy, Stability).
        3. Inter-Sovereign Flows (Migration).
        4. Event Generation (Conflict, Trade War).
        5. Narrative Generation.
        """

        # 1. Global Market
        EconomyEngine.simulate_global_market(self.sovereigns)

        # 2. Individual Updates
        for sov in self.sovereigns:
            EconomyEngine.compute_economic_metrics(sov)
            DemographicsEngine.simulate_shift(sov, step_index)
            sov.calculate_stability()

        # 3. Inter-Sovereign Flows
        DemographicsEngine.process_migration(self.sovereigns)

        # 4. Event Generation
        for sov in self.sovereigns:
            events = self._check_events(sov, step_index)
            self.world_events.extend(events)

            # 5. Narrative
            thought = NarrativeEngine.generate_thought(sov)
            self.synthetic_data.append(thought)

    def _check_events(self, sov: Sovereign, step: int) -> List[Dict]:
        events = []

        # Stability Crisis
        if sov.stability_index < 0.25:
            events.append({
                "step": step,
                "type": "Civil Unrest",
                "sovereign": sov.name,
                "details": f"Regime stability critical ({sov.stability_index:.2f}). Riots in capital."
            })

        # Economic Crisis
        if sov.economy.inflation_rate > 0.15:
            events.append({
                "step": step,
                "type": "Hyperinflation",
                "sovereign": sov.name,
                "details": f"Currency collapse. Inflation at {sov.economy.inflation_rate*100:.0f}%."
            })

        # Conflict
        if sov.military.doctrine == "Expansionist" and sov.military.readiness > 0.8:
            # Check for weak neighbor
            for enemy_name in sov.adversaries:
                # Find enemy obj
                enemy = next((s for s in self.sovereigns if s.name == enemy_name), None)
                if enemy and enemy.stability_index < 0.4:
                    events.append({
                        "step": step,
                        "type": "Military Incursion",
                        "sovereign": sov.name,
                        "details": f"Deploying peacekeepers to unstable region: {enemy.name}."
                    })
                    enemy.stability_index -= 0.1 # Impact
                    break

        # Cyber
        if sov.military.cyber_capability > 0.8 and random.random() < 0.1:
             events.append({
                "step": step,
                "type": "Cyber Operation",
                "sovereign": sov.name,
                "details": f"APT group linked to {sov.name} targeting global infrastructure."
            })

        return events

    def generate_full_simulation(self, steps=10):
        self.initialize_world()
        for i in range(steps):
            self.run_simulation_step(i)

        return {
            "sovereigns": [s.dict() for s in self.sovereigns],
            "events": self.world_events,
            "synthetic_data": self.synthetic_data
        }
