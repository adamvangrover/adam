import random
from typing import List
from .sovereign import PopulationSegment, Sovereign

class DemographicsEngine:
    @staticmethod
    def generate_profile(region_type: str) -> List[PopulationSegment]:
        segments = []
        if region_type == "Advanced Economy":
            segments.append(PopulationSegment(label="Aging Traditionalists", percentage=0.4, sentiment=0.6, tech_adoption=0.3))
            segments.append(PopulationSegment(label="Digital Natives", percentage=0.3, sentiment=0.4, tech_adoption=0.9))
            segments.append(PopulationSegment(label="Urban Professional", percentage=0.3, sentiment=0.5, tech_adoption=0.8))
        elif region_type == "Emerging Market":
            segments.append(PopulationSegment(label="Youth Bulge", percentage=0.6, sentiment=0.3, tech_adoption=0.7))
            segments.append(PopulationSegment(label="Rural Traditional", percentage=0.3, sentiment=0.8, tech_adoption=0.2))
            segments.append(PopulationSegment(label="Elite", percentage=0.1, sentiment=0.9, tech_adoption=0.9))
        else:
            # Default
            segments.append(PopulationSegment(label="General Public", percentage=1.0, sentiment=0.5, tech_adoption=0.5))

        return segments

    @staticmethod
    def simulate_shift(sovereign: Sovereign, time_step: int):
        """
        Updates internal demographic shifts:
        - Sentiment drift
        - Tech adoption
        - Economic strain impact
        """
        for seg in sovereign.demographics:
            # 1. Random flux in sentiment
            seg.sentiment += random.uniform(-0.02, 0.02)

            # 2. Tech adoption S-curve
            seg.tech_adoption += (1 - seg.tech_adoption) * 0.05

            # 3. Economic Strain impact
            # High inflation/unemployment -> High strain -> Lower sentiment
            econ = sovereign.economy
            strain = (econ.inflation_rate * 5) + (econ.unemployment_rate * 2)
            seg.economic_strain = min(1.0, max(0.0, strain))

            if seg.economic_strain > 0.6:
                # Radicalization
                seg.sentiment -= 0.05
            elif seg.economic_strain < 0.2:
                # Satisfaction
                seg.sentiment += 0.02

            seg.sentiment = max(0.0, min(1.0, seg.sentiment))

    @staticmethod
    def process_migration(sovereigns: List[Sovereign]):
        """
        Moves population from unstable/poor sovereigns to stable/rich ones.
        """
        # Sort by 'desirability' (Stability + Income)
        # We'll use a simplified model where bottom 20% lose population to top 20%

        sorted_sovs = sorted(sovereigns, key=lambda s: s.stability_index - s.economy.unemployment_rate)

        num_migrants = len(sovereigns) // 5
        if num_migrants < 1:
            return

        losers = sorted_sovs[:num_migrants]
        gainers = sorted_sovs[-num_migrants:]

        for loser in losers:
            for gainer in gainers:
                # Transfer logic
                # Reduce percentage of a random segment in loser
                # We don't actually move the 'segment object', we just adjust local percentages
                # In a real simulation, we'd model diaspora, but here we just simulate brain drain/labor flight.

                # Drain from 'Youth' or 'Professional' if possible
                source_seg = next((s for s in loser.demographics if "Youth" in s.label or "Professional" in s.label), loser.demographics[0])

                flow_rate = 0.005 # 0.5% population shift per step

                if source_seg.percentage > flow_rate:
                    source_seg.percentage -= flow_rate

                    # Add to gainer (distributed across segments or new 'Migrant' segment)
                    # For simplicity, add to their 'General' or similar
                    target_seg = next((s for s in gainer.demographics if "Urban" in s.label or "Youth" in s.label), gainer.demographics[0])
                    target_seg.percentage += flow_rate

                    # Economic impact
                    # Loser loses GDP potential (brain drain)
                    loser.economy.gdp_growth -= 0.001
                    # Gainer gains labor but maybe social friction
                    gainer.economy.gdp_growth += 0.001
