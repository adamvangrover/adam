import random
from typing import List
from .sovereign import PopulationSegment

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
    def simulate_shift(segments: List[PopulationSegment], time_step: int):
        # Evolve demographics over time
        for seg in segments:
            # Random flux
            seg.sentiment += random.uniform(-0.05, 0.05)
            seg.sentiment = max(0.0, min(1.0, seg.sentiment))

            # Tech adoption S-curve simulation
            seg.tech_adoption += (1 - seg.tech_adoption) * 0.05
