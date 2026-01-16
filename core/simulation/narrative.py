import random
from .sovereign import Sovereign

class NarrativeEngine:
    """
    Generates synthetic 'thoughts' or 'internal monologues' for Sovereign entities
    to serve as LLM training data.
    """

    TEMPLATES = {
        "Digital Autocracy": [
            "Surveillance metrics indicate {sentiment} sentiment in sector {sector}. Initiating protocol {protocol}.",
            "Resource allocation for {resource} optimized. Central bank digital currency adoption at {adoption}%.",
            "Suppressing dissenting narrative in {sector}. Stability index critical."
        ],
        "Market Democracy": [
            "Market volatility high due to {resource} shortage. proposing stimulus package.",
            "Electorate in {sector} demanding action on {resource}. Polling data suggests shift.",
            "Trade negotiations with {ally} progressing. Tariff adjustment considered."
        ],
        "Resource State": [
            "Exporting {resource} at premium. Reserves accumulating.",
            "Infrastructure strain in {sector}. Seeking foreign direct investment.",
            "Geopolitical pressure from {adversary} increasing. Mobilizing defense assets."
        ]
    }

    @staticmethod
    def generate_thought(sovereign: Sovereign) -> str:
        template_key = sovereign.ideology if sovereign.ideology in NarrativeEngine.TEMPLATES else "Market Democracy"
        templates = NarrativeEngine.TEMPLATES.get(template_key, NarrativeEngine.TEMPLATES["Market Democracy"])

        template = random.choice(templates)

        # Fill context
        sector = sovereign.demographics[0].label if sovereign.demographics else "General"
        resource = sovereign.resources[0].name if sovereign.resources else "Capital"
        ally = sovereign.allies[0] if sovereign.allies else "Global North"
        adversary = sovereign.adversaries[0] if sovereign.adversaries else "Insurgents"

        thought = template.format(
            sentiment="declining" if sovereign.stability_index < 0.5 else "stable",
            sector=sector,
            protocol="OMEGA-9",
            resource=resource,
            adoption=int(sovereign.demographics[0].tech_adoption * 100) if sovereign.demographics else 0,
            ally=ally,
            adversary=adversary
        )

        return f"[{sovereign.name}][INTERNAL_MEMO]: {thought}"
