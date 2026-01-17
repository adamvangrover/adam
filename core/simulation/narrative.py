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
            "Suppressing dissenting narrative in {sector}. Stability index critical.",
            "Cyber-sovereignty firewall detected {cyber_threat} intrusion attempts. Counter-measures active.",
            "Deploying autonomous monitoring drones to {region} border. Control is absolute."
        ],
        "Market Democracy": [
            "Market volatility high due to {resource} shortage. Proposing stimulus package.",
            "Electorate in {sector} demanding action on {resource}. Polling data suggests shift.",
            "Trade negotiations with {ally} progressing. Tariff adjustment considered.",
            "Inflation at {inflation}%. Central Bank indicates rate hike imminent.",
            "Public outcry over recent {adversary} aggression. Defense spending bill introduced."
        ],
        "Resource State": [
            "Exporting {resource} at premium. Reserves accumulating.",
            "Infrastructure strain in {sector}. Seeking foreign direct investment.",
            "Geopolitical pressure from {adversary} increasing. Mobilizing defense assets.",
            "Nationalizing {resource} extraction assets to secure sovereign wealth.",
            "OPEC+ alignment discussions regarding {resource} output caps."
        ]
    }

    CRISIS_TEMPLATES = [
        "EMERGENCY: {resource} supply chain collapse. Strategic reserves at 10%.",
        "ALERT: Mass migration from {adversary} overwhelming border controls.",
        "DEFCON UPDATE: {adversary} mobilizing kinetic assets near contested zone.",
        "SYSTEM FAILURE: Cyber-attack on banking infrastructure. {inflation}% currency devaluation.",
        "SANCTIONS: Blockade by {adversary} impacting GDP by {gdp_loss}%."
    ]

    @staticmethod
    def generate_thought(sovereign: Sovereign) -> str:
        # Determine context based on state
        is_crisis = sovereign.stability_index < 0.4 or sovereign.economy.inflation_rate > 0.1

        if is_crisis:
            templates = NarrativeEngine.CRISIS_TEMPLATES
        else:
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
            protocol=f"OMEGA-{random.randint(1,9)}",
            resource=resource,
            adoption=int(sovereign.demographics[0].tech_adoption * 100) if sovereign.demographics else 0,
            ally=ally,
            adversary=adversary,
            inflation=f"{sovereign.economy.inflation_rate*100:.1f}",
            cyber_threat="APT-29" if random.random() > 0.5 else "Anonymous",
            region=sovereign.region,
            gdp_loss=f"{abs(sovereign.economy.gdp_growth)*100:.1f}"
        )

        return f"[{sovereign.name}][INTERNAL_MEMO]: {thought}"
