import logging

logger = logging.getLogger(__name__)

# Explicitly mapping valid sector agents
SECTOR_MAP = {
    "consumer_discretionary": "ConsumerDiscretionarySpecialist",
    "consumer_staples": "ConsumerStaplesSpecialist",
    "energy": "EnergySpecialist",
    "financials": "FinancialsSpecialist",
    "healthcare": "HealthcareSpecialist",
    "industrials": "IndustrialsSpecialist",
    "materials": "MaterialsSpecialist",
    "real_estate": "RealEstateSpecialist",
    "technology": "TechnologySpecialist",
    "telecommunication_services": "TelecommunicationServicesSpecialist",
    "utilities": "UtilitiesSpecialist",
}

def validate_sector(sector: str) -> str | None:
    """
    Validates a given sector name and returns the corresponding specialist class name.
    """
    if not isinstance(sector, str):
        logger.error(f"Sector must be a string, got {type(sector)}")
        return None

    normalized_sector = sector.lower().strip()
    specialist_class = SECTOR_MAP.get(normalized_sector)

    if not specialist_class:
        logger.warning(f"Invalid or unsupported sector: {sector}. Must be one of: {', '.join(SECTOR_MAP.keys())}")
        return None

    return specialist_class
