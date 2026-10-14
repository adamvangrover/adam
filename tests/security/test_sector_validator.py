import pytest
from core.security.sector_validator import validate_sector, SECTOR_MAP
from core.agents.industry_specialist_agent import IndustrySpecialistAgent

def test_validate_sector_valid():
    assert validate_sector("technology") == "TechnologySpecialist"
    assert validate_sector("Technology") == "TechnologySpecialist"
    assert validate_sector(" TECHNOLOGY ") == "TechnologySpecialist"
    assert validate_sector("consumer_discretionary") == "ConsumerDiscretionarySpecialist"

def test_validate_sector_all_mapped():
    for sector, expected_class in SECTOR_MAP.items():
        assert validate_sector(sector) == expected_class

def test_validate_sector_invalid():
    assert validate_sector("invalid_sector") is None
    assert validate_sector("../malicious_path") is None
    assert validate_sector("sys") is None

def test_validate_sector_non_string():
    assert validate_sector(None) is None
    assert validate_sector(123) is None
    assert validate_sector(["technology"]) is None

def test_industry_specialist_agent_loading():
    # Valid sector
    agent = IndustrySpecialistAgent({"sector": "technology"})
    assert agent.specialist is not None
    assert agent.specialist.__class__.__name__ == "TechnologySpecialist"

    # Invalid sector traversal attempt / untrusted input
    invalid_agent = IndustrySpecialistAgent({"sector": "../security"})
    assert invalid_agent.specialist is None
