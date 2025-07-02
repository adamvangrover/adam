import csv
import os
from typing import List, Dict, Any, Optional
# Assuming models.py is in backend/src/main/python and this script can import from there
# This might require adjusting PYTHONPATH or project structure for actual execution
# For this environment, I'll duplicate simplified model definitions or assume it's accessible.

# Simplified model definitions for this script if direct import is an issue in this environment:
from dataclasses import dataclass, field
from enum import Enum

class DriverType(Enum):
    MACROECONOMIC = "Macroeconomic"
    FUNDAMENTAL = "Fundamental"
    TECHNICAL = "Technical"
    GEOPOLITICAL = "Geopolitical"
    COMPANY_SPECIFIC = "CompanySpecific"
    INDUSTRY_SPECIFIC = "IndustrySpecific"
    # Add any other types from CSV if necessary
    UNKNOWN = "Unknown" # Default for parsing robustness

class ImpactPotential(Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    UNKNOWN = "Unknown"

class TimeHorizon(Enum):
    SHORT_TERM = "Short-term"
    MEDIUM_TERM = "Medium-term"
    LONG_TERM = "Long-term"
    VARIES = "Varies"
    UNKNOWN = "Unknown"

class Trend(Enum):
    INCREASING = "Increasing"
    DECREASING = "Decreasing"
    STABLE = "Stable"
    IMPROVING = "Improving" # from sample
    ELEVATED = "Elevated" # from sample
    UNKNOWN = "Unknown"


@dataclass
class Industry:
    id: str
    name: str
    description: Optional[str] = None
    # In a real KG, these would be populated after all nodes are loaded
    # For now, we can store ids from link tables if needed later
    linked_driver_ids: List[str] = field(default_factory=list)


@dataclass
class TradingLevelData:
    price: Optional[float] = None
    volume: Optional[int] = None
    volatility: Optional[float] = None
    timestamp: Optional[str] = None # ISO format date string, or datetime object


@dataclass
class Company:
    id: str
    name: str
    industry_id: str
    ownership_structure: Optional[str] = None
    corporate_structure: Optional[str] = None
    financials: Dict[str, Any] = field(default_factory=dict)
    trading_levels: Optional[TradingLevelData] = None
    # Similar to Industry, for drivers
    linked_driver_ids: List[str] = field(default_factory=list)


@dataclass
class Driver:
    id: str
    name: str
    description: str
    type: DriverType
    impact_potential: Optional[ImpactPotential] = None
    time_horizon: Optional[TimeHorizon] = None
    metrics: Dict[str, str] = field(default_factory=dict)
    related_macro_factor_ids: List[str] = field(default_factory=list)


@dataclass
class MacroEnvironmentFactor:
    id: str
    name: str
    current_value: Optional[Any] = None
    trend: Optional[Trend] = None
    impact_narrative: Optional[str] = None
    source: Optional[str] = None

# Helper to safely get enum members
def get_enum_member(enum_class, value_str, default_member=None):
    try:
        return enum_class(value_str)
    except ValueError:
        if default_member:
            return default_member
        # Attempt case-insensitive match or partial match if needed, or raise error
        for member in enum_class:
            if member.name.lower() == value_str.lower():
                return member
        if default_member: return default_member
        raise ValueError(f"'{value_str}' is not a valid member of {enum_class.__name__}")


DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'sample_data')

def load_csv(filename: str) -> List[Dict[str, str]]:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return []
    with open(filepath, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        return [row for row in reader]

def process_data():
    raw_industries = load_csv('industries.csv')
    raw_companies = load_csv('companies.csv')
    raw_drivers = load_csv('drivers.csv')
    raw_macro_factors = load_csv('macro_factors.csv')
    raw_company_drivers_links = load_csv('company_drivers_link.csv')
    raw_industry_drivers_links = load_csv('industry_drivers_link.csv')

    industries: Dict[str, Industry] = {}
    for row in raw_industries:
        ind = Industry(id=row['id'], name=row['name'], description=row.get('description'))
        industries[ind.id] = ind

    companies: Dict[str, Company] = {}
    for row in raw_companies:
        financials = {
            'revenue_mn_usd': float(row['revenue_mn_usd']) if row.get('revenue_mn_usd') else None,
            'pe_ratio': float(row['pe_ratio']) if row.get('pe_ratio') else None,
            'debt_to_equity_ratio': float(row['debt_to_equity_ratio']) if row.get('debt_to_equity_ratio') else None,
        }
        trading_levels = TradingLevelData(
            price=float(row['current_price']) if row.get('current_price') else None,
            volume=int(row['current_volume']) if row.get('current_volume') else None,
            volatility=float(row['current_volatility_30d']) if row.get('current_volatility_30d') else None,
            # timestamp=datetime.now().isoformat() # Or from data if available
        )
        comp = Company(
            id=row['id'],
            name=row['name'],
            industry_id=row['industryId'],
            ownership_structure=row.get('ownershipStructure'),
            corporate_structure=row.get('corporateStructure'),
            financials=financials,
            trading_levels=trading_levels
        )
        companies[comp.id] = comp

    drivers: Dict[str, Driver] = {}
    for row in raw_drivers:
        metrics = {}
        if row.get('metric_name_1') and row.get('metric_value_1'):
            metrics[row['metric_name_1']] = row['metric_value_1']
        if row.get('metric_name_2') and row.get('metric_value_2'):
            metrics[row['metric_name_2']] = row['metric_value_2']

        related_macros = []
        if row.get('relatedMacroFactorId_1'):
            related_macros.append(row['relatedMacroFactorId_1'])
        # Add more if columns like relatedMacroFactorId_2 exist

        drv = Driver(
            id=row['id'],
            name=row['name'],
            description=row.get('description',''),
            type=get_enum_member(DriverType, row['type'], DriverType.UNKNOWN),
            impact_potential=get_enum_member(ImpactPotential, row.get('impactPotential',''), ImpactPotential.UNKNOWN) if row.get('impactPotential') else None,
            time_horizon=get_enum_member(TimeHorizon, row.get('timeHorizon',''), TimeHorizon.UNKNOWN) if row.get('timeHorizon') else None,
            metrics=metrics,
            related_macro_factor_ids=related_macros
        )
        drivers[drv.id] = drv

    macro_factors: Dict[str, MacroEnvironmentFactor] = {}
    for row in raw_macro_factors:
        factor = MacroEnvironmentFactor(
            id=row['id'],
            name=row['name'],
            current_value=row.get('currentValue'),
            trend=get_enum_member(Trend, row.get('trend',''), Trend.UNKNOWN) if row.get('trend') else None,
            impact_narrative=row.get('impactNarrative'),
            source=row.get('source')
        )
        macro_factors[factor.id] = factor

    # Populate link information (simplified)
    for link in raw_company_drivers_links:
        company_id = link['companyId']
        driver_id = link['driverId']
        if company_id in companies and driver_id in drivers:
            companies[company_id].linked_driver_ids.append(driver_id)
            # In a real KG, you'd create an edge here.
            # If the driver is 'CompanySpecific', it could be directly on the company object.
            # For now, this is a simple list of IDs.

    for link in raw_industry_drivers_links:
        industry_id = link['industryId']
        driver_id = link['driverId']
        if industry_id in industries and driver_id in drivers:
            industries[industry_id].linked_driver_ids.append(driver_id)


    # Output (for demonstration)
    print("--- Industries ---")
    for ind_id, ind_obj in industries.items():
        print(ind_obj)

    print("\n--- Companies ---")
    for comp_id, comp_obj in companies.items():
        print(comp_obj)

    print("\n--- Drivers ---")
    for drv_id, drv_obj in drivers.items():
        print(drv_obj)

    print("\n--- Macro Factors ---")
    for mf_id, mf_obj in macro_factors.items():
        print(mf_obj)

    # This data can now be used to populate a knowledge graph or other database
    # For example, one could iterate through these dicts and create nodes and edges.
    # nodes = []
    # edges = []
    # for item in list(industries.values()) + list(companies.values()) + list(drivers.values()) + list(macro_factors.values()):
    #     nodes.append({'id': item.id, 'label': item.__class__.__name__, 'properties': item.__dict__})
    # # Edges would be created from industry_id in company, linked_driver_ids, etc.

    return {
        "industries": industries,
        "companies": companies,
        "drivers": drivers,
        "macro_factors": macro_factors
    }

if __name__ == '__main__':
    loaded_data = process_data()
    print(f"\nLoaded {len(loaded_data['industries'])} industries, "
          f"{len(loaded_data['companies'])} companies, "
          f"{len(loaded_data['drivers'])} drivers, "
          f"{len(loaded_data['macro_factors'])} macro_factors.")
    # In a real application, you'd pass this data to a KG loading module.
    # e.g., knowledge_graph_loader.load_data(loaded_data)
