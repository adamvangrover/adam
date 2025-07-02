import csv
import os
from typing import Dict, Any, Optional # Added Optional
# Assuming models.py and knowledge_graph.py are in the same directory or PYTHONPATH is set
from .knowledge_graph import KnowledgeGraph, Node, Edge # Use . for relative import
from .models import DriverType, ImpactPotential, TimeHorizon, Trend # Use . for relative import

# Simplified version of get_enum_member for brevity
def get_enum_member(enum_class, value_str, default_member):
    try:
        return enum_class(value_str)
    except ValueError:
        for member in enum_class: # Case-insensitive fallback
            if member.name.lower() == value_str.lower(): return member
        return default_member


# Path to the data directory, assuming this script is in backend/src/main/python
# Adjust if your execution context is different.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'sample_data')

def load_csv_to_dicts(filename: str) -> list[dict[str, str]]:
    filepath = os.path.join(DATA_DIR, filename)
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} not found.")
        return []
    with open(filepath, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        return [row for row in reader]

def build_graph_from_csvs(kg: KnowledgeGraph) -> None:
    # Load raw data
    raw_industries = load_csv_to_dicts('industries.csv')
    raw_companies = load_csv_to_dicts('companies.csv')
    raw_drivers = load_csv_to_dicts('drivers.csv')
    raw_macro_factors = load_csv_to_dicts('macro_factors.csv')
    raw_company_drivers_links = load_csv_to_dicts('company_drivers_link.csv')
    raw_industry_drivers_links = load_csv_to_dicts('industry_drivers_link.csv')

    # Create Industry Nodes
    for row in raw_industries:
        props = {key: val for key, val in row.items() if key != 'id'}
        kg.add_node(node_id=row['id'], label='Industry', properties=props)

    # Create Company Nodes and BELONGS_TO_INDUSTRY Edges
    for row in raw_companies:
        company_id = row['id']
        industry_id = row['industryId']
        # Separate properties for the node from relationship info or complex fields
        props = {
            'name': row['name'],
            'ownershipStructure': row.get('ownershipStructure'),
            'corporateStructure': row.get('corporateStructure'),
            'financials': {
                'revenue_mn_usd': float(row['revenue_mn_usd']) if row.get('revenue_mn_usd') else None,
                'pe_ratio': float(row['pe_ratio']) if row.get('pe_ratio') else None,
                'debt_to_equity_ratio': float(row['debt_to_equity_ratio']) if row.get('debt_to_equity_ratio') else None,
            },
            'tradingLevels': {
                'price': float(row['current_price']) if row.get('current_price') else None,
                'volume': int(row['current_volume']) if row.get('current_volume') else None,
                'volatility': float(row['current_volatility_30d']) if row.get('current_volatility_30d') else None,
            }
        }
        kg.add_node(node_id=company_id, label='Company', properties=props)
        if kg.get_node(industry_id): # Ensure industry node exists
            kg.add_edge(source_id=company_id, target_id=industry_id, relationship_type='BELONGS_TO_INDUSTRY')
        else:
            print(f"Warning: Industry {industry_id} not found for company {company_id}.")


    # Create Driver Nodes
    for row in raw_drivers:
        driver_id = row['id']
        metrics = {}
        if row.get('metric_name_1') and row.get('metric_value_1'):
            metrics[row['metric_name_1']] = row['metric_value_1']
        if row.get('metric_name_2') and row.get('metric_value_2'):
            metrics[row['metric_name_2']] = row['metric_value_2']

        related_macros = []
        if row.get('relatedMacroFactorId_1'):
            related_macros.append(row['relatedMacroFactorId_1'])

        props = {
            'name': row['name'],
            'description': row.get('description',''),
            'type': get_enum_member(DriverType, row['type'], DriverType.UNKNOWN).value, # Store enum value
            'impactPotential': get_enum_member(ImpactPotential, row.get('impactPotential',''), ImpactPotential.UNKNOWN).value if row.get('impactPotential') else None,
            'timeHorizon': get_enum_member(TimeHorizon, row.get('timeHorizon',''), TimeHorizon.UNKNOWN).value if row.get('timeHorizon') else None,
            'metrics': metrics,
            'relatedMacroFactorIds': related_macros # Will link these later if needed as edges
        }
        kg.add_node(node_id=driver_id, label='Driver', properties=props)

    # Create MacroFactor Nodes
    for row in raw_macro_factors:
        macro_id = row['id']
        props = {
            'name': row['name'],
            'currentValue': row.get('currentValue'),
            'trend': get_enum_member(Trend, row.get('trend',''), Trend.UNKNOWN).value if row.get('trend') else None,
            'impactNarrative': row.get('impactNarrative'),
            'source': row.get('source')
        }
        kg.add_node(node_id=macro_id, label='MacroFactor', properties=props)

        # Link Drivers to MacroFactors (if specified in driver data)
        # This demonstrates creating edges from properties that are IDs
        for driver_node in kg.find_nodes_by_label('Driver'):
            if macro_id in driver_node.properties.get('relatedMacroFactorIds', []):
                kg.add_edge(source_id=driver_node.id, target_id=macro_id, relationship_type='RELATED_TO_MACROFACTOR')


    # Link Companies to Drivers (AFFECTED_BY_DRIVER)
    for link in raw_company_drivers_links:
        company_id = link['companyId']
        driver_id = link['driverId']
        if kg.get_node(company_id) and kg.get_node(driver_id):
            kg.add_edge(source_id=company_id, target_id=driver_id, relationship_type='AFFECTED_BY_DRIVER')
        else:
            print(f"Warning: Company {company_id} or Driver {driver_id} not found for linking.")

    # Link Industries to Drivers (AFFECTED_BY_DRIVER)
    for link in raw_industry_drivers_links:
        industry_id = link['industryId']
        driver_id = link['driverId']
        if kg.get_node(industry_id) and kg.get_node(driver_id):
            kg.add_edge(source_id=industry_id, target_id=driver_id, relationship_type='AFFECTED_BY_DRIVER')
        else:
            print(f"Warning: Industry {industry_id} or Driver {driver_id} not found for linking.")

    print(f"Knowledge graph built: {len(kg.nodes)} nodes, {len(kg.edges)} edges.")


# Global KG instance for simplicity in this example
# In a real app, this might be managed by the FastAPI app lifecycle or a dedicated service
KG_INSTANCE: Optional[KnowledgeGraph] = None

def get_kg_instance() -> KnowledgeGraph:
    global KG_INSTANCE
    if KG_INSTANCE is None:
        print("Building new KG instance...")
        KG_INSTANCE = KnowledgeGraph()
        build_graph_from_csvs(KG_INSTANCE)
    return KG_INSTANCE

if __name__ == '__main__':
    # Example of building and inspecting the graph
    kg = get_kg_instance()
    print(f"\nLoaded Knowledge Graph: {kg}")

    print("\nSample Company (AAPL):")
    aapl_node = kg.get_node("AAPL")
    if aapl_node:
        print(aapl_node)
        print("AAPL Drivers:")
        for driver_edge in kg.adj.get("AAPL", []):
            if driver_edge.relationship_type == 'AFFECTED_BY_DRIVER':
                print(f"  - {kg.get_node(driver_edge.target_id)}")

        print("AAPL Industry:")
        for industry_edge in kg.adj.get("AAPL", []):
            if industry_edge.relationship_type == 'BELONGS_TO_INDUSTRY':
                industry_node = kg.get_node(industry_edge.target_id)
                print(f"  - {industry_node}")
                if industry_node:
                    print(f"  Industry Drivers for {industry_node.id}:")
                    for ind_driver_edge in kg.adj.get(industry_node.id,[]):
                        if ind_driver_edge.relationship_type == 'AFFECTED_BY_DRIVER':
                            print(f"    - {kg.get_node(ind_driver_edge.target_id)}")
    else:
        print("AAPL node not found.")

    print("\nSample Driver (DRV001):")
    drv001_node = kg.get_node("DRV001")
    if drv001_node:
        print(drv001_node)
        print("DRV001 related Macro Factors:")
        for mf_edge in kg.adj.get("DRV001", []):
            if mf_edge.relationship_type == 'RELATED_TO_MACROFACTOR':
                 print(f"  - {kg.get_node(mf_edge.target_id)}")
    else:
        print("DRV001 node not found.")
