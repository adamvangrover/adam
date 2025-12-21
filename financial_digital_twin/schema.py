"""
This module defines the core schema for the Financial Knowledge Graph.

It uses Python classes to represent the different types of nodes (entities)
and edges (relationships) in the graph, based on the specifications in
Phase 1 and Section 3 of the strategic roadmap.

The schema is intended to be used as a code-based reference for data ingestion,
query generation, and validation.
"""

from dataclasses import dataclass
from typing import Optional

# --- Core Nodes (Entities) ---

@dataclass
class Company:
    """Represents a company, which can be a borrower, guarantor, investor, etc."""
    legal_name: str
    tax_id: str
    industry_code: str
    address: str
    risk_rating: Optional[float] = None
    time_series_id: Optional[str] = None # For linking to TSDB

@dataclass
class Loan:
    """Represents a specific credit facility, like a term loan or revolver."""
    loan_id: str
    loan_type: str # e.g., 'Term Loan', 'Revolver'
    amount: float
    maturity_date: str # Using str for simplicity, can be changed to date object
    status: str # e.g., 'Active', 'Defaulted', 'Paid Off'

@dataclass
class Security:
    """Represents a tradable asset, like a bond or syndicated loan share."""
    cusip_isin: str
    issue_date: str
    face_value: float
    security_type: str # e.g., 'Bond', 'Syndicated Loan'
    time_series_id: Optional[str] = None # For linking to TSDB

@dataclass
class Collateral:
    """Represents an asset securing a loan."""
    asset_type: str
    appraised_value: float
    location: str

@dataclass
class Individual:
    """Represents a key individual, like an executive or board member."""
    name: str
    title: str

@dataclass
class Covenant:
    """Represents a financial or operational performance requirement."""
    covenant_id: str
    description: str
    covenant_type: str # e.g., 'Financial', 'Operational'

@dataclass
class Financials:
    """Represents a specific financial statement (e.g., 10-K, 10-Q)."""
    filing_id: str
    filing_type: str # '10-K', '10-Q'
    filing_date: str
    company_id: str # Foreign key to the company it belongs to

# --- Core Edges (Relationships) ---
# These classes define the connections between nodes and their properties.

@dataclass
class IsBorrowerOf:
    """Edge: (Company)-[:IS_BORROWER_OF]->(Loan)"""
    pass # No properties on this edge

@dataclass
class SecuredBy:
    """Edge: (Loan)-[:SECURED_BY]->(Collateral)"""
    lien_position: int
    appraisal_value: float

@dataclass
class Issued:
    """Edge: (Company)-[:ISSUED]->(Security)"""
    pass

@dataclass
class HoldsPositionIn:
    """Edge: (Investor:Company)-[:HOLDS_POSITION_IN]->(Security)"""
    shares: int
    purchase_date: str

@dataclass
class HasParent:
    """Edge: (Company)-[:HAS_PARENT]->(Company)"""
    pass

@dataclass
class WorksFor:
    """Edge: (Individual)-[:WORKS_FOR]->(Company)"""
    start_date: str
    end_date: Optional[str] = None

@dataclass
class SubjectTo:
    """Edge: (Loan)-[:SUBJECT_TO]->(Covenant)"""
    pass
