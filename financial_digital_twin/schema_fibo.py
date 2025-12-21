"""
This module defines the core schema for the Financial Knowledge Graph,
aligned with the Financial Industry Business Ontology (FIBO).

This schema co-exists with the legacy `schema.py` and represents the
strategic, enterprise-grade data model for the Digital Twin. It is based
on the mappings defined in `ontology.md`.

The classes are designed to be used as code-based references for data ingestion,
query generation, and validation against the FIBO standard.
"""

from dataclasses import dataclass
from typing import Optional

# --- Core Nodes (Entities) - FIBO Aligned ---
# The class names and attributes are chosen to reflect their FIBO counterparts.

@dataclass
class LegalEntity:
    """
    Represents a legal entity, such as a company or organization.
    Corresponds to fibo-be-le-lei:LegalEntity
    """
    legal_name: str  # Maps to fibo-fnd-utl-av:hasName
    lei_code: str    # Maps to fibo-be-le-lei:hasLegalEntityIdentifier
    headquarters_address: str # Maps to fibo-fnd-org-fm:hasHeadquartersAddress
    risk_rating: Optional[float] = None # Internal property, not directly in FIBO class
    time_series_id: Optional[str] = None # For linking to TSDB

@dataclass
class Loan:
    """
    Represents a loan as a financial instrument.
    Corresponds to fibo-fbc-fi-fi:Loan
    """
    loan_id: str # Internal identifier
    currency: str # Maps to fibo-fnd-acc-cur:hasCurrency
    principal_amount: float # Maps to fibo-loan-ln-ln:hasPrincipalAmount
    maturity_date: str # Maps to fibo-fnd-agr-ctr:hasMaturityDate
    status: str # Internal property

@dataclass
class Security:
    """
    Represents a tradable security.
    Corresponds to fibo-sec-sec-bsic:Security
    """
    cusip: str # Maps to fibo-sec-sec-id:hasCUSIP
    isin: Optional[str] = None # Maps to fibo-sec-sec-id:hasISIN
    issue_date: str # Maps to fibo-fnd-agr-ctr:hasIssueDate
    face_value: float # Not a direct FIBO property, but essential for modeling
    time_series_id: Optional[str] = None # For linking to TSDB

@dataclass
class NaturalPerson:
    """
    Represents a key individual.
    Corresponds to fibo-be-oac-opty:NaturalPerson
    """
    full_name: str # Maps to fibo-fnd-utl-av:hasName
    role: str # Describes the person's role, e.g., 'CEO', 'Director'

@dataclass
class Covenant:
    """
    Represents a covenant associated with a loan.
    Corresponds to fibo-loan-ln-covenant:Covenant
    """
    covenant_id: str # Internal identifier
    description: str # Maps to fibo-fnd-utl-av:hasDescription
    is_legally_binding: bool = True # Maps to fibo-fnd-agr-ctr:isLegallyBinding

@dataclass
class Collateral:
    """
    Represents collateral securing a loan.
    Corresponds to fibo-loan-ln-ln:Collateral
    """
    description: str # Maps to fibo-fnd-utl-av:hasDescription
    appraised_value: float # Relates to fibo-fnd-acc-cur:hasMonetaryAmount

@dataclass
class FinancialReport:
    """
    Represents a financial report filed by a company.
    Corresponds to fibo-fbc-fct-fse:FinancialReport
    """
    report_id: str # Internal identifier
    filing_date: str # Maps to fibo-fnd-rel-rel:isFiledOn
    company_lei: str # Foreign key to the LegalEntity it belongs to

# --- Relationships ---
# In a graph, relationships are typically represented as edges.
# In a code-based schema, these can be represented by linking properties
# or dedicated relationship classes if properties are needed on the edge.
# For simplicity, we will imply relationships via linking properties in the nodes
# (e.g., `FinancialReport.company_lei`).
