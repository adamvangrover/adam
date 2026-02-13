from typing import List, Dict, Any
from .model import EvidenceChunk
import uuid

class CitationEngine:
    """
    Simulates the 'Visual' Ingestion Pipeline with Spatial RAG.
    Protocol: Spatial Awareness
    """
    def retrieve_context(self, borrower_name: str, query: str) -> List[EvidenceChunk]:
        """
        Retrieves relevant document chunks with bounding boxes.
        Mock implementation: Returns mock chunks for AAPL, TSLA, JPM, and TechCorp.
        """
        chunks = []
        name_lower = borrower_name.lower()

        if "apple" in name_lower:
            chunks.append(EvidenceChunk(
                doc_id="AAPL_10Q_FY25_Q1.pdf",
                page_number=6,
                text="Services net sales increased 11% year-over-year to an all-time quarterly record, driven by growth in paid subscriptions.",
                bbox=[0.12, 0.22, 0.88, 0.35],
                confidence=0.99,
                chunk_type="10-K/Q"
            ))
            chunks.append(EvidenceChunk(
                doc_id="AAPL_Credit_Agreement_2023.pdf",
                page_number=42,
                text="Section 6.01. Financial Covenants. The Borrower shall not permit the Consolidated Leverage Ratio as of the last day of any fiscal quarter to exceed 3.50 to 1.00.",
                bbox=[0.1, 0.1, 0.9, 0.2],
                confidence=1.0,
                chunk_type="Credit Agreement"
            ))
            chunks.append(EvidenceChunk(
                doc_id="AAPL_Risk_Assessment_2025.pdf",
                page_number=3,
                text="Geopolitical tensions in the Greater China region could adversely affect supply chain continuity and consumer demand.",
                bbox=[0.2, 0.4, 0.8, 0.5],
                confidence=0.95,
                chunk_type="Research"
            ))

        elif "tesla" in name_lower:
             chunks.append(EvidenceChunk(
                doc_id="TSLA_10Q_FY24_Q3.pdf",
                page_number=8,
                text="Automotive gross margin excluding regulatory credits stabilized at 17.1%, despite continued pricing pressure in key markets.",
                bbox=[0.15, 0.25, 0.85, 0.38],
                confidence=0.98,
                chunk_type="10-K/Q"
            ))
             chunks.append(EvidenceChunk(
                doc_id="TSLA_Revolver_Agreement.pdf",
                page_number=15,
                text="Negative Pledge. The Borrower will not create, incur, assume or permit to exist any Lien on any property or asset now owned or hereafter acquired by it...",
                bbox=[0.1, 0.6, 0.9, 0.75],
                confidence=0.99,
                chunk_type="Credit Agreement"
            ))
             chunks.append(EvidenceChunk(
                doc_id="TSLA_Energy_Report.pdf",
                page_number=2,
                text="Energy Generation and Storage revenue increased 52% YoY, becoming a significant contributor to free cash flow.",
                bbox=[0.2, 0.1, 0.8, 0.2],
                confidence=0.94,
                chunk_type="Research"
            ))

        elif "jpmorgan" in name_lower or "chase" in name_lower:
             chunks.append(EvidenceChunk(
                doc_id="JPM_Earnings_Release_4Q24.pdf",
                page_number=1,
                text="Net Interest Income (NII) was $24.2 billion, up 3%, driven by higher rates and revolving balances in Card Services.",
                bbox=[0.1, 0.15, 0.9, 0.25],
                confidence=0.99,
                chunk_type="Earnings"
            ))
             chunks.append(EvidenceChunk(
                doc_id="JPM_10K_FY24.pdf",
                page_number=45,
                text="The Firm estimates that the proposed Basel III Endgame capital rules could increase risk-weighted assets by approximately 25%.",
                bbox=[0.15, 0.4, 0.85, 0.55],
                confidence=0.96,
                chunk_type="10-K/Q"
            ))
             chunks.append(EvidenceChunk(
                doc_id="JPM_Risk_Report_2024.pdf",
                page_number=12,
                text="Credit costs of $2.5 billion reflected net charge-offs of $2.2 billion, predominantly in Card Services and Single-Family Residential.",
                bbox=[0.1, 0.7, 0.9, 0.8],
                confidence=0.95,
                chunk_type="Risk Report"
            ))

        elif "techcorp" in name_lower or "tech" in query.lower():
            chunks.append(EvidenceChunk(
                doc_id="TechCorp_10K_2025.pdf",
                page_number=3,
                text="TechCorp reported a 15% decline in EBITDA due to rising semiconductor costs.",
                bbox=[0.1, 0.2, 0.9, 0.3],
                confidence=0.98,
                chunk_type="10-K/Q"
            ))
            chunks.append(EvidenceChunk(
                doc_id="TechCorp_Revolver.pdf",
                page_number=55,
                text="Section 7.02. Events of Default. If the Borrower fails to pay any principal of any Loan when due...",
                bbox=[0.1, 0.5, 0.9, 0.6],
                confidence=0.99,
                chunk_type="Credit Agreement"
            ))
            chunks.append(EvidenceChunk(
                doc_id="Gartner_Market_Outlook.pdf",
                page_number=5,
                text="Global semiconductor demand is projected to soften in Q3 2026.",
                bbox=[0.2, 0.2, 0.8, 0.4],
                confidence=0.92,
                chunk_type="Research"
            ))
        else:
            chunks.append(EvidenceChunk(
                doc_id="Generic_Borrower_Profile.pdf",
                page_number=1,
                text=f"The borrower {borrower_name} operates in a stable industry.",
                bbox=[0.1, 0.1, 0.9, 0.2],
                confidence=0.80,
                chunk_type="General"
            ))

        return chunks

# Global Instance
citation_engine = CitationEngine()
