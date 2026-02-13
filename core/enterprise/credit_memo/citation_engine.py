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
                doc_id="AAPL_10K_2025.pdf",
                page_number=4,
                text="Net sales for the iPhone category decreased 2% year-over-year due to supply chain constraints.",
                bbox=[0.1, 0.15, 0.9, 0.25],
                confidence=0.99
            ))
            chunks.append(EvidenceChunk(
                doc_id="AAPL_10K_2025.pdf",
                page_number=18,
                text="The Company faces significant legal risks regarding App Store policies in the EU under the Digital Markets Act.",
                bbox=[0.1, 0.5, 0.9, 0.6],
                confidence=0.96
            ))
        elif "tesla" in name_lower:
             chunks.append(EvidenceChunk(
                doc_id="TSLA_10K_2025.pdf",
                page_number=7,
                text="Automotive gross margin excluding regulatory credits fell to 16.5% due to price cuts.",
                bbox=[0.2, 0.3, 0.8, 0.4],
                confidence=0.97
            ))
             chunks.append(EvidenceChunk(
                doc_id="TSLA_10K_2025.pdf",
                page_number=12,
                text="Cybercab production ramp remains slower than anticipated due to 4680 cell yield issues.",
                bbox=[0.1, 0.6, 0.9, 0.7],
                confidence=0.94
            ))
        elif "jpmorgan" in name_lower or "chase" in name_lower:
             chunks.append(EvidenceChunk(
                doc_id="JPM_10K_2025.pdf",
                page_number=5,
                text="Net Interest Income (NII) rose 12% to record levels, driven by higher rates and loan growth.",
                bbox=[0.15, 0.2, 0.85, 0.3],
                confidence=0.98
            ))
             chunks.append(EvidenceChunk(
                doc_id="JPM_Risk_Report_2025.pdf",
                page_number=22,
                text="Commercial Real Estate (CRE) exposure remains a key monitoring area, with office vacancy rates impacting valuation.",
                bbox=[0.1, 0.4, 0.9, 0.5],
                confidence=0.95
            ))
        elif "techcorp" in name_lower or "tech" in query.lower():
            # Mock Doc 1: 10-K
            chunks.append(EvidenceChunk(
                doc_id="TechCorp_10K_2025.pdf",
                page_number=3,
                text="TechCorp reported a 15% decline in EBITDA due to rising semiconductor costs.",
                bbox=[0.1, 0.2, 0.9, 0.3], # Normalized x0, y0, x1, y1
                confidence=0.98
            ))
            chunks.append(EvidenceChunk(
                doc_id="TechCorp_10K_2025.pdf",
                page_number=12,
                text="The company faces significant litigation risk from a patent dispute with chip suppliers.",
                bbox=[0.1, 0.5, 0.9, 0.6],
                confidence=0.95
            ))
            # Mock Doc 2: Market Report
            chunks.append(EvidenceChunk(
                doc_id="Gartner_Market_Outlook.pdf",
                page_number=5,
                text="Global semiconductor demand is projected to soften in Q3 2026.",
                bbox=[0.2, 0.2, 0.8, 0.4],
                confidence=0.92
            ))
        else:
            # Generic
            chunks.append(EvidenceChunk(
                doc_id="Generic_Borrower_Profile.pdf",
                page_number=1,
                text=f"The borrower {borrower_name} operates in a stable industry.",
                bbox=[0.1, 0.1, 0.9, 0.2],
                confidence=0.80
            ))

        return chunks

# Global Instance
citation_engine = CitationEngine()
