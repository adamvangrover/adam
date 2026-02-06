from typing import List, Dict, Any
import rdflib
from rdflib import Graph, Literal, BNode, Namespace, RDF, RDFS, URIRef

class SymbolicVerifier:
    """
    The 'Symbolic Verification' layer.
    Uses an RDF Graph (FIBO stub) to check for semantic contradictions in the agent's output.
    """

    def __init__(self):
        self.g = Graph()
        self.ADAM = Namespace("http://adam.system/ontology/")

        # Initialize with Stub Knowledge
        self._load_stub_knowledge()

    def _load_stub_knowledge(self):
        """
        Loads a hardcoded slice of the knowledge graph.
        In production, this would query a Triple Store.
        """
        # Entity: Term Loan B
        term_loan = self.ADAM.TermLoanB
        self.g.add((term_loan, RDF.type, self.ADAM.DebtInstrument))
        self.g.add((term_loan, self.ADAM.hasSeniority, self.ADAM.Subordinated))
        self.g.add((term_loan, RDFS.label, Literal("Term Loan B")))

        # Entity: Parent Corp
        parent = self.ADAM.ParentCorp
        self.g.add((parent, RDF.type, self.ADAM.HoldingCompany))
        self.g.add((parent, RDFS.label, Literal("Parent Corporation")))

    def verify(self, text: str) -> List[str]:
        """
        Scans the text for entities and verifies their attributes against the graph.
        Returns a list of 'Flags' (error messages).
        """
        flags = []
        if not text:
            return flags

        text_lower = text.lower()

        # --- Check 1: Seniority Mismatch ---
        # If the agent discusses "Term Loan B", check if it misidentifies its seniority.
        if "term loan b" in text_lower:
            # Query Graph
            # Find the seniority of Term Loan B
            q = """
            SELECT ?seniority WHERE {
                ?loan <http://www.w3.org/2000/01/rdf-schema#label> "Term Loan B" .
                ?loan <http://adam.system/ontology/hasSeniority> ?seniority .
            }
            """

            for row in self.g.query(q):
                seniority_uri = str(row.seniority)
                # If graph says "Subordinated" but text says "Senior"
                if "subordinated" in seniority_uri.lower():
                    if "senior" in text_lower and "secured" in text_lower:
                         flags.append(
                             "SYMBOLIC FAIL: Agent identified 'Term Loan B' as 'Senior Secured'. "
                             "Knowledge Graph asserts it is 'Subordinated'."
                         )

        # --- Check 2: Corporate Structure ---
        # If agent implies Parent is an "Operating Company"
        if "parent" in text_lower and "operating" in text_lower:
             # Check Graph for Parent Type
             if (self.ADAM.ParentCorp, RDF.type, self.ADAM.HoldingCompany) in self.g:
                 flags.append(
                     "SYMBOLIC FAIL: Agent refers to Parent as 'Operating'. "
                     "Knowledge Graph classifies it as a 'HoldingCompany'."
                 )

        return flags
