from typing import List, Dict, Any, Union
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
        self.g.add((term_loan, self.ADAM.hasSeniority, self.ADAM.Subordinated)) # Contradiction trap
        self.g.add((term_loan, RDFS.label, Literal("Term Loan B")))

        # Entity: Revolving Credit Facility (RCF)
        rcf = self.ADAM.Revolver
        self.g.add((rcf, RDF.type, self.ADAM.DebtInstrument))
        self.g.add((rcf, self.ADAM.hasSeniority, self.ADAM.SeniorSecured))
        self.g.add((rcf, RDFS.label, Literal("Revolver")))

        # Entity: Senior Unsecured Notes
        sun = self.ADAM.SeniorUnsecuredNotes
        self.g.add((sun, RDF.type, self.ADAM.DebtInstrument))
        self.g.add((sun, self.ADAM.hasSeniority, self.ADAM.Unsecured))
        self.g.add((sun, RDFS.label, Literal("Senior Unsecured Notes")))

        # Entity: Parent Corp
        parent = self.ADAM.ParentCorp
        self.g.add((parent, RDF.type, self.ADAM.HoldingCompany))
        self.g.add((parent, RDFS.label, Literal("Parent Corporation")))

    def verify(self, text: str) -> List[Dict[str, str]]:
        """
        Scans the text for entities and verifies their attributes against the graph.
        Returns a list of structured 'Flags'.
        """
        flags = []
        if not text:
            return flags

        text_lower = text.lower()

        # --- Check 1: Seniority Mismatch (Term Loan B) ---
        if "term loan b" in text_lower:
            # Query Graph for Seniority
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
                         flags.append({
                             "type": "SYMBOLIC_FAIL",
                             "severity": "HIGH",
                             "entity": "Term Loan B",
                             "message": "Agent identified 'Term Loan B' as 'Senior Secured', but Knowledge Graph asserts it is 'Subordinated'."
                         })

        # --- Check 2: Seniority Mismatch (Senior Unsecured Notes) ---
        if "unsecured notes" in text_lower:
             # If agent claims they are Secured
             if "secured" in text_lower and "unsecured" not in text_lower: # Naive check
                 flags.append({
                     "type": "SYMBOLIC_FAIL",
                     "severity": "HIGH",
                     "entity": "Senior Unsecured Notes",
                     "message": "Agent identified Notes as Secured, but Knowledge Graph asserts they are Unsecured."
                 })

        # --- Check 3: Corporate Structure ---
        # If agent implies Parent is an "Operating Company"
        if "parent" in text_lower and "operating" in text_lower:
             # Check Graph for Parent Type
             if (self.ADAM.ParentCorp, RDF.type, self.ADAM.HoldingCompany) in self.g:
                 flags.append({
                     "type": "SYMBOLIC_FAIL",
                     "severity": "MEDIUM",
                     "entity": "Parent Corporation",
                     "message": "Agent refers to Parent as 'Operating', but Knowledge Graph classifies it as a 'HoldingCompany'."
                 })

        return flags
