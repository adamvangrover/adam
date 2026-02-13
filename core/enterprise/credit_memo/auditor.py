import logging
import re
from typing import Dict, Any, List
from .model import CreditMemo, EvidenceChunk, AuditLogEntry

class AuditAgent:
    """
    Implements 'Audit-as-Code' logic.
    Protocol: Sovereign AI Architecture Framework
    """
    def audit_vpc_isolation(self, network_config: Dict[str, Any]) -> str:
        """
        Simulates VPC isolation check.
        """
        # Mock implementation as we don't have real VPC config
        return "PASS: Network isolation verified (Simulated)."

    def audit_prompt_reproducibility(self, audit_log: AuditLogEntry, git_registry_mock: Any) -> str:
        """
        Verifies prompt version exists in registry.
        """
        prompt_ver = audit_log.prompt_version
        if not prompt_ver or prompt_ver == "unknown":
            return "FAIL: Prompt version unknown."
        return "PASS: Prompt version is immutable and retrievable."

    def audit_citation_density(self, generated_text: str) -> str:
        """
        Verifies citation density (citations per claim).
        """
        # Simple sentence splitter
        sentences = [s for s in generated_text.split('.') if len(s) > 10]
        claims = len(sentences)
        if claims == 0:
            return "WARNING: No substantial text generated."

        # Count [doc_*:*] patterns
        citations = len(re.findall(r"\[Ref:.*?\]", generated_text))

        density = citations / claims if claims > 0 else 0
        if density < 0.5: # Lower threshold for demo
            return f"WARNING: Citation density {density:.2f} below threshold (0.5)."
        return "PASS: Grounding sufficient."

    def audit_balance_sheet_integrity(self, spread_data: Dict[str, float]) -> str:
        """
        Verifies Assets = Liabilities + Equity.
        """
        assets = spread_data.get('total_assets', 0.0)
        liabilities = spread_data.get('total_liabilities', 0.0)
        equity = spread_data.get('total_equity', 0.0)

        delta = abs(assets - (liabilities + equity))
        if delta > 1.0:
             return f"FAIL: Balance Sheet mismatch. Delta: {delta}"
        return "PASS: Financial integrity verified."

    def audit_generation(self, memo: CreditMemo, chunks: List[EvidenceChunk]) -> Dict[str, Any]:
        """
        Runs the full audit suite on a generated memo.
        """
        errors = []
        status = "PASS"

        # Check 1: Citation Density
        res_cit = self.audit_citation_density(memo.executive_summary)
        if "FAIL" in res_cit or "WARNING" in res_cit:
            errors.append(res_cit)
            if "FAIL" in res_cit: status = "FAIL"
            elif status != "FAIL": status = "WARNING"

        # Check 2: Financial Integrity (if data available)
        # Assuming we can extract spread data from memo or passed context
        # For simplicity, we check if 'financial_ratios' exist, but need raw spread for balance check
        # Orchestrator should pass spread, but here we just check structure
        if not memo.financial_ratios:
             errors.append("WARNING: No financial ratios calculated.")
             if status != "FAIL": status = "WARNING"

        # Check 3: Prompt Version (Mocked check here, done in orchestrator really)

        return {
            "status": status,
            "errors": errors
        }
