import re
from typing import Dict, Any, List

class AuditAgent:
    """
    Automated 'Audit Agent' that verifies compliance with Architectural Decision Records (ADRs).
    """

    @staticmethod
    def audit_vpc_isolation(network_config: Dict[str, Any]) -> str:
        """
        Verifies that private subnets do not have direct internet access.
        """
        for subnet in network_config.get('subnets', []):
            if subnet.get('type') == 'private' and subnet.get('has_internet_gateway') is True:
                return "FAIL: Private subnet has IGW attached."
            if subnet.get('tags', {}).get('component') == 'inference_engine' and subnet.get('outbound_rule') != 'nat_gateway_only':
                 return "FAIL: Inference Engine not strictly isolated."
        return "PASS: Network isolation verified."

    @staticmethod
    def audit_prompt_reproducibility(audit_log: Dict[str, Any], prompt_registry: Any) -> str:
        """
        Verifies that the prompt version used in the audit log exists in the registry.
        """
        prompt_sha = audit_log.get('prompt_version_id') or audit_log.get('model_state', {}).get('prompt_version_id')
        if not prompt_sha:
             return "FAIL: No prompt version ID found in audit log."

        # Simplified check for demo purposes
        # In a real system, we'd check against git commit SHAs
        if not prompt_registry.get(prompt_sha):
            # For demo, we might return PASS if it matches a known pattern or mock it
            # But let's assume the registry has a 'commit_exists' or similar
             return "FAIL: Prompt version SHA not found in Registry."
        return "PASS: Prompt version is immutable and retrievable."

    @staticmethod
    def audit_citation_density(generated_text: str) -> str:
        """
        Verifies that the generated text has a sufficient density of citations.
        """
        # Count sentences (simple split by period)
        sentences = [s for s in generated_text.split('.') if len(s.strip()) > 5]
        claims = len(sentences)
        if claims == 0:
            return "WARNING: No claims found."

        # Count citations [doc_id:chunk_id]
        citations = len(re.findall(r"\[doc_.*?:.*?\]", generated_text))

        density = citations / claims if claims > 0 else 0
        if density < 0.8: # Threshold: 80% of sentences must be cited
            return f"WARNING: Citation density {density:.2f} below threshold (0.8)."
        return "PASS: Grounding sufficient."

    @staticmethod
    def audit_balance_sheet_integrity(spread_json: Dict[str, Any]) -> str:
        """
        Verifies that Assets = Liabilities + Equity.
        """
        assets = spread_json.get('total_assets', 0.0)
        liabilities = spread_json.get('total_liabilities', 0.0)
        equity = spread_json.get('total_equity', 0.0)

        if abs(assets - (liabilities + equity)) > 1.0: # Allow $1.00 rounding error
            return f"FAIL: Balance Sheet mismatch. Delta: {assets - (liabilities + equity)}"
        return "PASS: Financial integrity verified."
