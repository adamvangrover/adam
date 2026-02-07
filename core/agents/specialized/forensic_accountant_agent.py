from typing import Dict, Any, List
import logging
import math
from collections import Counter
from core.agents.agent_base import AgentBase
from core.agents.mixins.audit_mixin import AuditMixin

class ForensicAccountantAgent(AgentBase, AuditMixin):
    """
    Specialized agent for detecting financial fraud and anomalies in ledger data.
    Uses statistical methods (Benford's Law) and heuristic rules.
    """

    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        AuditMixin.__init__(self)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Analyzes a ledger or transaction list for anomalies.
        Expected kwargs:
            transactions: List[Dict] with 'amount' field.
        """
        transactions = kwargs.get("transactions", [])
        if not transactions:
            return {"status": "SKIPPED", "reason": "No transactions provided"}

        logging.info(f"ForensicAccountant scanning {len(transactions)} transactions.")

        # 1. Benford's Law Analysis
        benford_score, benford_dist = self._check_benford(transactions)

        # 2. Round Number Analysis
        round_number_pct = self._check_round_numbers(transactions)

        # 3. Verdict
        anomalies = []
        risk_level = "LOW"

        if benford_score > 0.15: # Threshold for divergence
            anomalies.append("Significant deviation from Benford's Law.")
            risk_level = "HIGH"

        if round_number_pct > 0.10: # More than 10% round numbers is suspicious in some contexts
            anomalies.append(f"High frequency of round numbers ({round_number_pct:.1%}).")
            if risk_level == "LOW": risk_level = "MEDIUM"

        result = {
            "risk_level": risk_level,
            "anomalies_detected": anomalies,
            "metrics": {
                "benford_divergence": benford_score,
                "round_number_ratio": round_number_pct
            },
            "benford_distribution": benford_dist
        }

        self.log_decision(
            activity_type="ForensicAudit",
            details={"transaction_count": len(transactions)},
            outcome=result
        )

        return result

    def _check_benford(self, transactions: List[Dict]) -> tuple[float, Dict[str, float]]:
        """
        Calculates the divergence from Benford's Law for leading digits 1-9.
        Returns a divergence score (0.0 = perfect fit) and the actual distribution.
        """
        leading_digits = []
        for t in transactions:
            amount = abs(t.get('amount', 0))
            if amount > 0:
                digit = str(amount).lstrip('0 .')[0]
                if digit.isdigit() and digit != '0':
                    leading_digits.append(digit)

        total = len(leading_digits)
        if total == 0:
            return 0.0, {}

        counts = Counter(leading_digits)
        actual_dist = {d: c/total for d, c in counts.items()}

        # Benford's Expected Distribution
        # P(d) = log10(1 + 1/d)
        score = 0.0
        dist_export = {}

        for d in range(1, 10):
            d_str = str(d)
            expected = math.log10(1 + 1/d)
            actual = actual_dist.get(d_str, 0.0)
            score += abs(expected - actual)
            dist_export[d_str] = {"actual": round(actual, 3), "expected": round(expected, 3)}

        return score, dist_export

    def _check_round_numbers(self, transactions: List[Dict]) -> float:
        """
        Checks the percentage of transactions that are 'round' numbers (multiple of 100).
        """
        round_count = 0
        total = 0

        for t in transactions:
            amount = t.get('amount', 0)
            if amount > 0:
                total += 1
                if amount % 100 == 0:
                    round_count += 1

        return round_count / total if total > 0 else 0.0

    def get_skill_schema(self) -> Dict[str, Any]:
        return {
            "name": "ForensicAccountantAgent",
            "description": "Detects financial anomalies using statistical analysis.",
            "skills": [
                {
                    "name": "audit_ledger",
                    "description": "Scans transactions for fraud patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "transactions": {"type": "array", "items": {"type": "object"}}
                        },
                        "required": ["transactions"]
                    }
                }
            ]
        }
