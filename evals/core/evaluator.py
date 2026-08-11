"""
Core Evaluator logic for the ADAM Gold Standard Harness.
"""

from typing import Dict, Any, List

class GoldStandardEvaluator:
    def __init__(self, system_version: str):
        self.system_version = system_version
        self.results = []
        self.critical_failures = 0

    def run_test(self, test_id: str, domain: str, test_func, **kwargs) -> Dict[str, Any]:
        """
        Executes a single test and records the result.
        """
        try:
            result = test_func(**kwargs)
            self.results.append({
                "test_id": test_id,
                "domain": domain,
                "status": result.get("status", "INCONCLUSIVE"),
                "severity": result.get("severity", "LOW")
            })
            if result.get("status") == "FAIL" and result.get("severity") == "CRITICAL":
                self.critical_failures += 1
            return result
        except Exception as e:
            self.results.append({
                "test_id": test_id,
                "domain": domain,
                "status": "FAIL",
                "severity": "CRITICAL",
                "error": str(e)
            })
            self.critical_failures += 1
            return {"status": "FAIL", "error": str(e)}

    def finalize(self) -> Dict[str, Any]:
        """
        Returns the final evaluation status.
        """
        return {
            "total_tests": len(self.results),
            "critical_failures": self.critical_failures,
            "certification": "FAIL" if self.critical_failures > 0 else "PASS"
        }
