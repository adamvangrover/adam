from typing import Dict, Any, List

class QuantumScanner:
    """
    Simulates a vulnerability scanner that checks for post-quantum cryptographic
    vulnerabilities using industry standards. It evaluates if current encryption
    algorithms are susceptible to quantum attacks, specifically Shor's algorithm.
    """

    # Algorithms vulnerable to Shor's algorithm (integer factorization & discrete logs)
    VULNERABLE_ALGORITHMS = {"RSA", "ECC", "ECDSA", "ECDH", "DSA"}

    # Algorithms considered quantum-resistant (symmetric with large keys, lattice-based, etc.)
    SAFE_ALGORITHMS = {"AES-256", "ChaCha20", "Kyber", "Dilithium", "SPHINCS+", "Falcon"}

    def scan_encryption_algorithm(self, algorithm: str) -> Dict[str, Any]:
        """
        Scans a specific encryption algorithm and returns a risk assessment.

        Args:
            algorithm (str): The name of the encryption algorithm (e.g., 'RSA', 'AES-256').

        Returns:
            Dict[str, Any]: Assessment containing vulnerability status, severity, and remediation advice.
        """
        algo_upper = algorithm.upper()

        # Check against known vulnerable list
        for vuln_algo in self.VULNERABLE_ALGORITHMS:
            if vuln_algo in algo_upper:
                return {
                    "algorithm": algorithm,
                    "vulnerable": True,
                    "severity": "CRITICAL",
                    "reason": f"{algorithm} is susceptible to Shor's algorithm on a sufficiently powerful quantum computer.",
                    "remediation": "Migrate to quantum-resistant algorithms such as NIST-approved Lattice-based cryptography (e.g., Kyber, Dilithium) or use AES-256 for symmetric encryption."
                }

        # Check against known safe list
        for safe_algo in self.SAFE_ALGORITHMS:
            if safe_algo.upper() in algo_upper:
                return {
                    "algorithm": algorithm,
                    "vulnerable": False,
                    "severity": "INFO",
                    "reason": f"{algorithm} is currently considered resistant to known quantum attacks.",
                    "remediation": "Continue monitoring NIST PQC standardization for any updates."
                }

        # Unknown or unspecified algorithms
        return {
            "algorithm": algorithm,
            "vulnerable": True, # Fail closed for unknown algorithms in a critical environment
            "severity": "HIGH",
            "reason": f"Algorithm '{algorithm}' is unrecognized or not explicitly whitelisted as quantum-safe.",
            "remediation": "Verify algorithm against NIST Post-Quantum Cryptography standards."
        }

    def scan_system(self, algorithms_in_use: List[str]) -> Dict[str, Any]:
        """
        Scans a list of algorithms used in a system and aggregates the risk.
        """
        assessments = [self.scan_encryption_algorithm(algo) for algo in algorithms_in_use]

        critical_vulnerabilities = sum(1 for a in assessments if a["severity"] == "CRITICAL")
        high_vulnerabilities = sum(1 for a in assessments if a["severity"] == "HIGH")

        system_vulnerable = critical_vulnerabilities > 0 or high_vulnerabilities > 0

        overall_severity = "LOW"
        if critical_vulnerabilities > 0:
            overall_severity = "CRITICAL"
        elif high_vulnerabilities > 0:
            overall_severity = "HIGH"

        return {
            "system_vulnerable": system_vulnerable,
            "overall_severity": overall_severity,
            "findings": assessments
        }
