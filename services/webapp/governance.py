import yaml
import os
import re
import logging
import hmac
import hashlib
import time
from flask import request, jsonify, abort

class GovernanceMiddleware:
    """
    Middleware to intercept and validate requests against the governance policy.
    Ensures 'High Risk' operations are checked before execution.
    """

    def __init__(self, app=None, policy_path='config/governance_policy.yaml'):
        self.policy = {}
        self.policy_path = policy_path
        self._load_policy()

        if app:
            self.init_app(app)

    def _load_policy(self):
        if os.path.exists(self.policy_path):
            with open(self.policy_path, 'r') as f:
                self.policy = yaml.safe_load(f)
        else:
            logging.warning(f"Governance policy not found at {self.policy_path}. Defaulting to ALLOW all.")
            self.policy = {"global_policy": {"default_action": "ALLOW"}}

    def init_app(self, app):
        """Register the before_request hook."""
        app.before_request(self.check_governance)

    def check_governance(self):
        """
        The core logic executed before every request.
        """
        # Skip static files or OPTIONS
        if request.method == 'OPTIONS' or request.path.startswith('/static'):
            return

        # 1. IP Blacklist Check
        client_ip = request.remote_addr
        if client_ip in self.policy.get('blacklisted_ips', []):
            logging.warning(f"Governance Block: IP {client_ip} is blacklisted.")
            abort(403, description="Access denied by governance policy.")

        # 2. Endpoint/Method Check
        for rule in self.policy.get('restricted_endpoints', []):
            # Simple path matching (startswith)
            if request.path.startswith(rule['path']):
                allowed_methods = rule.get('methods', [])
                if '*' in allowed_methods or request.method in allowed_methods:
                    self._enforce_rule(rule)

        # 3. Payload Keyword Scanning (for POST/PUT)
        if request.method in ['POST', 'PUT']:
            data = request.get_data(as_text=True)
            for keyword in self.policy.get('keywords_flagged', []):
                if keyword in data:
                    logging.warning(f"Governance Block: Request contains flagged keyword '{keyword}'")
                    abort(400, description="Request content blocked by governance policy.")

    def _enforce_rule(self, rule):
        """
        Enforce a specific rule.
        """
        # Role check could be integrated with JWT here, but keeping it simple for now.
        # Ideally, we inspect get_jwt_identity() or verify_jwt_in_request()

        risk = rule.get('risk_level', 'LOW')
        if risk in ['HIGH', 'CRITICAL']:
            # 🛡️ Sentinel: Support for "Break Glass" Human Override
            # This allows authorized operators to bypass governance blocks in emergencies.
            override_token = request.headers.get('X-Governance-Override')
            if override_token:
                # 🛡️ Sentinel: Secure Override Verification
                # We require a signed token to prevent unauthorized overrides.
                # In a real system, this secret would be strictly managed (e.g. Vault).
                secret_key = os.environ.get('GOVERNANCE_OVERRIDE_SECRET', 'dev-secret-do-not-use-in-prod').encode()

                # Format expected: "timestamp:signature"
                try:
                    ts_str, signature = override_token.split(':', 1)
                    timestamp = int(ts_str)

                    # Replay attack prevention (token valid for 5 minutes)
                    if abs(time.time() - timestamp) > 300:
                         raise ValueError("Token expired")

                    # Verify signature
                    payload = f"{ts_str}:{request.path}".encode()
                    expected_signature = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()

                    if not hmac.compare_digest(signature, expected_signature):
                         raise ValueError("Invalid signature")

                    # Import HMMParser lazily to avoid circular dependencies if any
                    try:
                        from core.system.hmm_protocol import HMMParser
                        log_entry = HMMParser.generate_log(
                            action_taken=f"OVERRIDE_GOVERNANCE_BLOCK ({request.method} {request.path})",
                            impact_analysis={
                                "Risk Level": risk,
                                "User IP": request.remote_addr,
                                "Override Token Provided": "YES (VERIFIED)"
                            },
                            audit_link="LOG-FILE"
                        )
                        logging.info(f"\n{log_entry}")
                    except ImportError:
                        # Fallback logging if core is not available
                        logging.warning(f"AUDIT: Governance Override used for {request.path} by {request.remote_addr}")

                    return  # ALLOW the request
                except (ValueError, AttributeError) as e:
                    logging.warning(f"Governance Override Failed: {e}")
                    # Fall through to block

            logging.warning(f"Governance Alert: High risk operation detected on {request.path}")
            # 🛡️ Sentinel: Enforce strict blocking for high-risk operations until role-based access control is fully integrated.
            abort(403, description="Access denied: High risk operation blocked by governance policy. Provide valid signed 'X-Governance-Override' header to bypass.")
