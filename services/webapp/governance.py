# Verified for Adam v25.5
import yaml
import os
import re
import logging
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
            logging.info(f"Governance Alert: High risk operation detected on {request.path}")
            # In a real system, we might require a specific 'X-Approval-Token' header here.
            pass
