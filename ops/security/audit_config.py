# ops/security/audit_config.py

import logging
import os
import re
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SECURITY] - %(levelname)s - %(message)s')
logger = logging.getLogger("SecurityAudit")

# Patterns to detect potential secrets
SECRET_PATTERNS = [
    (r"sk-[a-zA-Z0-9]{32,}", "Potential OpenAI API Key"),
    (r"xox[baprs]-[0-9]{12}-[0-9]{12}-[a-zA-Z0-9]{24}", "Potential Slack Token"),
    (r"ghp_[a-zA-Z0-9]{36}", "Potential GitHub Token"),
    (r"AIza[0-9A-Za-z-_]{35}", "Potential Google API Key"),
]

def scan_file(filepath):
    issues = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            for pattern, description in SECRET_PATTERNS:
                if re.search(pattern, content):
                    issues.append(f"{description} found in {filepath}")
    except Exception as e:
        logger.warning(f"Could not scan {filepath}: {e}")
    return issues

def audit_configs(config_dir="config"):
    logger.info(f"Starting Security Audit on directory: {config_dir}")
    failed = False

    for root, _, files in os.walk(config_dir):
        for file in files:
            if file.endswith(('.yaml', '.yml', '.json', '.py', '.ini')):
                filepath = os.path.join(root, file)
                issues = scan_file(filepath)
                if issues:
                    for issue in issues:
                        logger.error(issue)
                    failed = True
                else:
                    logger.debug(f"Scanned {file}: Clean")

    if failed:
        logger.error("Security Audit FAILED: Secrets detected in configuration files.")
        sys.exit(1)
    else:
        logger.info("Security Audit PASSED: No secrets detected.")
        sys.exit(0)

if __name__ == "__main__":
    audit_configs()
