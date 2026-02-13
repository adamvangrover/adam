#!/usr/bin/env python3
"""
Demo Script: Adam Sovereign Credit Bundle (v1.0.0)
--------------------------------------------------
This script demonstrates the "Glass Box" system integration by:
1. Loading the Sovereign Bundle configuration.
2. Instantiating the specialized agents (Quant, Risk Officer).
3. Running "Audit-as-Code" checks on a Golden Dataset.

Usage:
    python scripts/demo_sovereign_bundle.py
"""

import sys
import os
import yaml
import json
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AdamSovereignDemo")

# Path Configuration
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUNDLE_ROOT = os.path.join(REPO_ROOT, "enterprise_bundle", "adam-sovereign-bundle")
GOVERNANCE_DIR = os.path.join(BUNDLE_ROOT, "governance")

# Add Governance directory to sys.path to allow importing adr_controls
sys.path.append(GOVERNANCE_DIR)

try:
    import adr_controls
except ImportError as e:
    logger.error(f"Failed to import adr_controls from {GOVERNANCE_DIR}. Error: {e}")
    sys.exit(1)

def load_yaml(filepath):
    try:
        with open(filepath, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None

def main():
    print("\n" + "="*60)
    print("   ADAM SOVEREIGN CREDIT BUNDLE (v1.0.0) DEMO")
    print("="*60 + "\n")

    # ---------------------------------------------------------
    # 1. Load Manifest
    # ---------------------------------------------------------
    logger.info("Step 1: Loading Sovereign Manifest...")
    manifest_path = os.path.join(BUNDLE_ROOT, "manifest.yaml")
    manifest = load_yaml(manifest_path)

    if manifest:
        logger.info(f"Loaded Manifest: {manifest.get('package')} v{manifest.get('version')}")
        logger.info(f"Security Constraints: {json.dumps(manifest.get('security_constraints'), indent=2)}")
    else:
        logger.error("Failed to load manifest.")
        return

    # ---------------------------------------------------------
    # 2. Instantiate Agents
    # ---------------------------------------------------------
    logger.info("\nStep 2: Instantiating Agents...")
    agents_dir = os.path.join(BUNDLE_ROOT, "agents")
    agent_files = ["quant.yaml", "risk_officer.yaml", "archivist.yaml"]

    for agent_file in agent_files:
        agent_path = os.path.join(agents_dir, agent_file)
        agent_config = load_yaml(agent_path)
        if agent_config:
            logger.info(f"Loaded Agent: {agent_config.get('id')} ({agent_config.get('role')})")
            # In a real system, we would instantiate the agent class here
            # e.g., agent = AgentFactory.create(agent_config)

    # ---------------------------------------------------------
    # 3. Run Audit-as-Code (Golden Dataset)
    # ---------------------------------------------------------
    logger.info("\nStep 3: Running Audit-as-Code checks...")
    golden_dataset_path = os.path.join(GOVERNANCE_DIR, "golden_dataset.jsonl")

    with open(golden_dataset_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            case = json.loads(line)
            logger.info(f"Test Case: {case['test_case']}")

            # Run Financial Math Check
            # Note: adr_controls.audit_financial_math expects a JSON string, which is what 'input' is in our dataset
            input_json_str = case['input']
            if isinstance(input_json_str, dict): # Handle if it was parsed as dict
                input_json_str = json.dumps(input_json_str)

            is_valid, message = adr_controls.audit_financial_math(input_json_str)

            expected = case.get('expected')
            status = "PASS" if (is_valid and expected == "PASS") or (not is_valid and expected == "FAIL") else "FAIL"

            logger.info(f"  Result: {message}")
            logger.info(f"  Status: {status} (Expected: {expected})")

    # ---------------------------------------------------------
    # 4. Run Citation Density Check (Example)
    # ---------------------------------------------------------
    logger.info("\nStep 4: verifying Citation Density Control...")

    good_text = "The revenue increased by 10% [doc_1:chunk_5]. Operating margin remained flat [doc_1:chunk_6]."
    bad_text = "The revenue increased by 10%. Operating margin remained flat. I think this is good."

    logger.info(f"Text 1: {good_text}")
    valid1, msg1 = adr_controls.audit_citation_density(good_text)
    logger.info(f"  Audit: {msg1}")

    logger.info(f"Text 2: {bad_text}")
    valid2, msg2 = adr_controls.audit_citation_density(bad_text)
    logger.info(f"  Audit: {msg2}")

    logger.info("\nDemo Complete.")

if __name__ == "__main__":
    main()
