import json
import logging
import sys
import os

# Ensure the root directory is in the path to allow absolute imports from core_kernel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core_kernel.universal_data_handlers import DataParser

def extract_debt_covenants(filing_text: str) -> str:
    """
    Skill to extract and evaluate debt covenants from text.
    Links the enterprise domain needs to the core execution engine.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("Initializing covenant extraction protocol...")

    # Utilize core kernel capabilities
    parser = DataParser()
    parsed_data = parser.parse_document(filing_text)

    logger.info(f"Document parsed from {parsed_data['metadata']['source']}.")

    # Simulated execution logic
    if "Debt" in parsed_data["parsed_content"] or "Covenant" in parsed_data["parsed_content"]:
        covenant_breach_probability = 0.85
        market_signal = "STRONG_SELL"
        justification = "High leverage and technical default risk identified."
    else:
        covenant_breach_probability = 0.10
        market_signal = "HOLD"
        justification = "No significant debt risks identified."

    result = {
        "target_entity": "Simulated Corp",
        "covenant_breach_probability": covenant_breach_probability,
        "market_divergence_signal": market_signal,
        "justification": justification,
        "provenance_hash": "a1b2c3d4e5f6g7h8i9j0"
    }

    logger.info("Extraction complete. Returning standardized JSON payload.")
    return json.dumps(result, indent=2)

if __name__ == "__main__":
    sample_filing = "The company has entered into a significant Debt agreement with a strict Covenant."
    print("Executing skill: extract_covenants")
    print(extract_debt_covenants(sample_filing))
