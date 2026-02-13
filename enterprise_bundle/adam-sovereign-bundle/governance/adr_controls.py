import sys
import json

def audit_citation_density(memo_text):
    """
    Control 03: Ensures the AI is not just chatting, but citing.
    Minimum standard: 1 citation per 2 sentences.
    """
    sentences = memo_text.count('.')
    citations = memo_text.count('[doc_')

    if sentences == 0: return True
    density = citations / sentences

    print(f"DEBUG: Citation Density = {density:.2f}")
    if density < 0.5:
        return False, "Insufficient Evidence Linking"
    return True, "Passed"

def audit_financial_math(spread_json):
    """
    Control 04: Accounting Identity Check.
    """
    data = json.loads(spread_json)
    assets = data.get('total_assets', 0)
    liabilities = data.get('total_liabilities', 0)
    equity = data.get('total_equity', 0)

    if abs(assets - (liabilities + equity)) > 1.0:
        return False, f"Balance Sheet Mismatch: {assets - (liabilities + equity)}"
    return True, "Passed"

if __name__ == "__main__":
    # Example runner logic
    pass
