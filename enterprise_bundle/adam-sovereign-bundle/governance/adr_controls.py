import sys
import json
import re

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

def audit_tone_check(memo_text):
    """
    Control 05: Sentiment Analysis.
    Ensures tone is professional (no overly emotive language).
    """
    emotive_words = ["amazing", "terrible", "skyrocketed", "plummeted", "fantastic", "horrible", "insane"]
    found_words = []
    for word in emotive_words:
        if re.search(r'\b' + re.escape(word) + r'\b', memo_text.lower()):
            found_words.append(word)

    if found_words:
        return False, f"Unprofessional tone detected: {', '.join(found_words)}"
    return True, "Passed"

def audit_absolute_statements(memo_text):
    """
    Control 06: Risk Management.
    Flags absolute claims like 'guaranteed', 'impossible'.
    """
    absolute_terms = ["guaranteed", "impossible", "certainly", "undoubtedly", "always", "never"]
    found_terms = []
    for term in absolute_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', memo_text.lower()):
            found_terms.append(term)

    if found_terms:
        return False, f"Absolute statement detected: {', '.join(found_terms)}"
    return True, "Passed"

if __name__ == "__main__":
    # Example runner logic
    pass
