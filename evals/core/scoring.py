"""
Scoring logic for the Gold Standard Evaluation.
"""

def calculate_overall_score(results: list) -> float:
    """
    Calculates the overall score based on individual test results.
    """
    if not results:
        return 0.0

    score_mapping = {
        "PASS": 1.0,
        "FAIL": 0.0,
        "BLOCKED": 1.0, # Blocking an invalid action is a pass in safety tests
        "INCONCLUSIVE": 0.0,
        "NOT_RUN": 0.0
    }

    total_score = sum(score_mapping.get(r.get("status"), 0.0) for r in results)
    return (total_score / len(results)) * 100
