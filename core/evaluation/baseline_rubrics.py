from core.evaluation.system_judge import EvaluationRubric

# Rubric for evaluating the Market Mayhem Overview
MARKET_MAYHEM_RUBRIC = [
    EvaluationRubric(
        criteria="Data Grounding",
        max_score=10,
        weight=0.4,
        description="The overview must cite specific, accurate market data (e.g., VIX levels, index moves) rather than vague generalizations."
    ),
    EvaluationRubric(
        criteria="Narrative Cohesion",
        max_score=10,
        weight=0.4,
        description="The narrative summary must logically connect the key drivers to the overall market sentiment."
    ),
    EvaluationRubric(
        criteria="Tone and Style",
        max_score=5,
        weight=0.2,
        description="The tone should match the 'Market Mayhem' brand—urgent, insightful, and professional."
    )
]

# Rubric for evaluating Market Predictions
MARKET_PREDICTIONS_RUBRIC = [
    EvaluationRubric(
        criteria="Probabilistic Rigor",
        max_score=10,
        weight=0.5,
        description="Probabilities must be logically justified by the rationale and avoid false certainty (e.g., 100% predictions)."
    ),
    EvaluationRubric(
        criteria="Actionability",
        max_score=10,
        weight=0.3,
        description="The prediction must be specific enough regarding timeframe and asset class to be actionable by an investor."
    ),
    EvaluationRubric(
        criteria="Falsifiability",
        max_score=5,
        weight=0.2,
        description="The prediction must be clearly falsifiable based on defined future events or metrics."
    )
]

# Rubric for evaluating Top Ten Conviction Names
CONVICTION_NAMES_RUBRIC = [
    EvaluationRubric(
        criteria="Thesis Depth",
        max_score=10,
        weight=0.5,
        description="The thesis must go beyond consensus views and highlight specific, differentiated drivers."
    ),
    EvaluationRubric(
        criteria="Catalyst Specificity",
        max_score=10,
        weight=0.3,
        description="Key catalysts must be specific events (e.g., 'Q3 Earnings margin expansion due to new product X') rather than vague trends."
    ),
    EvaluationRubric(
        criteria="Valuation Context",
        max_score=5,
        weight=0.2,
        description="Target prices must be anchored by a coherent valuation framework implied in the thesis."
    )
]
