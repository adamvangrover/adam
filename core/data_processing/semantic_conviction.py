import numpy as np
from typing import List, Union

def calculate_conviction(scores: Union[List[float], np.ndarray]) -> float:
    """
    Calculates a conviction score based on a list of individual scores.
    """
    if not scores:
        return 0.0

    # Ensure scores is a numpy array
    scores_array = np.array(scores)

    # Calculate the mean score
    mean_score = np.mean(scores_array)

    # Calculate the standard deviation
    std_dev = np.std(scores_array)

    # A simple conviction metric: mean - (0.5 * std_dev)
    # This penalizes high variance in the scores.
    conviction = mean_score - (0.5 * std_dev)

    # Normalize to 0-1 range
    return max(0.0, min(1.0, conviction))

def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def semantic_conviction(logits: np.ndarray) -> float:
    """
    Calculates semantic conviction from logits.
    """
    probabilities = softmax(logits)
    entropy = -np.sum(probabilities * np.log(probabilities + 1e-9))
    max_entropy = np.log(len(logits))
    normalized_entropy = entropy / max_entropy
    conviction = 1.0 - normalized_entropy
    return conviction

def get_semantic_conviction(scores: List[float]) -> float:
    """
    Wrapper for semantic conviction calculation.
    """
    try:
        # Check if scores is a list of lists or just a list
        if isinstance(scores[0], list):
             # Flatten the list if it is a list of lists
            flat_scores = [item for sublist in scores for item in sublist]
            scores_array = np.array(flat_scores)
        else:
            scores_array = np.array(scores)

        # Assuming scores are logits for semantic conviction
        # If they are just raw scores, we might need to interpret them differently
        # For this example, let's treat them as logits if they are not normalized

        # Determine if we should treat as probabilities or logits
        # Simple heuristic: if any value > 1 or < 0, treat as logits
        if np.any(scores_array > 1.0) or np.any(scores_array < 0.0):
             return semantic_conviction(scores_array)
        else:
             # Already probabilities? Or just scores?
             # Let's use simple variance based conviction
             return calculate_conviction(scores)

    except Exception as e:
        print(f"Error calculating semantic conviction: {e}")
        return 0.0

def hybrid_conviction(
    semantic_score: float,
    statistical_score: float,
    semantic_weight: float = 0.5
) -> float:
    """
    Combines semantic and statistical conviction scores.
    """
    return (semantic_score * semantic_weight) + (statistical_score * (1 - semantic_weight))

def test_semantic_conviction():
    # Test case 1: High conviction (one high score)
    logits1 = np.array([10.0, 1.0, 1.0])
    print(f"Logits: {logits1}, Conviction: {semantic_conviction(logits1)}")

    # Test case 2: Low conviction (uniform scores)
    logits2 = np.array([1.0, 1.0, 1.0])
    print(f"Logits: {logits2}, Conviction: {semantic_conviction(logits2)}")

    scores = [0.8, 0.9, 0.7, 0.85]
    print(f"Scores: {scores}, Conviction: {calculate_conviction(scores)}")

if __name__ == "__main__":
    test_semantic_conviction()
