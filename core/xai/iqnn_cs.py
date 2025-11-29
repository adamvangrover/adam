from __future__ import annotations
import numpy as np
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)

class IQNNCS:
    """
    Interpretable Quantum Neural Network for Credit Scoring (IQNN-CS).

    This class implements the conceptual framework for the IQNN-CS architecture described in
    'The Quantum-AI Convergence in Credit Risk' (2025). It focuses on the explainability
    metrics, specifically the Inter-Class Attribution Alignment (ICAA).

    Attributes:
        input_dim (int): Dimensionality of the input data.
        num_classes (int): Number of credit rating classes (e.g., AAA, BBB, Junk).
        feature_names (List[str]): Names of the input features.
    """

    def __init__(self, input_dim: int, num_classes: int, feature_names: List[str] = None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(input_dim)]

        # Placeholder for attribution history: {class_id: [attribution_vectors]}
        self.attribution_history: Dict[int, List[np.ndarray]] = {i: [] for i in range(num_classes)}

        logger.info(f"Initialized IQNN-CS with {input_dim} features and {num_classes} classes.")

    def record_prediction(self, feature_vector: np.ndarray, predicted_class: int, attribution_vector: np.ndarray):
        """
        Records a single prediction and its explanation (attribution vector) for ICAA calculation.

        Args:
            feature_vector (np.ndarray): The input data sample.
            predicted_class (int): The class predicted by the model (quantum or classical).
            attribution_vector (np.ndarray): Feature importance scores (e.g., from SHAP or Gradient).
        """
        if predicted_class not in self.attribution_history:
             self.attribution_history[predicted_class] = []

        self.attribution_history[predicted_class].append(attribution_vector)
        logger.debug(f"Recorded prediction for class {predicted_class}.")

    def calculate_icaa(self) -> Dict[int, float]:
        """
        Calculates the Inter-Class Attribution Alignment (ICAA) metric.

        ICAA measures the divergence in feature attribution within each predicted class.
        High alignment (score close to 1.0) implies consistent reasoning.

        Returns:
            Dict[int, float]: A dictionary mapping class IDs to their ICAA scores.
        """
        icaa_scores = {}

        for class_id, attributions in self.attribution_history.items():
            if not attributions or len(attributions) < 2:
                icaa_scores[class_id] = 0.0
                continue

            # Convert list to matrix for vectorized operations
            attr_matrix = np.vstack(attributions)

            # Normalize rows to unit vectors to calculate cosine similarity
            norms = np.linalg.norm(attr_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10  # Avoid division by zero
            normalized_matrix = attr_matrix / norms

            # Calculate pairwise cosine similarity
            similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

            # Average similarity (excluding self-similarity on diagonal)
            n = len(attributions)
            sum_similarity = np.sum(similarity_matrix) - n
            avg_similarity = sum_similarity / (n * (n - 1))

            icaa_scores[class_id] = float(avg_similarity)

        return icaa_scores

    def generate_explanation_report(self) -> str:
        """
        Generates a human-readable report on the model's interpretability status.
        """
        icaa_scores = self.calculate_icaa()
        report_lines = ["--- IQNN-CS Interpretability Report ---"]

        for class_id, score in icaa_scores.items():
            status = "ROBUST" if score > 0.8 else "INCONSISTENT" if score < 0.5 else "MODERATE"
            report_lines.append(f"Class {class_id} ICAA: {score:.4f} [{status}]")

        avg_score = np.mean(list(icaa_scores.values())) if icaa_scores else 0.0
        report_lines.append(f"Global Model Interpretability Score: {avg_score:.4f}")

        if avg_score > 0.75:
             report_lines.append("CONCLUSION: The model demonstrates consistent economic logic across predictions.")
        else:
             report_lines.append("CONCLUSION: Warning - The model may be overfitting to noise or using inconsistent reasoning.")

        return "\n".join(report_lines)

# Example usage/Test block
if __name__ == "__main__":
    # Simulate a scenario
    iqnn = IQNNCS(input_dim=5, num_classes=3, feature_names=["Leverage", "Liquidity", "Vol", "Sentiment", "Macro"])

    # Simulate Class 0 (Low Risk) - consistently cares about Leverage (idx 0) and Liquidity (idx 1)
    for _ in range(10):
        attr = np.array([0.8, 0.5, 0.1, 0.1, 0.0]) + np.random.normal(0, 0.05, 5)
        iqnn.record_prediction(np.random.rand(5), 0, attr)

    # Simulate Class 2 (High Risk) - Inconsistent reasoning (sometimes Vol, sometimes Macro)
    for _ in range(10):
        if np.random.rand() > 0.5:
            attr = np.array([0.1, 0.1, 0.9, 0.1, 0.0])  # Focus on Vol
        else:
            attr = np.array([0.1, 0.1, 0.0, 0.1, 0.9])  # Focus on Macro
        iqnn.record_prediction(np.random.rand(5), 2, attr)

    print(iqnn.generate_explanation_report())
