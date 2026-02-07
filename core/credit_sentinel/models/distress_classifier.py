import numpy as np
import logging
from typing import Dict, Any, Optional

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.exceptions import NotFittedError
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)

class DistressClassifier:
    """
    A Random Forest model to predict the probability of corporate distress
    (bankruptcy/default) based on financial ratios.

    Features (X):
    - EBITDA / Interest Expense
    - Net Debt / EBITDA
    - Debt / Equity
    - Current Ratio
    - Return on Assets (ROA)
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        if SKLEARN_AVAILABLE:
            self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            self.scaler = StandardScaler()
            self._initialize_mock_model()
        else:
            logger.warning("scikit-learn not found. DistressClassifier will run in mock mode.")

    def _initialize_mock_model(self):
        """
        Initializes the model with some synthetic data so it's ready to use.
        In production, this would load a pickled model from disk.
        """
        # Synthetic Data: [Cov, Leverage, D/E, Current, ROA]
        X_train = np.array([
            [5.0, 1.5, 0.5, 2.0, 0.10], # Healthy
            [4.0, 2.0, 0.8, 1.5, 0.08], # Healthy
            [1.0, 5.0, 2.5, 0.8, -0.05], # Distressed
            [0.5, 8.0, 4.0, 0.5, -0.15], # Distressed
        ])
        y_train = np.array([0, 0, 1, 1]) # 0 = Safe, 1 = Distress

        self.scaler.fit(X_train)
        self.model.fit(X_train, y_train)
        logger.info("DistressClassifier initialized with synthetic data.")

    def predict_distress(self, ratios: Dict[str, float]) -> Dict[str, Any]:
        """
        Predicts distress probability.

        Args:
            ratios: Dictionary containing keys:
                    ['interest_coverage', 'leverage', 'debt_to_equity', 'current_ratio', 'roa']
        """
        if not SKLEARN_AVAILABLE:
            # Fallback Mock Logic
            score = 0.1
            if ratios.get('interest_coverage', 5) < 1.5: score += 0.4
            if ratios.get('leverage', 2) > 4.0: score += 0.4
            return {"probability": min(0.99, score), "label": "Distressed" if score > 0.5 else "Safe"}

        # Extract features in correct order
        features = [
            ratios.get('interest_coverage', 0.0),
            ratios.get('leverage', 0.0),
            ratios.get('debt_to_equity', 0.0),
            ratios.get('current_ratio', 0.0),
            ratios.get('roa', 0.0)
        ]

        try:
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            prob = self.model.predict_proba(X_scaled)[0][1] # Probability of Class 1 (Distress)

            return {
                "probability": float(prob),
                "label": "Distressed" if prob > 0.5 else "Safe",
                "factors": ratios
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

    def train(self, X: np.ndarray, y: np.ndarray):
        """Retrains the model with new data."""
        if SKLEARN_AVAILABLE:
            self.scaler.fit(X)
            self.model.fit(X, y)
