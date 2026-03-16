# core/agents/machine_learning_model_training_agent.py

import asyncio
import logging
import os
import pickle
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from core.agents.agent_base import AgentBase

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MachineLearningModelTrainingAgent(AgentBase):
    """
    Agent responsible for training and managing machine learning models.
    """

    def __init__(self, config: Dict[str, Any], kernel: Optional[Any] = None):
        """
        Initializes the ML Training Agent.

        Args:
            config (dict): Configuration dictionary.
            kernel (Optional[Any]): Semantic Kernel instance.
        """
        super().__init__(config, kernel=kernel)
        # Initialize machine learning models and parameters if needed

    async def execute(self, *args, **kwargs):
        """
        Trains and evaluates a machine learning model.

        Args:
            data_sources (list): Sources of data via kwargs.
            model_type (str): Type of model via kwargs.
            model_name (str): Name for saving model via kwargs.

        Returns:
            dict: Training results.
        """
        data_sources = kwargs.get('data_sources')
        model_type = kwargs.get('model_type')
        model_name = kwargs.get('model_name')

        logger.info(f"MLTrainingAgent starting training for {model_name} ({model_type})")

        if not data_sources:
            return {"error": "No data sources provided."}

        try:
            loop = asyncio.get_running_loop()

            # Execute training pipeline in thread pool
            result = await loop.run_in_executor(None, self._run_pipeline, data_sources, model_type, model_name)
            return result

        except Exception as e:
            logger.error(f"Training error: {e}")
            return {"error": str(e)}

    def _run_pipeline(self, data_sources, model_type, model_name):
        # Load and preprocess the data
        data = self.load_data(data_sources)
        if data is None or data.empty:
             raise ValueError("Data loading failed or empty data.")

        preprocessed_data = self.preprocess_data(data)

        # Train the model
        model = self.train_model(preprocessed_data, model_type)

        # Evaluate the model
        metrics = self.evaluate_model(model, preprocessed_data)

        # Save the model
        self.save_model(model, model_name)

        return {"status": "success", "model_name": model_name, "metrics": metrics}

    def load_data(self, data_sources):
        """
        Loads data from the specified sources.
        """
        # Placeholder: Assume first source is a csv path or DataFrame
        source = data_sources[0]
        if isinstance(source, str) and source.endswith('.csv'):
            if os.path.exists(source):
                return pd.read_csv(source)
            else:
                logger.warning(f"File not found: {source}")
                return None
        elif isinstance(source, pd.DataFrame):
            return source

        # Mock data for testing if no file
        return pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [2, 4, 6, 8, 10]
        })

    def preprocess_data(self, data):
        """
        Preprocesses the data for model training.
        """
        # Minimal preprocessing
        return data.dropna()

    def train_model(self, data, model_type):
        """
        Trains a machine learning model of the specified type.
        """
        X = data.drop('target', axis=1, errors='ignore')
        # If no target column, assume last column is target for simplicity
        if 'target' not in data.columns:
             X = data.iloc[:, :-1]
             y = data.iloc[:, -1]
        else:
             y = data['target']

        if model_type == "linear_regression":
            model = LinearRegression()
            model.fit(X, y)
            return model
        elif model_type == "decision_tree":
            model = DecisionTreeRegressor()
            model.fit(X, y)
            return model

        logger.warning(f"Unknown model type {model_type}, defaulting to LinearRegression")
        model = LinearRegression()
        model.fit(X, y)
        return model

    def evaluate_model(self, model, data):
        """
        Evaluates the performance of the trained model.
        """
        X = data.drop('target', axis=1, errors='ignore')
        if 'target' not in data.columns:
             X = data.iloc[:, :-1]
             y = data.iloc[:, -1]
        else:
             y = data['target']

        score = model.score(X, y)
        logger.info(f"Model Score (R^2): {score}")
        return {"r2_score": score}

    def save_model(self, model, model_name):
        """
        Saves the trained model to a file.
        """
        os.makedirs("models", exist_ok=True)
        path = f"models/{model_name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"Model saved to {path}")

if __name__ == "__main__":
    agent = MachineLearningModelTrainingAgent({})
    async def main():
        res = await agent.execute(
            data_sources=["mock_data"],
            model_type="linear_regression",
            model_name="test_model"
        )
        print(res)
    asyncio.run(main())
