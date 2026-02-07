from typing import List, Dict, Any
import logging
from collections import deque
import statistics

class PredictiveScaler:
    """
    Analyzes historical telemetry to predict future resource needs.
    Uses a simple Moving Average model (scaffolding for future ARIMA/LSTM).
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.cpu_history = deque(maxlen=window_size)
        self.gpu_history = deque(maxlen=window_size)

    def ingest_metric(self, cpu: float, gpu: float):
        self.cpu_history.append(cpu)
        self.gpu_history.append(gpu)

    def predict_next_hour(self) -> Dict[str, Any]:
        """
        Predicts average load for the next hour based on trend.
        """
        if len(self.cpu_history) < 2:
            return {"status": "INSUFFICIENT_DATA"}

        avg_cpu = statistics.mean(self.cpu_history)
        avg_gpu = statistics.mean(self.gpu_history)

        # Simple Trend: Last vs Avg
        cpu_trend = self.cpu_history[-1] - avg_cpu
        gpu_trend = self.gpu_history[-1] - avg_gpu

        predicted_cpu = avg_cpu + (cpu_trend * 1.5) # Linear extrapolation
        predicted_gpu = avg_gpu + (gpu_trend * 1.5)

        recommendation = "MAINTAIN"
        if predicted_cpu > 80:
            recommendation = "SCALE_OUT_SOON"
        elif predicted_gpu > 80:
            recommendation = "PROVISION_GPU"

        return {
            "predicted_cpu": round(predicted_cpu, 1),
            "predicted_gpu": round(predicted_gpu, 1),
            "trend": "INCREASING" if cpu_trend > 0 else "STABLE",
            "recommendation": recommendation
        }
