import json
import datetime
from typing import Dict, Any

class JsonExporter:
    """
    Exports market data and analysis into structured JSON format for downstream consumption.
    """
    def __init__(self, output_dir: str = "showcase/data/adam_daily"):
        self.output_dir = output_dir

    def format_payload(self, report_date: str, risk_score: int, regime: str, assets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Structures the payload to match the MarketMayhemLedger schema.
        """
        data_points = []
        for asset, details in assets.items():
            data_points.append({
                "metric_name": asset,
                "value": details.get("value", 0.0),
                "trend": details.get("trend", "Neutral"),
                "confidence": details.get("confidence", "Moderate")
            })

        return {
            "report_date": report_date,
            "systemic_risk_score": risk_score,
            "macro_regime": regime,
            "data_points": data_points,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }

    def export(self, payload: Dict[str, Any], filename: str = "market_data.json"):
        """
        Writes the JSON payload to disk.
        """
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(payload, f, indent=4)
        return filepath

if __name__ == "__main__":
    exporter = JsonExporter(output_dir="/tmp")
    data = {
        "SPX": {"value": 4500, "trend": "Up", "confidence": "High"},
        "US10Y": {"value": 4.5, "trend": "Up", "confidence": "High"}
    }
    payload = exporter.format_payload("2023-10-15", 65, "Restrictive", data)
    path = exporter.export(payload, "test_export.json")
    print(f"Exported to {path}")
