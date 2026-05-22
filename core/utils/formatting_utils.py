# core/utils/formatting_utils.py

from flask import jsonify


def format_data(data, format="json"):
    """
    Formats data into the specified format (default: JSON).
    """
    match format:
        case "json":
            return jsonify(data)
        case "csv":
            # Convert data to CSV format
            pass  # Placeholder for CSV conversion logic
        case _:
            raise ValueError("Invalid format.")

# Add more formatting utility functions as needed
