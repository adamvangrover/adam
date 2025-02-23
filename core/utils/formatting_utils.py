//formatting_utils.py
from flask import jsonify

def format_data(data, format="json"):
  """
  Formats data into the specified format (default: JSON).
  """
  if format == "json":
    return jsonify(data)
  elif format == "csv":
    # Convert data to CSV format
    pass  # Placeholder for CSV conversion logic
  else:
    raise ValueError("Invalid format.")

# Add more formatting utility functions as needed
