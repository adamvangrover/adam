# core/utils/api_utils.py

from typing import Dict, Any, List

class APIValidator:
    """
    Validates API request data structures.
    """
    @staticmethod
    def validate(request_data: Dict[str, Any], required_parameters: List[str]) -> bool:
        """
        Validates API request data against a list of required parameters.
        Raises ValueError if a required parameter is missing to maintain legacy compatibility.
        """
        for param in required_parameters:
            if param not in request_data:
                raise ValueError(f"Missing required parameter: {param}")
        return True

def validate_api_request(request_data: Dict[str, Any], required_parameters: List[str]):
    """
    Legacy wrapper for API validation.
    """
    APIValidator.validate(request_data, required_parameters)

# Add more API utility functions as needed
