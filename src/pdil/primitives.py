from typing import Any, Dict

class Primitive:
    """Base class for all PDIL primitives."""
    pass

class DataPrimitive(Primitive):
    """Primitive for handling data inputs."""
    def __init__(self, data: Dict[str, Any]):
        self.data = data
