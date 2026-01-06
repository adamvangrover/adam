from typing import Dict, Any, Type, Optional
from pydantic import BaseModel

class SchemaRegistry:
    """
    Central registry for Pydantic schemas.
    Merged: Uses strict typing (main) to ensure only BaseModels are registered,
    improving type safety across the autopoietic system.
    """
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, schema: Type[BaseModel]):
        """Registers a schema model under a unique name."""
        cls._registry[name] = schema

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        """Retrieves a schema model by name."""
        return cls._registry.get(name)