from typing import Dict, Type, Optional
from pydantic import BaseModel

class SchemaRegistry:
    """
    Central registry for Pydantic schemas.
    Merged: Prioritizes the strict typing from 'main' (Type[BaseModel]) to ensure
    data integrity, while keeping the simple interface from the guide.
    """
    # Adopted '_registry' from main as it is more semantically accurate than '_schemas'
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, schema: Type[BaseModel]):
        """Registers a schema model under a unique name."""
        cls._registry[name] = schema

    @classmethod
    def get(cls, name: str) -> Optional[Type[BaseModel]]:
        """Retrieves a schema model by name."""
        return cls._registry.get(name)