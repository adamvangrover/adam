from typing import Dict, Any, Type
from pydantic import BaseModel

class SchemaRegistry:
    """
    Minimal placeholder for SchemaRegistry.
    """
    _registry: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, name: str, schema: Type[BaseModel]):
        cls._registry[name] = schema

    @classmethod
    def get(cls, name: str) -> Type[BaseModel]:
        return cls._registry.get(name)
