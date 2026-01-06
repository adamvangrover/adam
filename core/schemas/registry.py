from typing import Dict, Any

class SchemaRegistry:
    _registry: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, schema: Any):
        cls._registry[name] = schema

    @classmethod
    def get(cls, name: str) -> Any:
        return cls._registry.get(name)
