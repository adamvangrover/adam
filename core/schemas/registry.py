from typing import Dict, Any, Type

class SchemaRegistry:
    _schemas: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, schema: Any):
        cls._schemas[name] = schema

    @classmethod
    def get(cls, name: str):
        return cls._schemas.get(name)
