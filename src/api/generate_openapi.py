import json
from src.schemas.core_types import AgentInput, AgentOutput
from pydantic.json_schema import models_json_schema

def generate_openapi():
    _, top_level_schema = models_json_schema(
        [(AgentInput, "validation"), (AgentOutput, "validation")],
        title="Adam API Reference"
    )
    print(json.dumps(top_level_schema, indent=2))

if __name__ == "__main__":
    generate_openapi()
