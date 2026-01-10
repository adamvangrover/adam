from enum import Enum
from pydantic import BaseModel, Field

class ServiceType(str, Enum):
    CORE_BRAIN = "core_brain"
    INGESTION_ENGINE = "ingestion_engine"
    SIMULATION_ENGINE = "simulation_engine"

class ServiceDefinition(BaseModel):
    name: ServiceType
    description: str
    docker_image: str
    port: int
    env_vars: list[str] = Field(default_factory=list)

# Define the standard services for the microservices architecture
CORE_BRAIN_SERVICE = ServiceDefinition(
    name=ServiceType.CORE_BRAIN,
    description="Handles orchestration, API requests, and LLM communication.",
    docker_image="adam-core:latest",
    port=8000,
    env_vars=["OPENAI_API_KEY", "POSTGRES_URL", "REDIS_URL"]
)

INGESTION_ENGINE_SERVICE = ServiceDefinition(
    name=ServiceType.INGESTION_ENGINE,
    description="Handles heavy document processing, OCR, and embedding generation.",
    docker_image="adam-ingest:latest",
    port=8001,
    env_vars=["NEO4J_URL", "NEO4J_PASSWORD"]
)

SIMULATION_ENGINE_SERVICE = ServiceDefinition(
    name=ServiceType.SIMULATION_ENGINE,
    description="Runs Monte Carlo simulations and quantitative risk models.",
    docker_image="adam-quant:latest",
    port=8002,
    env_vars=["NUM_THREADS"]
)
