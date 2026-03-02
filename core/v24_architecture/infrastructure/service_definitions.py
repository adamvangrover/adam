from enum import Enum
from pydantic import BaseModel

class ServiceType(Enum):
    CORE_BRAIN = "core_brain"
    INGESTION_ENGINE = "ingestion_engine"
    SIMULATION_ENGINE = "simulation_engine"

class ServiceConfig(BaseModel):
    service_name: ServiceType
    host: str
    port: int
    workers: int = 1

class InfrastructureConfig:
    """
    Defines the microservices topology.
    """
    SERVICES = {
        ServiceType.CORE_BRAIN: ServiceConfig(
            service_name=ServiceType.CORE_BRAIN,
            host="0.0.0.0",  # nosec B104
            port=8000
        ),
        ServiceType.INGESTION_ENGINE: ServiceConfig(
            service_name=ServiceType.INGESTION_ENGINE,
            host="ingest-service",
            port=8001,
            workers=4 # Heavy lifting
        ),
        ServiceType.SIMULATION_ENGINE: ServiceConfig(
            service_name=ServiceType.SIMULATION_ENGINE,
            host="quant-service",
            port=8002,
            workers=2
        )
    }

    @staticmethod
    def get_config(service_type: ServiceType) -> ServiceConfig:
        return InfrastructureConfig.SERVICES[service_type]
