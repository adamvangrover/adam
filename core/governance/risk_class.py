from enum import Enum

class RiskClass(str, Enum):
    READ = "read"
    COMPUTE = "compute"
    WRITE_LOCAL = "write_local"
    WRITE_STATE = "write_state"
    EXTERNAL = "external"
    FINANCIAL = "financial"
    ADMIN = "admin"
