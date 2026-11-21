from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum

class EnvironmentMode(str, Enum):
    MOCK = "MOCK"
    LIVE = "LIVE"

class DataConfig(BaseModel):
    dataset_path: str = "showcase/data/artisanal/artisanal_training_data.json"
    validation_split: float = 0.2
    weight_column: str = "weight"

class ModelConfig(BaseModel):
    base_model: str = "Llama-3-8B-Instruct"
    adapter_name: str = "Adam-SLM-Alpha"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

class TrainingPipelineConfig(BaseModel):
    env_mode: EnvironmentMode = EnvironmentMode.MOCK
    run_name: str
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    epochs: int = 3
    learning_rate: float = 2e-4
    output_dir: str = "tinker_lab/pipeline_v2/outputs"
