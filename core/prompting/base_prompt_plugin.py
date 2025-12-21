from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

import yaml
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Define a generic type for the output schema
OutputT = TypeVar("OutputT", bound=BaseModel)

class PromptMetadata(BaseModel):
    """Metadata for tracking prompt lineage and configuration."""
    model_config = ConfigDict(populate_by_name=True)

    prompt_id: str
    version: str = "1.0.0"
    author: str
    # Renamed to llm_config to avoid conflict with Pydantic's model_config
    llm_config: Dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.7, "max_tokens": 1024},
        alias="model_config"
    )
    tags: List[str] = []

class BasePromptPlugin(ABC, Generic[OutputT]):
    """
    Abstract Base Class for Prompt-as-Code plugins.

    Lifecycle:
    1. validate_inputs(inputs) -> Checks if input vars match the schema.
    2. render(inputs) -> Compiles Jinja2 template into a raw string.
    3. [External LLM Call happens here]
    4. parse_response(raw_text) -> Converts LLM string output to Pydantic object.
    """

    def __init__(
        self,
        metadata: PromptMetadata,
        template_string: Optional[str] = None,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None
    ):
        self.metadata = metadata
        self.template_engine = Template(template_string) if template_string else None
        self.system_engine = Template(system_template) if system_template else None
        self.user_engine = Template(user_template) if user_template else None

        if not self.template_engine and not (self.system_engine or self.user_engine):
            raise ValueError("Must provide either template_string OR system/user templates.")

    @abstractmethod
    def get_input_schema(self) -> type[BaseModel]:
        """Returns the Pydantic model required for input variables."""
        pass

    @abstractmethod
    def get_output_schema(self) -> type[OutputT]:
        """Returns the Pydantic model expected for the output."""
        pass

    @classmethod
    def from_yaml(cls, filepath: Union[str, Path]) -> BasePromptPlugin:
        """
        Factory method to instantiate a plugin from a YAML configuration file.
        The YAML must contain 'prompt_id', 'version', 'author', 'model_config',
        and 'template_body' (or 'system_template'/'user_template').
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            config = yaml.safe_load(f)

        metadata = PromptMetadata(
            prompt_id=config.get('prompt_id', 'unknown'),
            version=config.get('version', '1.0.0'),
            author=config.get('author', 'system'),
            model_config=config.get('model_config', {})
        )

        return cls(
            metadata=metadata,
            template_string=config.get('template_body'),
            system_template=config.get('system_template'),
            user_template=config.get('user_template')
        )

    def validate_inputs(self, inputs: Dict[str, Any]) -> BaseModel:
        """Strict validation of runtime variables against schema."""
        Schema = self.get_input_schema()
        try:
            return Schema(**inputs)
        except ValidationError as e:
            raise ValueError(f"Input Validation Failed for {self.metadata.prompt_id}: {e}")

    def render(self, inputs: Dict[str, Any]) -> str:
        """
        Compiles the prompt template with validated inputs.
        If system/user templates are used, they are concatenated.
        """
        validated_data = self.validate_inputs(inputs)
        context = validated_data.model_dump()

        if self.template_engine:
            return self.template_engine.render(**context)

        parts = []
        if self.system_engine:
            parts.append(f"SYSTEM: {self.system_engine.render(**context)}")
        if self.user_engine:
            parts.append(f"USER: {self.user_engine.render(**context)}")
        return "\n\n".join(parts)

    def render_messages(self, inputs: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Renders the prompt as a list of messages for Chat APIs (e.g., OpenAI).
        """
        validated_data = self.validate_inputs(inputs)
        context = validated_data.model_dump()
        messages = []

        if self.system_engine:
            messages.append({"role": "system", "content": self.system_engine.render(**context)})

        if self.user_engine:
            messages.append({"role": "user", "content": self.user_engine.render(**context)})

        if not messages and self.template_engine:
            # Fallback: Treat the whole string as a user message or parse it?
            # For strict correctness, we just return it as user content if unstructured.
            messages.append({"role": "user", "content": self.template_engine.render(**context)})

        return messages

    def parse_response(self, raw_response: str) -> OutputT:
        """
        Parses raw LLM output into the strict output schema.
        Assumes LLM returns JSON. Override if parsing unstructured text.
        """
        Schema = self.get_output_schema()
        try:
            # sanitize markdown code blocks if present
            clean_json = raw_response.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_json)
            return Schema(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Output Parsing Failed: {e}")

    def to_audit_log(self, inputs: Dict[str, Any], raw_output: str) -> str:
        """Generates a JSONL formatted string for logging."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prompt_id": self.metadata.prompt_id,
            "version": self.metadata.version,
            "inputs": inputs,
            "raw_output": raw_output,
            "model_config": self.metadata.llm_config
        }
        return json.dumps(log_entry)
