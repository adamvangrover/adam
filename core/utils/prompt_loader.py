import yaml
import logging
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class PromptConfig(BaseModel):
    id: str
    version: str
    owner: str
    description: str
    llm_settings: Dict[str, Any] = Field(alias="model_config")
    input_variables: List[Dict[str, str]]
    system_template: str
    user_template: str

class PromptLoader:
    """
    Loads and validates prompts from the YAML registry.
    Implements 'Prompt-as-Code' by treating prompts as configuration artifacts.
    """

    def __init__(self, registry_path: str = "prompt_library"):
        self.registry_path = registry_path
        self.cache: Dict[str, PromptConfig] = {}

    def load_prompt(self, prompt_id: str, version: Optional[str] = None) -> PromptConfig:
        """
        Loads a specific prompt version.
        If version is None, attempts to find the latest version (simplified logic: explicit path required for now).
        """
        try:
            # Hardcoded mapping for the demo to ensure it works
            if prompt_id == "commercial_credit_risk_analysis":
                filename = "credit/commercial_credit_v1.yaml"
            else:
                raise FileNotFoundError(f"Prompt ID {prompt_id} not found in registry mapping.")

            full_path = f"{self.registry_path}/{filename}"

            with open(full_path, 'r') as f:
                data = yaml.safe_load(f)

            # Validate against schema
            # We need to handle the alias correctly.
            # Pydantic v2 model_validate or just passing dict works if populated correctly.
            config = PromptConfig(**data)

            if version and config.version != version:
                 logging.warning(f"Requested version {version} but loaded {config.version}")

            self.cache[prompt_id] = config
            return config

        except Exception as e:
            logging.error(f"Failed to load prompt {prompt_id}: {e}")
            raise

    def render_messages(self, prompt_config: PromptConfig, inputs: Dict[str, Any]) -> list:
        """
        Renders the system and user messages with the provided inputs.
        """
        system_msg = prompt_config.system_template
        user_msg = prompt_config.user_template

        for var in prompt_config.input_variables:
            var_name = var['name']
            if var_name in inputs:
                user_msg = user_msg.replace(f"{{{{{var_name}}}}}", str(inputs[var_name]))
            else:
                # Optional warning, maybe some vars are optional
                pass

        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
