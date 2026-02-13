import os
import yaml
import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class PromptModelConfig(BaseModel):
    provider: str
    model: str
    temperature: float
    top_p: float

class PromptInput(BaseModel):
    name: str
    type: str

class PromptDefinition(BaseModel):
    id: str
    version: str
    owner: str
    description: str
    llm_config: PromptModelConfig = Field(alias="model_config")
    input_variables: list[PromptInput]
    system_template: str
    user_template: str

class PromptRegistry:
    """
    Manages the lifecycle and retrieval of versioned prompts.
    Protocol: ADAM-V-NEXT (Prompt-as-Code)
    """
    def __init__(self, prompts_dir: str = None):
        if prompts_dir is None:
            self.prompts_dir = os.path.join(os.path.dirname(__file__), 'prompts')
        else:
            self.prompts_dir = prompts_dir
        self._prompts: Dict[str, PromptDefinition] = {}
        self._load_prompts()

    def _load_prompts(self):
        """Loads all YAML prompts from the directory."""
        if not os.path.exists(self.prompts_dir):
            logging.warning(f"Prompts directory not found: {self.prompts_dir}")
            return

        for filename in os.listdir(self.prompts_dir):
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                filepath = os.path.join(self.prompts_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = yaml.safe_load(f)
                        prompt = PromptDefinition(**data)
                        # Store by ID (simple registry for now, could be id:version)
                        self._prompts[prompt.id] = prompt
                        logging.info(f"Loaded prompt: {prompt.id} v{prompt.version}")
                except Exception as e:
                    logging.error(f"Failed to load prompt {filename}: {e}")

    def get_prompt(self, prompt_id: str) -> Optional[PromptDefinition]:
        """Retrieves a prompt by ID."""
        return self._prompts.get(prompt_id)

    def render_prompt(self, prompt_id: str, variables: Dict[str, Any]) -> Dict[str, str]:
        """
        Renders the system and user templates with the provided variables.
        """
        prompt_def = self.get_prompt(prompt_id)
        if not prompt_def:
            raise ValueError(f"Prompt {prompt_id} not found.")

        # Simple Jinja2-style replacement (double curly braces)
        user_content = prompt_def.user_template
        for var in prompt_def.input_variables:
            key = var.name
            val = variables.get(key, f"MISSING_VAR:{key}")
            user_content = user_content.replace(f"{{{{{key}}}}}", str(val))

        return {
            "system": prompt_def.system_template,
            "user": user_content,
            "config": prompt_def.llm_config.model_dump(),
            "version": prompt_def.version
        }

# Global Instance
registry = PromptRegistry()
