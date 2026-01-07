from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field, ValidationError

from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata
from core.prompting.json_rpc_library import TEMPLATE_REGISTRY
from core.schemas.json_rpc import JsonRpcRequest, AdaptiveConvictionMetadata

class JsonRpcInput(BaseModel):
    """Generic input schema for JSON-RPC prompts."""
    topic: Optional[str] = None
    task: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    sources: Optional[List[str]] = None
    tools: Optional[str] = None # JSON string of tools
    history: Optional[str] = None

class JsonRpcOutput(BaseModel):
    """
    The expected output from the LLM when using Adaptive Conviction.
    It combines the AdaptiveConvictionMetadata with the actual Action.
    """
    thought_trace: Optional[str] = None
    conviction_score: float
    action: Dict[str, Any] # Can be JsonRpcRequest or clarification dict

class JsonRpcPromptPlugin(BasePromptPlugin[JsonRpcOutput]):
    """
    Plugin for executing JSON-RPC compatible prompts from the library.
    """

    def get_input_schema(self) -> Type[BaseModel]:
        return JsonRpcInput

    def get_output_schema(self) -> Type[JsonRpcOutput]:
        return JsonRpcOutput

    @classmethod
    def from_registry(cls, template_name: str, author: str = "system") -> 'JsonRpcPromptPlugin':
        """
        Factory to create a plugin instance from the TEMPLATE_REGISTRY.
        """
        if template_name not in TEMPLATE_REGISTRY:
            raise ValueError(f"Template '{template_name}' not found in registry.")

        config = TEMPLATE_REGISTRY[template_name]

        metadata = PromptMetadata(
            prompt_id=config["name"],
            version="1.0.0",
            author=author,
            description=config.get("description", "")
        )

        return cls(
            metadata=metadata,
            template_string=config["template"]
        )
