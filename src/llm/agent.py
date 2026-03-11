from pydantic_ai import Agent
try:
    from pydantic_ai.providers.litellm import LiteLLMProvider
except ImportError:
    # Older versions of pydantic-ai don't have LiteLLMProvider
    LiteLLMProvider = None

from .schemas import SpreadsheetBatchOutput
from src.core.config import settings

def get_agent(model_name: str = "openai/gpt-4o") -> Agent:
    """
    Initializes a PydanticAI Agent configured with the LiteLLM Provider.
    This architecture handles proxy routing, multi-model execution, and
    enforces structural determinism using SpreadsheetBatchOutput.

    Args:
        model_name: The LiteLLM formatted model name.

    Returns:
        A configured Agent instance ready to process tabular data.
    """

    system_prompt = (
        "You are an expert data analyst and ingestion engine. "
        "Your goal is to process the provided spreadsheet data (formatted as Markdown tables). "
        "You must output a strictly typed JSON object conforming to the SpreadsheetBatchOutput schema. "
        "For each row, provide the original ID (if applicable), extracted entities, sentiment score, "
        "and any translated text or column transformations based on the user's instructions. "
        "Ensure the structure exactly matches the requested output schema."
    )

    if LiteLLMProvider is not None:
        # Provider configured through LiteLLM for universal access
        provider = LiteLLMProvider(
            model_name=model_name,
            # LiteLLM automatically retrieves API keys from environment if set,
            # but we can explicitly pass them if needed.
            # api_key=settings.openai_api_key
        )

        # Create the agent with strict result type
        agent = Agent(
            provider,
            result_type=SpreadsheetBatchOutput,
            system_prompt=system_prompt,
            retries=3 # Automatic retries on structural failure
        )
    else:
        # Older version of pydantic-ai, just use the string identifier
        agent = Agent(
            model_name,
            result_type=SpreadsheetBatchOutput,
            system_prompt=system_prompt,
            retries=3 # Automatic retries on structural failure
        )

    return agent
