# core/llm_plugin.py
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv
import yaml  #if needed
from core.utils.config_utils import load_config  # assuming load_config is now implemented

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class LLMPluginError(Exception):
    """Base class for LLM plugin exceptions."""
    pass

class LLMConfigurationError(LLMPluginError):
    """Raised for configuration issues."""
    pass

class LLMAPIError(LLMPluginError):
    """Raised for errors during API calls."""
    pass

class BaseLLMPlugin(ABC):
    """Abstract base class for all LLM plugins."""

    @abstractmethod
    def query(self, prompt: str, **kwargs) -> str:
        """
        Sends a query to the LLM and returns the response.

        Args:
            prompt: The prompt to send to the LLM.
            **kwargs:  Additional keyword arguments (e.g., for temperature, max_tokens).

        Returns:
            The LLM's response as a string.
        """
        pass

    @abstractmethod
    def identify_intent_and_entities(self, query: str) -> Tuple[str, Dict[str, Any], float]:
        """
        Identifies the intent and entities in a user query.

        Args:
            query: The user's query string.

        Returns:
            A tuple containing:
            - The identified intent (e.g., "analyze_company", "get_stock_price").
            - A dictionary of extracted entities (e.g., {"company": "Apple", "metric": "revenue"}).
            - A confidence score (0.0 to 1.0).
        """
        pass

class OpenAIPlugin(BaseLLMPlugin):
    """LLM plugin for OpenAI's API."""

    def __init__(self, model: str = "gpt-3.5-turbo", api_base: Optional[str] = None):
        self.model = model
        self.api_base = api_base
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMConfigurationError("OPENAI_API_KEY not found in environment variables.")

        # We import openai lazily, so the import error only occurs if this plugin is used.
        try:
            import openai
            self.openai = openai
            self.openai.api_key = self.api_key
            if api_base:
              self.openai.api_base = api_base
        except ImportError:
            raise LLMConfigurationError("OpenAI library not installed. Run 'pip install openai'.")
        except Exception as e:
            raise LLMConfigurationError(f"Error initializing OpenAI:{e}")

    def query(self, prompt: str, **kwargs) -> str:
        """Sends a query to the OpenAI API."""
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."}, #Optional
                    {"role": "user", "content": prompt}
                ],
                **kwargs  # Pass any additional parameters (temperature, max_tokens, etc.)
            )
            # Access the response correctly.  The structure changed.
            return response.choices[0].message.content

        except self.openai.OpenAIError as e:  # Catch OpenAI-specific errors
            logger.exception(f"OpenAI API error: {e}")
            raise LLMAPIError(f"OpenAI API error: {e}") from e
        except Exception as e:  # Catch more generic errors
            logger.exception(f"An unexpected error occurred: {e}")
            raise LLMAPIError(f"An unexpected error occurred: {e}") from e

    def identify_intent_and_entities(self, query: str) -> Tuple[str, Dict[str, Any], float]:
        """Identifies intent and entities using OpenAI."""
        prompt = f"""
        Identify the intent and entities in the following user query.
        Return the results in JSON format.  Include a confidence score (0.0 to 1.0).

        Query: {query}

        Example Response:
        {{
          "intent": "analyze_company",
          "entities": {{"company": "Apple", "metric": "revenue"}},
          "confidence": 0.95
        }}
        """
        try:
            response_text = self.query(prompt)
            # Attempt to load the response as JSON.
            try:
                response_json = json.loads(response_text)
            except json.JSONDecodeError:
                raise LLMAPIError(f"Invalid JSON response from OpenAI: {response_text}")

            intent = response_json.get("intent", "unknown")
            entities = response_json.get("entities", {})
            confidence = float(response_json.get("confidence", 0.0))  # Ensure it's a float.

            return intent, entities, confidence

        except Exception as e: # Catch general exception
             logger.exception(f"Error identifying intent and entities: {e}")
             return "unknown", {}, 0.0

class CoherePlugin(BaseLLMPlugin):
    """LLM plugin for Cohere's API (Placeholder - adapt from OpenAIPlugin)."""
    def __init__(self, model: str = "command-xlarge-nightly"):
        self.model = model
        self.api_key = os.getenv("COHERE_API_KEY")

        if not self.api_key:
            raise LLMConfigurationError("COHERE_API_KEY not found in environment variables")

        try:
            import cohere
            self.cohere = cohere.Client(self.api_key)
        except ImportError:
            raise LLMConfigurationError("Cohere library not installed.  Run 'pip install cohere'.")
        except Exception as e:
            raise LLMConfigurationError("Error initializing Cohere Client")

    def query(self, prompt: str, **kwargs) -> str:
        try:
            response = self.cohere.chat(
                message=prompt,
                model = self.model,
                **kwargs
            )
            return response.text
        except Exception as e:
            logger.exception("Cohere API Error")
            raise LLMAPIError(f"Cohere API Error:{e}")

    def identify_intent_and_entities(self, query: str) -> Tuple[str, Dict[str, Any], float]:
        # TODO: IMPLEMENT. Use similar pattern as OpenAIPlugin
        #For now return placeholders:
        logger.warning("Cohere identify_intent_and_entities is a placeholder.")
        return "unknown", {}, 0.0

class LLMPluginFactory:
    """Factory class for creating LLM plugin instances."""

    @staticmethod
    def create_plugin(config_path: str = "config/llm_plugins.yaml") -> BaseLLMPlugin:
        """
        Creates an LLM plugin instance based on the configuration.

        Args:
            config_path: Path to the LLM plugin configuration file.

        Returns:
            An instance of the appropriate BaseLLMPlugin subclass.

        Raises:
            LLMConfigurationError: If the configuration is invalid or the plugin cannot be created.
        """
        config = load_config(config_path)

        if not config:
            raise LLMConfigurationError(f"Could not load LLM plugin configuration from {config_path}")

        active_plugin_name = config.get("active_plugin")
        if not active_plugin_name:
            raise LLMConfigurationError("No active_plugin specified in LLM plugin configuration.")

        plugins_config = config.get("llm_plugins", {})
        if active_plugin_name not in plugins_config:
            raise LLMConfigurationError(f"Configuration for plugin '{active_plugin_name}' not found.")

        plugin_config = plugins_config[active_plugin_name]
        plugin_class_name = plugin_config.get("class")
        if not plugin_class_name:
            raise LLMConfigurationError(f"No 'class' specified for plugin '{active_plugin_name}'.")

        # Dynamically import and instantiate the plugin class.
        try:
            module_name, class_name = plugin_class_name.rsplit('.', 1)
            module = __import__(module_name, fromlist=[class_name])
            plugin_class = getattr(module, class_name)
            plugin_instance = plugin_class(**plugin_config)  # Pass config as keyword arguments
            return plugin_instance
        except (ImportError, AttributeError, TypeError) as e:
            raise LLMConfigurationError(f"Could not create plugin '{active_plugin_name}': {e}") from e
        except Exception as e: #Catch exception
            logger.exception("Could not create plugin")
            raise LLMConfigurationError(f"Could not create plugin: {e}")

# --- Example Usage (and for testing) ---
if __name__ == '__main__':
    try:
        # Create dummy llm_plugin.yaml
        llm_config_data = {
            'llm_plugins': {
                'openai': {
                    'class': 'core.llm_plugin.OpenAIPlugin',
                    'model': 'gpt-3.5-turbo'
                },
            },
            'active_plugin': 'openai'
        }
        with open('config/llm_plugins.yaml', 'w') as f:
            yaml.dump(llm_config_data, f)
        # ---
        plugin = LLMPluginFactory.create_plugin()

        # Test a query.
        prompt = "What is the capital of France?"
        response = plugin.query(prompt)
        print(f"Response to '{prompt}': {response}")

        # Test intent and entity identification.
        query = "Analyze the financial performance of AAPL."
        intent, entities, confidence = plugin.identify_intent_and_entities(query)
        print(f"Query: {query}")
        print(f"Intent: {intent}")
        print(f"Entities: {entities}")
        print(f"Confidence: {confidence}")

    except LLMPluginError as e:
        print(f"Error: {e}")
    finally:
        # Remove the dummy config file
        if os.path.exists('config/llm_plugins.yaml'):
            os.remove('config/llm_plugins.yaml')
            
            import openai

class LLMPlugin:
    def __init__(self, config):
        """
        Initializes the LLM plugin with the provided configuration.

        :param config: Configuration dictionary containing necessary settings like API keys.
        """
        self.api_key = config.get("api_key", "")
        self.model = config.get("model", "gpt-4")
        
        if not self.api_key:
            raise ValueError("API Key is required for the LLM plugin.")
        
        openai.api_key = self.api_key  # Set the OpenAI API key for API calls

    def generate_content(self, prompt, task_description=None):
        """
        Generates content using the LLM based on the provided prompt.
        
        :param prompt: The main prompt to send to the LLM.
        :param task_description: (Optional) A description of the task for additional context.
        
        :return: Generated content from the LLM.
        """
        try:
            # Adding task description to the prompt if provided
            if task_description:
                prompt = f"{task_description}: {prompt}"

            # Make an API call to OpenAI's GPT model (or any other LLM you're using)
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                max_tokens=200,  # Set the maximum number of tokens
                temperature=0.7,  # Control randomness in the output
                n=1,  # Number of completions to generate
                stop=None,  # No specific stopping condition
            )

            # Extract the text response
            generated_text = response.choices[0].text.strip()
            return generated_text

        except Exception as e:
            print(f"Error generating content with LLM: {e}")
            return None

    def summarize_text(self, text):
        """
        Summarizes the provided text using the LLM.

        :param text: The text to summarize.
        :return: Summary of the provided text.
        """
        prompt = f"Summarize the following text: {text}"
        return self.generate_content(prompt)

    def answer_question(self, question, context):
        """
        Answers a specific question based on a given context using the LLM.

        :param question: The question to answer.
        :param context: The context to use for answering the question.
        :return: The answer generated by the LLM.
        """
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return self.generate_content(prompt)

# core/llm_plugin.py
from abc import ABC, abstractmethod
import openai  # Example: OpenAI
import anthropic  # Example: Anthropic
# ... import other LLM libraries as needed ...
import logging
from core.utils.config_utils import load_config
from core.utils.token_utils import count_tokens as count_tokens_generic

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class LLMBase(ABC):
    """Abstract base class for LLM integrations."""

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generates text from a prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Model-specific parameters (e.g., temperature, max_tokens).

        Returns:
            The generated text.
        """
        pass

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Gets the token count for a given text."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Returns the name of the LLM."""
        pass

    @abstractmethod
    def get_context_length(self) -> int:
      """Returns the LLM context length"""
      pass

class OpenAILLM(LLMBase):
    """Implementation for OpenAI's LLMs."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        openai.api_key = self.api_key

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = openai.ChatCompletion.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message['content'].strip()
        except Exception as e:
            logging.error(f"OpenAI API error: {e}")
            return "" # Or raise the exception, or return a specific error message

    def get_token_count(self, text: str) -> int:
      # return count_tokens_generic(text, encoding_name='cl100k_base')
      # The above generic one is not model specific, this one below is.
      try:
        encoding = tiktoken.encoding_for_model(self.model_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens
      except KeyError:
        logging.warning(f"Model {self.model_name} not found, using 'cl100k_base' for token count.")
        return count_tokens_generic(text) # Fallback
      except Exception as e:
            logging.error(f"Error counting tokens: {e}")
            return 0  # Return 0 on error

    def get_model_name(self) -> str:
        return self.model_name

    def get_context_length(self) -> int:
        # Lookup table for context lengths (update as needed)
        context_lengths = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
            # Add other models as needed
        }
        return context_lengths.get(self.model_name, 4096) # Default to 4096


class AnthropicLLM(LLMBase):
    """Implementation for Anthropic's LLMs."""

    def __init__(self, api_key: str, model_name: str = "claude-2"):
        self.api_key = api_key
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=f"{anthropic.HUMAN_PROMPT}{prompt}{anthropic.AI_PROMPT}",
                **kwargs
            )
            return response.completion.strip()
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            return ""

    def get_token_count(self, text: str) -> int:
        # return count_tokens_generic(text)  # Anthropic uses a different tokenizer
        # Anthropic provides a way to count tokens:
        return self.client.count_tokens(text)


    def get_model_name(self) -> str:
        return self.model_name

    def get_context_length(self) -> int:
        # Lookup table for context lengths (update as needed)
        context_lengths = {
          "claude-2": 100000,
          "claude-instant-1": 100000
        }
        return context_lengths.get(self.model_name, 100000)  #Sensible default

# Add other LLM implementations (GoogleVertexAILLM, etc.) as needed.

class LLMPlugin:
    """
    Manages interactions with LLMs.  This is the main class that the rest of
    the system will use.
    """

    def __init__(self, config_path: str = "config/llm_plugin.yaml"):
        self.config = load_config(config_path)
        self.llm = self._initialize_llm()

    def _initialize_llm(self) -> LLMBase:
        """Initializes the LLM based on the configuration."""
        provider = self.config.get("provider", "openai").lower()  # Default to OpenAI

        if provider == "openai":
            api_key = self.config.get("openai_api_key")
            model_name = self.config.get("openai_model_name", "gpt-3.5-turbo")
            if not api_key:
                raise ValueError("OpenAI API key not found in llm_plugin.yaml")
            return OpenAILLM(api_key, model_name)
        elif provider == "anthropic":
            api_key = self.config.get("anthropic_api_key")
            model_name = self.config.get("anthropic_model_name", "claude-2")
            if not api_key:
                raise ValueError("Anthropic API key not found in llm_plugin.yaml")
            return AnthropicLLM(api_key, model_name)
        # Add other providers (Google, Cohere, etc.) here
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generates text using the configured LLM."""
        if self.llm:
            return self.llm.generate_text(prompt, **kwargs)
        else:
            logging.error("LLM not initialized.")
            return ""

    def get_token_count(self, text:str) -> int:
        """Return token count"""
        if self.llm:
            return self.llm.get_token_count(text)
        else:
            logging.error("LLM not initialized.")
            return 0

    def get_context_length(self) -> int:
      """Return the context length"""
      if self.llm:
        return self.llm.get_context_length()
      else:
        logging.error("LLM not initialized")
        return 0 # A default

# Example usage (for testing)
if __name__ == "__main__":

    # Create a dummy llm_plugin.yaml for testing purposes
    dummy_config = {
        "provider": "openai",
        "openai_api_key": "YOUR_OPENAI_API_KEY",  # Replace with a test key or mock
        "openai_model_name": "gpt-3.5-turbo"
    }
    with open("config/llm_plugin.yaml", "w") as f:
        yaml.dump(dummy_config, f)

    try:
        plugin = LLMPlugin()
        prompt = "What is the capital of France?"
        generated_text = plugin.generate_text(prompt, max_tokens=50)
        print(f"Generated text: {generated_text}")
        token_count = plugin.get_token_count(prompt)
        print(f"Token count: {token_count}")
        print(f"Context Length: {plugin.get_context_length()}")
    finally:
        os.remove("config/llm_plugin.yaml") # Clean up

# config/llm_plugin.yaml
provider: openai  # Or "anthropic", "google", etc.
openai_api_key: "YOUR_OPENAI_API_KEY"  # Replace with your actual API key
openai_model_name: "gpt-3.5-turbo"  # Or "gpt-4", etc.

# anthropic_api_key: "YOUR_ANTHROPIC_API_KEY"  # Uncomment and fill in when using Anthropic
# anthropic_model_name: "claude-2"

import os
api_key = os.getenv("OPENAI_API_KEY", "fallback_value_if_needed")

import tiktoken
def count_tokens(text, model_name="gpt-4"):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))
    except KeyError:
        return len(tiktoken.get_encoding("cl100k_base").encode(text))


def _initialize_llm(self) -> LLMBase:
    provider_map = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        # Add future providers here
    }
    provider = self.config.get("provider", "openai").lower()
    if provider in provider_map:
        return provider_map[provider](
            self.config.get(f"{provider}_api_key"),
            self.config.get(f"{provider}_model_name")
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_content(self, prompt: str, task: str = "default", **kwargs) -> str:
    task_prompts = {
        "summarize": f"Summarize the following text: {prompt}",
        "qa": f"Context: {kwargs.get('context', '')}\nQuestion: {prompt}\nAnswer:"
    }
    prompt = task_prompts.get(task, prompt)
    return self.generate_text(prompt, **kwargs)

from unittest.mock import patch

@patch("openai.ChatCompletion.create")
def test_generate_text(mock_openai):
    mock_openai.return_value = {"choices": [{"message": {"content": "Paris"}}]}
    plugin = LLMPlugin()
    assert plugin.generate_text("What is the capital of France?") == "Paris"
