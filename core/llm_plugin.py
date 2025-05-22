# core/llm_plugin.py
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv
import yaml
from pathlib import Path
import json
import time
import hashlib
try:
    import tiktoken
except ImportError:
    tiktoken = None  # Set to None if not installed


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

class BaseLLM(ABC):
    """Abstract base class for LLM integrations."""

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generates text from a prompt."""
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
        """Returns the LLM's context length (token limit)."""
        pass


class OpenAILLM(BaseLLM):
    """Implementation for OpenAI's LLMs."""

    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model_name = model_name
        self._openai = None  # Lazy initialization

    @property
    def openai(self):
        """Lazy initialization of the OpenAI client."""
        if self._openai is None:
            try:
                import openai
                openai.api_key = self.api_key
                self._openai = openai
            except ImportError:
                raise LLMConfigurationError("OpenAI library not installed. Run 'pip install openai'.")
            except Exception as e:
                raise LLMConfigurationError(f"Error initializing OpenAI: {e}")
        return self._openai

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = self.openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.exception(f"OpenAI API error: {e}")
            raise LLMAPIError(f"OpenAI API Error: {e}") from e

    def get_token_count(self, text: str) -> int:
        if tiktoken is None:
            logger.warning("tiktoken not installed. Using whitespace splitting for token counting.  Install with 'pip install tiktoken'.")
            return len(text.split())
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            return len(encoding.encode(text))
        except KeyError:
            logger.warning(f"Model {self.model_name} not found, using 'cl100k_base' for token count.")
            return self.get_token_count_generic(text)  # Fallback
        except Exception as e:
            logger.exception(f"Error counting tokens: {e}")
            raise

    def get_token_count_generic(self, text: str) -> int:
        """Fallback token counting if model-specific encoding is unavailable."""
        if tiktoken is None:
            return len(text.split()) #basic fallback.
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            logger.exception("Error: Could not perform fallback token count")
            raise

    def get_model_name(self) -> str:
        return self.model_name

    def get_context_length(self) -> int:
        context_lengths = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
        }
        return context_lengths.get(self.model_name, 4096)  # Default to 4096

class HuggingFaceLLM(BaseLLM):
    """Integration for Hugging Face models, supports local and API-based inference."""

    def __init__(self, model_name: str = "google/flan-t5-base", use_pipeline: bool = True, api_key:str = None):
        self.model_name = model_name
        self.use_pipeline = use_pipeline 
        self._tokenizer = None
        self._model = None
        self._pipeline = None
        self.api_key = api_key
        if self.api_key:
            os.environ["HF_API_TOKEN"] = self.api_key
    @property
    def tokenizer(self):
        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            except ImportError:
                raise LLMConfigurationError("Transformers library not installed. Run 'pip install transformers[torch]' or 'pip install transformers[tf]'.")
            except Exception as e:
                raise LLMConfigurationError(f"Could not load tokenizer for {self.model_name}: {e}")
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
                try:
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                except:
                    self._model = AutoModelForCausalLM.from_pretrained(self.model_name)

            except ImportError:
                raise LLMConfigurationError("Transformers library not installed. Run 'pip install transformers[torch]' or 'pip install transformers[tf]'.")
            except Exception as e:
                raise LLMConfigurationError(f"Could not load model for {self.model_name}: {e}")
        return self._model
    @property
    def pipeline(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline("text-generation", model=self.model_name, tokenizer=self.model_name)
            except Exception as e:
                raise LLMConfigurationError(f"Could not load pipeline {e}")
        return self._pipeline
    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            if self.use_pipeline:
                result = self.pipeline(prompt, **kwargs)
                return result[0]["generated_text"]
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt") 
                outputs = self.model.generate(**inputs, **kwargs)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        except Exception as e:
            logger.exception(f"Hugging Face generation error: {e}")
            raise LLMAPIError(f"Hugging Face Generation Error: {e}") from e


    def get_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_model_name(self) -> str:
        return self.model_name

    def get_context_length(self) -> int:
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        context_lengths = {
            "google/flan-t5-base": 512,
            "bigscience/bloom": 1024,
        }
        return context_lengths.get(self.model_name, 512)


class CohereLLM(BaseLLM):
    """Integration for Cohere's LLMs."""

    def __init__(self, api_key: str, model_name: str = "command"):
        self.api_key = api_key
        self.model_name = model_name
        self._client = None  # Lazy initialization

    @property
    def client(self):
        """Lazy initialization of the Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(self.api_key)
            except ImportError:
                raise LLMConfigurationError("Cohere library not installed. Run 'pip install cohere'.")
            except Exception as e:
                raise LLMConfigurationError(f"Error initializing Cohere client: {e}")
        return self._client

    def generate_text(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.generate(model=self.model_name, prompt=prompt, **kwargs)
            return response.generations[0].text.strip()
        except Exception as e:
            logger.exception(f"Cohere API error: {e}")
            raise LLMAPIError(f"Cohere API Error: {e}") from e

    def get_token_count(self, text: str) -> int:
        try:
            return len(self.client.tokenize(text=text).tokens)
        except Exception as e:
            logger.exception(f"Cohere token counting error: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model_name

    def get_context_length(self) -> int:
        context_lengths = {
            "command": 4096,
            "command-light":4096,
            "command-xlarge-nightly": 4096
        }
        return context_lengths.get(self.model_name, 4096)

class PromptTemplate:
    """Handles dynamic prompt generation."""
    templates = {
        "qa": "Context: {context}\nQuestion: {question}\nAnswer:",
        "summarization": "Summarize the following:\n{input}",
        "default": "{input}" 
    }

    @classmethod
    def format(cls, template_name: str, **kwargs) -> str:
        try:
            template = cls.templates.get(template_name, cls.templates["default"])
            return template.format(**kwargs)
        except KeyError as e:
             logger.exception(f"Missing key in kwargs for prompt formatting: {e}")
             raise
        except Exception as e:
            logger.exception(f"Error with prompt {e}")
            raise

class CacheManager:
    """Caches API responses using a file-based cache."""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


    def get_cache_key(self, prompt: str, model_name: str) -> str:
        """
        Generates a unique cache key based on the prompt and model name.
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
        return f"{model_name}_{prompt_hash}.json"

    def get(self, prompt: str, model_name: str) -> Optional[str]:
        key = self.get_cache_key(prompt, model_name)
        cache_file = self.cache_dir / key
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    if time.time() - data["timestamp"] < data.get("ttl", 86400): 
                        logger.info(f"Cache hit: {key}")
                        return data["response"]
                    else:
                        logger.info(f"Cache expired: {key}")
                        cache_file.unlink() 
            except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Error reading from cache: {e}. Deleting {cache_file}")
                cache_file.unlink(missing_ok=True) 
        return None

    def set(self, prompt: str, model_name: str, response: str, ttl: int = 86400) -> None:
        key = self.get_cache_key(prompt, model_name)
        cache_file = self.cache_dir / key
        try:
            data = {
                "prompt": prompt,
                "response": response,
                "timestamp": time.time(),
                "ttl": ttl 
            }
            with open(cache_file, "w") as f:
                json.dump(data, f)
            logger.info(f"Cached response: {key}")
        except Exception as e:
            logger.exception(f"Could not write to cache {e}")
            raise

class LLMPlugin:
    """Manages LLM interactions with caching and configuration."""

    def __init__(self, config_path: str = "config/llm_plugin.yaml", use_cache: bool = True):
        # Assuming core.utils.config_utils.load_config exists and works
        # from core.utils.config_utils import load_config 
        # self.config = load_config(config_path) # This would be ideal
        # For now, using simpler internal load to avoid circular dependency if config_utils imports this
        self.config = self._load_internal_config(config_path)
        self.cache = CacheManager() if use_cache else None
        self.llm = self._initialize_llm()

    def _load_internal_config(self, config_path: str) -> Dict[str, Any]:
        """Loads LLM plugin configuration from a YAML file (internal)."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"LLM configuration file not found: {config_path}")
            # Fallback to a default configuration or raise specific error
            # For now, returning a default that might allow basic (e.g. HuggingFace local) to work
            logger.warning("Returning default LLM config as llm_plugin.yaml was not found.")
            return {"provider": "huggingface", "huggingface_model_name": "google/flan-t5-base", "huggingface_use_pipeline": True}
        except yaml.YAMLError as e:
            logger.exception(f"Error parsing LLM configuration file: {e}")
            raise LLMConfigurationError(f"Error parsing LLM config file: {e}")


    def _initialize_llm(self) -> BaseLLM:
        """Initializes the LLM based on the configuration."""
        provider_map = {
            "openai": OpenAILLM,
            "huggingface": HuggingFaceLLM,
            "cohere": CohereLLM,
        }
        provider = self.config.get("provider", "huggingface").lower() # Default to huggingface if no provider

        if provider not in provider_map:
             raise ValueError(f"Unsupported LLM provider: {provider}")

        api_key_env_var = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_env_var)
        
        # For HuggingFace, API key is optional (for inference API or gated models)
        # For others, it's typically required.
        if not api_key and provider not in ["huggingface"]: # allow HF to proceed without API key for local models
            raise LLMConfigurationError(f"API key for {provider} ({api_key_env_var}) not found in environment variables.")

        # Get model name from config, or use default from the LLM class if not specified
        default_model_name = provider_map[provider](api_key="dummy_key_if_needed_for_default_name_only").get_model_name() if provider != "huggingface" else "google/flan-t5-base" # HF needs specific default
        model_name = self.config.get(f"{provider}_model_name", default_model_name)


        if provider == "huggingface":
            use_pipeline = self.config.get("huggingface_use_pipeline", True)
            # Pass api_key which might be None (handled by HuggingFaceLLM)
            return HuggingFaceLLM(model_name=model_name, use_pipeline=use_pipeline, api_key=api_key) 
        else:
            if not api_key: # Should have been caught earlier, but as a safeguard
                 raise LLMConfigurationError(f"API key for {provider} is required but not found.")
            return provider_map[provider](api_key=api_key, model_name=model_name)

    def generate_text(self, prompt: str, task: str = "default", **kwargs) -> str:
        """Generates text using the configured LLM with caching and task-specific prompting."""
        formatted_prompt = PromptTemplate.format(task, input=prompt, **kwargs)

        if self.cache:
            cached_response = self.cache.get(formatted_prompt, self.llm.get_model_name())
            if cached_response:
                return cached_response

        response = self.llm.generate_text(formatted_prompt, **kwargs)
        
        if self.cache: # Corrected indentation for this block
            self.cache.set(formatted_prompt, self.llm.get_model_name(), response)

        return response # Corrected indentation for return

    def get_token_count(self, text: str) -> int:
        """Returns the token count for a given text."""
        return self.llm.get_token_count(text)

    def get_model_name(self) -> str:
        """Returns the current LLM model name."""
        return self.llm.get_model_name()

    def identify_intent_and_entities(self, query: str) -> Tuple[str, Dict[str, Any], float]:
        """Identifies the intent and entities in a user query using the configured LLM."""
        if self.llm:
            # Check if the specific LLM instance implements this method
            if hasattr(self.llm, 'identify_intent_and_entities') and callable(getattr(self.llm, 'identify_intent_and_entities')):
                return self.llm.identify_intent_and_entities(query)
            else:
                # Fallback or generic implementation if the specific LLM class doesn't have it
                logger.warning(f"The current LLM '{self.get_model_name()}' does not have a specific 'identify_intent_and_entities' method. Using fallback.")
                # Basic fallback: try to format a prompt and parse. This is highly dependent on LLM capability.
                prompt = f"Identify the intent and entities in the following query: \"{query}\". Respond in JSON format with keys 'intent', 'entities' (dictionary), and 'confidence' (float)."
                response_text = self.generate_text(prompt, task="default") # Using default task for direct prompting
                try:
                    response_json = json.loads(response_text)
                    intent = response_json.get("intent", "unknown")
                    entities = response_json.get("entities", {})
                    confidence = float(response_json.get("confidence", 0.0))
                    return intent, entities, confidence
                except json.JSONDecodeError:
                    logger.error(f"Fallback intent/entity identification failed: Invalid JSON response: {response_text}")
                    return "unknown", {}, 0.0
                except Exception as e:
                    logger.error(f"Fallback intent/entity identification failed: {e}")
                    return "unknown", {}, 0.0
        else:
            logger.error("LLM not initialized.")
            return "unknown", {}, 0.0

    def get_context_length(self) -> int:
        """Returns the context length of the current LLM model."""
        return self.llm.get_context_length()

# Example usage (for testing)
if __name__ == "__main__":
    try:
        dummy_config_content = {
            "provider": "huggingface", 
            "huggingface_model_name": "google/flan-t5-base",
            "huggingface_use_pipeline": True
        }
        # Ensure config directory exists
        if not os.path.exists("config"):
            os.makedirs("config")
        with open("config/llm_plugin.yaml", "w") as f:
            yaml.dump(dummy_config_content, f)
        
        # Note: For OpenAI/Cohere, set relevant API keys in environment for this test
        # e.g. os.environ["OPENAI_API_KEY"] = "your_key_here" 
        
        plugin = LLMPlugin(config_path="config/llm_plugin.yaml")

        prompt = "What is the capital of France?"
        generated_text = plugin.generate_text(prompt, max_length=50) # HuggingFace uses max_length
        print(f"Generated text for '{prompt}': {generated_text}")

        summarization_prompt = "Artificial intelligence is transforming many industries. Large language models are a key component of this transformation, enabling new applications and capabilities."
        summary = plugin.generate_text(summarization_prompt, task="summarization", max_length=50)
        print(f"Summary: {summary}")
        
        token_count = plugin.get_token_count(prompt)
        print(f"Token count for '{prompt}': {token_count}")
        print(f"Context Length for {plugin.get_model_name()}: {plugin.get_context_length()}")

    except LLMPluginError as e:
        print(f"LLM Plugin Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists("config/llm_plugin.yaml"):
            os.remove("config/llm_plugin.yaml")
# End of the coherent, first definition block of LLMPlugin and related classes.
# The rest of the original file contained duplicate/alternative definitions.
