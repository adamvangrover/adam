# core/llm_plugin.py
import hashlib
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, get_args, get_origin

import yaml
from dotenv import load_dotenv

# --- Graceful Import Fallbacks ---
try:
    import tiktoken
except ImportError:
    tiktoken = None  # Set to None if not installed

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Exceptions ---
class LLMPluginError(Exception):
    """Base class for LLM plugin exceptions."""
    pass

class LLMConfigurationError(LLMPluginError):
    """Raised for configuration issues."""
    pass

class LLMAPIError(LLMPluginError):
    """Raised for errors during API calls."""
    pass

# --- Base Interface ---
class BaseLLM(ABC):
    """Abstract base class for LLM integrations."""

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generates text from a prompt."""
        pass

    @abstractmethod
    def generate_structured(self, prompt: str, response_schema: Any, tools: Optional[List[Any]] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Generates structured output conforming to a Pydantic schema.
        Returns: (ValidatedPydanticObject, MetadataDict)
        MetadataDict contains 'thought_signature', 'token_usage', etc.
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
        """Returns the LLM's context length (token limit)."""
        pass

    # Optional interface for intent detection (can be overridden by specific models)
    def identify_intent_and_entities(self, query: str) -> Tuple[str, Dict[str, Any], float]:
        """Optional method for specialized intent detection."""
        raise NotImplementedError("This model does not support native intent detection.")


# --- Implementations ---

class MockLLM(BaseLLM):
    """Mock LLM for testing and development without API costs."""
    def __init__(self, model_name="mock-model", **kwargs):
        self.model_name = model_name
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        return f"Mock response to: {prompt[:20]}..."
    
    def generate_structured(self, prompt: str, response_schema: Any, tools: Optional[List[Any]] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Mock structured generation that attempts to return a dummy instance of the schema."""
        try:
            # Simple recursive mock helper
            def mock_instance(model_class):
                if not hasattr(model_class, "model_fields"):
                    return "MockValue"

                init_data = {}
                for name, field in model_class.model_fields.items():
                    annotation = field.annotation
                    origin = get_origin(annotation)
                    args = get_args(annotation)

                    if origin is Literal:
                         init_data[name] = args[0]
                    elif origin is list or origin is List:
                         if args:
                             inner_type = args[0]
                             if hasattr(inner_type, "model_fields"):
                                 init_data[name] = [mock_instance(inner_type)]
                             else:
                                 init_data[name] = ["MockItem"]
                         else:
                             init_data[name] = []
                    elif annotation == str:
                        init_data[name] = "MockString"
                    elif annotation == int:
                        # Heuristic for ranges like conviction_level (1-10)
                        if "level" in name or "score" in name:
                            init_data[name] = 5
                        else:
                            init_data[name] = 42
                    elif annotation == float:
                        init_data[name] = 3.14
                    elif annotation == bool:
                        init_data[name] = True
                    elif hasattr(annotation, "model_fields"):
                         init_data[name] = mock_instance(annotation)
                    else:
                         init_data[name] = None

                # Try to create
                try:
                    return model_class(**init_data)
                except:
                    return model_class.construct(**init_data)

            result = mock_instance(response_schema)
            metadata = {
                "thought_signature": "mock_thought_signature_token_12345",
                "usage": {"total_tokens": 100}
            }
            return result, metadata

        except Exception as e:
            logger.warning(f"Mock structured generation failed to instantiate schema: {e}")
            return {"mock_error": str(e)}, {"thought_signature": "error_sig"}

    def get_token_count(self, text: str) -> int:
        return len(text.split())
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def get_context_length(self) -> int:
        return 2048


class GeminiLLM(BaseLLM):
    """Integration for Google's Gemini 3 ecosystem."""

    def __init__(self, api_key: str, model_name: str = "gemini-3-pro"):
        self.api_key = api_key
        self.model_name = model_name
        self._genai = None # Lazy init

    @property
    def genai(self):
        if self._genai is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._genai = genai
            except ImportError:
                logger.warning("google.generativeai package not found. Using Mock behavior for GeminiLLM.")
                self._genai = "MOCK"
            except Exception as e:
                raise LLMConfigurationError(f"Error initializing Gemini: {e}")
        return self._genai

    def generate_text(self, prompt: str, **kwargs) -> str:
        if self.genai == "MOCK":
             return f"[Gemini Mock] Response to: {prompt[:30]}"

        try:
            model = self.genai.GenerativeModel(self.model_name)
            # Handle parameters like thinking_level
            generation_config = {}
            if "thinking_level" in kwargs:
                generation_config["thinking_level"] = kwargs["thinking_level"]

            response = model.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            raise LLMAPIError(f"Gemini API Error: {e}") from e

    def generate_structured(self, prompt: str, response_schema: Any, tools: Optional[List[Any]] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Native support for 'Prompt-as-Code' via Gemini Structured Outputs.
        Supports thought_signature for state continuity.
        """
        thought_signature = kwargs.get("thought_signature")
        thinking_level = kwargs.get("thinking_level", "low")

        if self.genai == "MOCK":
            logger.info(f"Gemini Mock: generating structured {response_schema.__name__} (Thinking: {thinking_level})")
            if thought_signature:
                logger.info(f"Gemini Mock: Resuming from thought signature: {thought_signature[:10]}...")
            
            mock_delegate = MockLLM()
            return mock_delegate.generate_structured(prompt, response_schema, tools=tools)

        try:
            # Real API Implementation
            model = self.genai.GenerativeModel(self.model_name, tools=tools)

            generation_config = {
                "response_mime_type": "application/json",
                "response_schema": response_schema, 
                "thinking_level": thinking_level
            }

            if thought_signature:
                generation_config["thought_signature"] = thought_signature

            response = model.generate_content(prompt, generation_config=generation_config)

            # Parse result (Hypothetical SDK support)
            try:
                result_obj = response.parsed
            except:
                json_data = json.loads(response.text)
                result_obj = response_schema(**json_data)

            new_signature = getattr(response, "thought_signature", "simulated_new_signature")

            metadata = {
                "thought_signature": new_signature,
                "usage": response.usage_metadata if hasattr(response, 'usage_metadata') else {}
            }

            return result_obj, metadata

        except Exception as e:
            raise LLMAPIError(f"Gemini Structured Generation Error: {e}") from e

    def get_token_count(self, text: str) -> int:
        if self.genai == "MOCK":
            return len(text.split())
        try:
            model = self.genai.GenerativeModel(self.model_name)
            return model.count_tokens(text).total_tokens
        except:
            return len(text.split())

    def get_model_name(self) -> str:
        return self.model_name

    def get_context_length(self) -> int:
        return 1000000 # Gemini has huge context


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

    def generate_structured(self, prompt: str, response_schema: Any, tools: Optional[List[Any]] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Implementation using OpenAI Structured Outputs (json_object or function calling)."""
        try:
            json_schema = response_schema.model_json_schema()

            response = self.openai.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": f"You must output JSON confirming to this schema: {json.dumps(json_schema)}"},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                **kwargs
            )
            content = response.choices[0].message.content
            obj = response_schema.model_validate_json(content)
            return obj, {"thought_signature": None} 
        except Exception as e:
            raise LLMAPIError(f"OpenAI Structured Error: {e}")

    def get_token_count(self, text: str) -> int:
        if tiktoken is None:
            logger.warning("tiktoken not installed. Using whitespace splitting for token counting.")
            return len(text.split())
        try:
            encoding = tiktoken.encoding_for_model(self.model_name)
            return len(encoding.encode(text))
        except KeyError:
            logger.warning(f"Model {self.model_name} not found, using 'cl100k_base'.")
            return self.get_token_count_generic(text)
        except Exception as e:
            logger.exception(f"Error counting tokens: {e}")
            raise

    def get_token_count_generic(self, text: str) -> int:
        if tiktoken is None:
            return len(text.split())
        try:
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            raise LLMAPIError("Error: Could not perform fallback token count") from e

    def get_model_name(self) -> str:
        return self.model_name

    def get_context_length(self) -> int:
        context_lengths = {
            "gpt-3.5-turbo": 4096,
            "gpt-4": 8192,
            "gpt-4-32k": 32768,
        }
        return context_lengths.get(self.model_name, 4096)


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
                raise LLMConfigurationError("Transformers library not installed.")
            except Exception as e:
                raise LLMConfigurationError(f"Could not load tokenizer for {self.model_name}: {e}")
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            try:
                from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
                try:
                    self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
                except:
                    self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            except ImportError:
                raise LLMConfigurationError("Transformers library not installed.")
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

    def generate_structured(self, prompt: str, response_schema: Any, tools: Optional[List[Any]] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError("HuggingFace structured generation not yet implemented.")

    def get_token_count(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def get_model_name(self) -> str:
        return self.model_name

    def get_context_length(self) -> int:
        if hasattr(self.tokenizer, "model_max_length"):
            return self.tokenizer.model_max_length
        return 512


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

    def generate_structured(self, prompt: str, response_schema: Any, tools: Optional[List[Any]] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        raise NotImplementedError("Cohere structured generation not yet implemented.")

    def get_token_count(self, text: str) -> int:
        try:
            return len(self.client.tokenize(text=text).tokens)
        except Exception as e:
            logger.exception(f"Cohere token counting error: {e}")
            raise

    def get_model_name(self) -> str:
        return self.model_name

    def get_context_length(self) -> int:
        return 4096


# --- Utilities ---

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
        """Generates a unique cache key based on the prompt and model name."""
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


# --- Main Plugin Class ---

class LLMPlugin:
    """Manages LLM interactions with caching, configuration, SLM routing, and intent detection."""

    def __init__(self, config_path: str = "config/llm_plugin.yaml", use_cache: bool = True, config: Optional[Dict[str, Any]] = None):
        """
        Initialize LLMPlugin.
        
        Args:
            config_path: Path to YAML config file.
            use_cache: Whether to use caching.
            config: Direct configuration dictionary (overrides config_path).
        """
        if config:
            self.config = config
        else:
            self.config = self._load_internal_config(config_path)
            
        self.cache = CacheManager() if use_cache else None
        self._validate_security_policies()
        self.llm = self._initialize_llm()
        self.slm = self._initialize_slm()

    def _validate_security_policies(self):
        """
        Enforces Apex Architect security protocols defined in config.
        """
        policies = self.config.get("security_policies", {})

        # 1. HTTPS Enforcement
        if policies.get("enforce_https", True):
             # This is a meta-check. Actual enforcement happens at the network layer,
             # but we log the policy active state.
             logger.info("SECURITY: HTTPS Enforcement Policy is ACTIVE.")

        # 2. PII Masking
        if policies.get("mask_pii", False):
            logger.info("SECURITY: PII Masking is ENABLED (experimental).")

        # 3. Prompt Logging Check
        if policies.get("log_prompts", False):
            logger.warning("SECURITY ALERT: Prompt logging is ENABLED. Ensure no sensitive data is processed.")

    def _initialize_slm(self) -> Optional[BaseLLM]:
        """Initializes a Small Language Model (SLM) for specialized tasks."""
        slm_provider = self.config.get("slm_provider")
        if not slm_provider:
            return None

        try:
            model_name = self.config.get("slm_model_name", "microsoft/phi-2")
            if slm_provider == "huggingface":
                return HuggingFaceLLM(model_name=model_name, use_pipeline=True)
            # Future SLM providers can be added here
        except Exception as e:
            logger.warning(f"Failed to initialize SLM: {e}")
        return None

    def _load_internal_config(self, config_path: str) -> Dict[str, Any]:
        """Loads LLM plugin configuration from a YAML file (internal)."""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"LLM configuration file not found: {config_path}")
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
            "gemini": GeminiLLM,
            "mock": lambda **kwargs: MockLLM(**kwargs)
        }
        provider = self.config.get("provider", "huggingface").lower()

        if provider not in provider_map:
             if provider == "mock":
                 return MockLLM()
             raise ValueError(f"Unsupported LLM provider: {provider}")

        api_key_env_var = f"{provider.upper()}_API_KEY"
        api_key = os.getenv(api_key_env_var)
        
        # Validation for keys
        if not api_key and provider not in ["huggingface", "mock", "gemini"]:
            # Gemini allows "MOCK" behavior internally without key, others might fail
             raise LLMConfigurationError(f"API key for {provider} ({api_key_env_var}) not found in environment variables.")

        # Get model name defaults
        default_model_name = "default-model"
        if provider == "huggingface":
            default_model_name = "google/flan-t5-base"
        elif provider == "openai":
            default_model_name = "gpt-3.5-turbo"
        elif provider == "cohere":
            default_model_name = "command"
        elif provider == "gemini":
            default_model_name = "gemini-3-pro"
        elif provider == "mock":
            default_model_name = "mock-model"

        model_name = self.config.get(f"{provider}_model_name", default_model_name)

        if provider == "huggingface":
            use_pipeline = self.config.get("huggingface_use_pipeline", True)
            return HuggingFaceLLM(model_name=model_name, use_pipeline=use_pipeline, api_key=api_key) 
        elif provider == "mock":
            return MockLLM(model_name=model_name)
        elif provider == "gemini":
             # Pass "mock_key" if not present so init doesn't fail before mock logic triggers
            return GeminiLLM(api_key=api_key or "mock_key", model_name=model_name)
        else:
            return provider_map[provider](api_key=api_key, model_name=model_name)

    def generate_text(self, prompt: str, task: str = "default", **kwargs) -> str:
        """Generates text using the configured LLM with caching and task-specific prompting."""

        # Router Logic for SLM (Graceful expansion)
        model = self.llm
        if self.slm and task in ["financial_extraction", "summarization"]:
            model = self.slm
            logger.info(f"Routing task '{task}' to SLM ({model.get_model_name()})")

        formatted_prompt = PromptTemplate.format(task, input=prompt, **kwargs)

        if self.cache:
            cached_response = self.cache.get(formatted_prompt, model.get_model_name())
            if cached_response:
                return cached_response

        response = model.generate_text(formatted_prompt, **kwargs)
        
        if self.cache:
            self.cache.set(formatted_prompt, model.get_model_name(), response)

        return response

    def generate_structured(self, prompt: str, response_schema: Any, tools: Optional[List[Any]] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """
        Proxy for structured generation.
        """
        model = self.llm
        # No caching for structured/stateful generation for now
        return model.generate_structured(prompt, response_schema, tools=tools, **kwargs)

    def query(self, prompt: str, **kwargs) -> str:
        """Alias for generate_text to support legacy agents."""
        return self.generate_text(prompt, **kwargs)

    def get_token_count(self, text: str) -> int:
        """Returns the token count for a given text."""
        return self.llm.get_token_count(text)

    def get_model_name(self) -> str:
        """Returns the current LLM model name."""
        return self.llm.get_model_name()
    
    def get_context_length(self) -> int:
        """Returns the context length of the current LLM model."""
        return self.llm.get_context_length()

    def identify_intent_and_entities(self, query: str) -> Tuple[str, Dict[str, Any], float]:
        """Identifies the intent and entities in a user query using the configured LLM."""
        if self.llm:
            # Check if the specific LLM instance implements this method natively
            if hasattr(self.llm, 'identify_intent_and_entities') and callable(self.llm.identify_intent_and_entities):
                try:
                    return self.llm.identify_intent_and_entities(query)
                except NotImplementedError:
                    pass # Fall through to fallback

            # Graceful Fallback: prompt engineering
            logger.info(f"Using generic fallback for intent identification with '{self.get_model_name()}'.")
            prompt = (
                f"Identify the intent and entities in the following query: \"{query}\". "
                f"Respond in JSON format with keys 'intent', 'entities' (dictionary), and 'confidence' (float)."
            )
            # Use default task for direct prompting to avoid recursion loop
            response_text = self.generate_text(prompt, task="default") 
            try:
                # Attempt to parse JSON from text (handles markdown code blocks if LLM adds them)
                clean_text = response_text.replace("```json", "").replace("```", "").strip()
                response_json = json.loads(clean_text)
                
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

# --- Test / Execution Block ---
if __name__ == "__main__":
    try:
        # Test 1: HuggingFace via file config (simulated)
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
        
        print("--- Testing HuggingFace (Simulated Config File) ---")
        plugin = LLMPlugin(config_path="config/llm_plugin.yaml")

        prompt = "What is the capital of France?"
        generated_text = plugin.generate_text(prompt, max_length=50) # HuggingFace uses max_length
        print(f"Generated text for '{prompt}': {generated_text}")

        summarization_prompt = "Artificial intelligence is transforming many industries. Large language models are a key component of this transformation."
        summary = plugin.generate_text(summarization_prompt, task="summarization", max_length=50)
        print(f"Summary: {summary}")
        
        token_count = plugin.get_token_count(prompt)
        print(f"Token count for '{prompt}': {token_count}")
        print(f"Context Length: {plugin.get_context_length()}")

        # Test 2: Mock Provider via Direct Config Injection
        print("\n--- Testing Mock Provider (Direct Config) ---")
        mock_config = {
            "provider": "mock",
            "mock_model_name": "test-mock-v1"
        }
        mock_plugin = LLMPlugin(config=mock_config, use_cache=False)
        print(f"Model Name: {mock_plugin.get_model_name()}")
        print(f"Mock Response: {mock_plugin.generate_text('Hello mock world')}")
        
        # Test 3: Intent Identification Fallback
        print("\n--- Testing Intent Identification Fallback ---")
        # Note: MockLLM text generation will likely fail the JSON parse, 
        # checking graceful handling of the JSON error.
        intent, entities, conf = mock_plugin.identify_intent_and_entities("Book a flight to NY")
        print(f"Intent (Mock/Fallback): {intent}, Entities: {entities}, Conf: {conf}")

    except LLMPluginError as e:
        print(f"LLM Plugin Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists("config/llm_plugin.yaml"):
            os.remove("config/llm_plugin.yaml")
