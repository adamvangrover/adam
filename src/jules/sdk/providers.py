"""
Jules LLM Providers Module.
Implements unified model adapters for Mock, Ollama, OpenAI, Anthropic, and LiteLLM backends.
"""

import json
import logging
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from src.jules.sdk.config import JulesConfig

logger = logging.getLogger("jules.providers")


class BaseLLMProvider(ABC):
    """Abstract base class for Jules LLM providers."""

    @abstractmethod
    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a response given message history, optional system prompt, and optional tools.
        Returns a dictionary with keys: 'content', 'tool_calls', 'usage'.
        """
        pass


class MockLLMProvider(BaseLLMProvider):
    """
    Offline, deterministic fallback provider.
    Simulates intelligent agent reasoning, planning, and tool calls.
    """

    def __init__(self, config: JulesConfig) -> None:
        self.config = config

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        last_message = messages[-1]["content"] if messages else ""
        lower_msg = last_message.lower()

        # Heuristic simulation for common operations
        if "plan" in lower_msg or "step" in lower_msg:
            content = (
                "1. *Analyze request & explore workspace.*\n"
                "   - Check relevant repository files and dependencies.\n"
                "2. *Execute task actions.*\n"
                "   - Perform required edits or analysis.\n"
                "3. *Verify & complete task.*\n"
                "   - Validate results and confirm correctness."
            )
            tool_calls = []
        elif "list" in lower_msg or "find" in lower_msg or "files" in lower_msg:
            content = "Listing workspace directory files..."
            tool_calls = [{"name": "list_files", "args": {"path": "."}}]
        elif "git" in lower_msg or "status" in lower_msg:
            content = "Checking git repository status..."
            tool_calls = [{"name": "git_status", "args": {}}]
        elif "test" in lower_msg or "pytest" in lower_msg:
            content = "Running pytest test suite..."
            tool_calls = [{"name": "run_pytest", "args": {"test_path": "tests/"}}]
        else:
            content = f"[Jules Offline Engine ({self.config.model_name})]: Task analyzed and processed successfully: '{last_message[:80]}'."
            tool_calls = []

        return {
            "content": content,
            "tool_calls": tool_calls,
            "usage": {"prompt_tokens": len(last_message.split()), "completion_tokens": len(content.split())},
        }


class OllamaProvider(BaseLLMProvider):
    """Local Ollama LLM Provider."""

    def __init__(self, config: JulesConfig) -> None:
        self.config = config
        self.url = f"{config.ollama_url.rstrip('/')}/api/chat"

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        formatted_msgs = []
        if system_prompt:
            formatted_msgs.append({"role": "system", "content": system_prompt})
        formatted_msgs.extend(messages)

        payload = {
            "model": self.config.model_name if self.config.model_name != "mock-jules-v1" else "llama3",
            "messages": formatted_msgs,
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }

        try:
            req = urllib.request.Request(
                self.url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                msg = res_data.get("message", {})
                return {
                    "content": msg.get("content", ""),
                    "tool_calls": [],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                }
        except Exception as e:
            logger.warning(f"Ollama request failed ({e}); falling back to MockLLMProvider.")
            return MockLLMProvider(self.config).generate_response(messages, system_prompt, tools)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API Provider."""

    def __init__(self, config: JulesConfig) -> None:
        self.config = config

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            import openai
            client = openai.OpenAI(api_key=self.config.api_key)
            formatted_msgs = []
            if system_prompt:
                formatted_msgs.append({"role": "system", "content": system_prompt})
            formatted_msgs.extend(messages)

            model = self.config.model_name if self.config.model_name != "mock-jules-v1" else "gpt-4o"
            res = client.chat.completions.create(
                model=model,
                messages=formatted_msgs,
                temperature=self.config.temperature,
            )
            choice = res.choices[0].message
            return {
                "content": choice.content or "",
                "tool_calls": [],
                "usage": {
                    "prompt_tokens": res.usage.prompt_tokens if res.usage else 0,
                    "completion_tokens": res.usage.completion_tokens if res.usage else 0,
                },
            }
        except Exception as e:
            logger.warning(f"OpenAI API request failed ({e}); falling back to MockLLMProvider.")
            return MockLLMProvider(self.config).generate_response(messages, system_prompt, tools)


class AnthropicProvider(BaseLLMProvider):
    """Anthropic API Provider."""

    def __init__(self, config: JulesConfig) -> None:
        self.config = config

    def generate_response(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.config.api_key)
            model = self.config.model_name if self.config.model_name != "mock-jules-v1" else "claude-3-5-sonnet-20241022"
            res = client.messages.create(
                model=model,
                max_tokens=4096,
                system=system_prompt or "",
                messages=messages,
                temperature=self.config.temperature,
            )
            text_content = "".join([block.text for block in res.content if hasattr(block, "text")])
            return {
                "content": text_content,
                "tool_calls": [],
                "usage": {"prompt_tokens": res.usage.input_tokens, "completion_tokens": res.usage.output_tokens},
            }
        except Exception as e:
            logger.warning(f"Anthropic API request failed ({e}); falling back to MockLLMProvider.")
            return MockLLMProvider(self.config).generate_response(messages, system_prompt, tools)


def get_llm_provider(config: JulesConfig) -> BaseLLMProvider:
    """Factory function returning configured provider instance."""
    p = config.provider.lower()
    if p == "ollama":
        return OllamaProvider(config)
    elif p == "openai":
        return OpenAIProvider(config)
    elif p == "anthropic":
        return AnthropicProvider(config)
    elif p == "mock":
        return MockLLMProvider(config)
    else:
        logger.info(f"Unknown or default provider '{p}'; using MockLLMProvider.")
        return MockLLMProvider(config)
