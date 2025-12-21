#documentation_agent.py
"""
This module defines the DocumentationAgent, a specialized agent responsible for
writing and updating documentation related to code changes.
"""

from typing import Any, Dict

from core.agents.agent_base import AgentBase


class DocumentationAgent(AgentBase):
    """
    The DocumentationAgent writes and updates documentation based on the
    code changes made by the CoderAgent.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.name = "DocumentationAgent"

    async def execute(self, code_artifact: Dict[str, str]) -> Dict[str, str]:
        """
        Takes a code artifact and generates documentation for it.

        :param code_artifact: A dictionary containing the 'file_path' and 'code'.
        :return: A dictionary containing the documentation and a suggested path.
        """
        source_code = code_artifact.get("code")
        file_path = code_artifact.get("file_path")
        
        # 1. Determine the path for the documentation file (simplified assumption)
        doc_path = file_path.replace("core/", "docs/").replace(".py", ".md")

        # 2. Construct a prompt for the LLM to generate documentation
        prompt = f"""
        You are a technical writer. Your task is to write clear and concise documentation
        for the following Python code. The documentation should explain the purpose of the
        code, its main functions, and how to use it.

        **Source Code from {file_path}:**
        ```python
        {source_code}
        ```

        Please provide the documentation in Markdown format.
        """

        # 3. Call the LLM to generate the documentation
        # generated_docs = await self.run_semantic_kernel_skill("doc_generation", "generate_markdown_docs", {"prompt": prompt})
        generated_docs = f"""
# Documentation for {file_path}

This document describes the functionality of the code in `{file_path}`.

## Overview

The module contains functions related to ...

## Functions

### `new_function()`

This function is a placeholder and should be documented properly.
"""
        return {
            "doc_path": doc_path,
            "documentation": generated_docs
        }

    def get_skill_schema(self) -> Dict[str, Any]:
        """
        Defines the skills of the DocumentationAgent.
        """
        schema = super().get_skill_schema()
        schema["skills"].append(
            {
                "name": "update_documentation",
                "description": "Updates documentation based on a code artifact.",
                "parameters": [
                    {"name": "code_artifact", "type": "dict", "description": "A dictionary containing the code to document."}
                ]
            }
        )
        return schema
