from __future__ import annotations
from typing import Any, Dict, List
import logging
import os
from jinja2 import Template
from core.agents.agent_base import AgentBase
from core.schemas.meta_agent_schemas import (
    DidacticArchitectInput,
    DidacticArchitectOutput,
    TutorialSection,
    PortableConfig,
    AudienceLevel
)

class DidacticArchitectAgent(AgentBase):
    """
    The Didactic Architect Agent is a meta-agent designed to build modular,
    self-contained, portable, and complementary tutorials and setups.
    It bridges the gap between code and comprehension.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        self.output_dir = config.get("output_dir", "docs/tutorials")

    async def execute(self, input_data: DidacticArchitectInput) -> DidacticArchitectOutput:
        """
        Generates tutorial content and portable configuration files.
        """
        logging.info(f"DidacticArchitectAgent generating tutorial for: {input_data.topic}")

        # 1. Read Real Context Files
        file_contents = self._read_context_files(input_data.context_files)

        # 2. Generate Content
        title = f"Mastering {input_data.topic}: A {input_data.target_audience.value.title()} Guide"
        sections = self._generate_sections(input_data, file_contents)

        # 3. Generate Portable Setups
        portable_configs = self._generate_portable_configs(input_data)

        output = DidacticArchitectOutput(
            title=title,
            sections=sections,
            setup_guide=self._generate_setup_guide(portable_configs),
            portable_configs=portable_configs
        )

        return output

    def _read_context_files(self, file_paths: List[str]) -> Dict[str, str]:
        """
        Reads the actual content of the provided context files.
        """
        contents = {}
        for path in file_paths:
            try:
                if os.path.exists(path):
                    with open(path, "r", encoding="utf-8") as f:
                        contents[path] = f.read()
                else:
                    logging.warning(f"Context file not found: {path}")
                    contents[path] = "# File not found"
            except Exception as e:
                logging.error(f"Error reading {path}: {e}")
                contents[path] = f"# Error reading file: {e}"
        return contents

    def _generate_sections(self, input_data: DidacticArchitectInput, file_contents: Dict[str, str]) -> List[TutorialSection]:
        """
        Generates the sections of the tutorial, using real file content if available.
        """
        sections = []

        # Introduction
        intro_content = f"Welcome to the {input_data.topic} tutorial. This module is designed for {input_data.target_audience.value} users."
        sections.append(TutorialSection(title="Introduction", content=intro_content, code_snippets=[]))

        # Core Concepts (using file content if relevant)
        concept_content = f"Understanding the fundamental architecture of {input_data.topic}."
        snippets = []

        # If we have file contents, use snippets from them
        if file_contents:
            for path, content in file_contents.items():
                # Take the first 10 lines as a preview/snippet
                preview = "\n".join(content.splitlines()[:15]) + "\n..."
                snippets.append(f"# From {path}\n{preview}")
        else:
             snippets.append("import core\n\nagent = core.Agent()")

        sections.append(TutorialSection(title="Core Concepts", content=concept_content, code_snippets=snippets))

        # Implementation
        impl_template = Template("Now let's look at how {{ topic }} is implemented. \n\nKey files involved: {{ files | join(', ') }}")
        impl_content = impl_template.render(topic=input_data.topic, files=input_data.context_files)

        sections.append(TutorialSection(title="Implementation", content=impl_content, code_snippets=["def run():\n    pass"]))

        return sections

    def _generate_portable_configs(self, input_data: DidacticArchitectInput) -> List[PortableConfig]:
        """
        Generates Dockerfiles or other config files to make the tutorial portable.
        """
        # Fix: Use double quotes for JSON array in Dockerfile CMD
        dockerfile_content = """FROM python:3.9-slim
RUN pip install -r requirements.txt
CMD ["python", "main.py"]"""

        return [
            PortableConfig(
                filename="Dockerfile.tutorial",
                content=dockerfile_content,
                description="A lightweight container for running this tutorial."
            ),
            PortableConfig(
                filename="requirements.txt",
                content="numpy\npandas\n",
                description="Dependencies for the tutorial."
            )
        ]

    def _generate_setup_guide(self, configs: List[PortableConfig]) -> str:
        """
        Generates a quick setup guide string.
        """
        guide = "## Quick Setup\n\n"
        for config in configs:
            guide += f"1. Save `{config.filename}`: {config.description}\n"
        guide += "\nRun `docker build -t tutorial -f Dockerfile.tutorial .` to start."
        return guide
