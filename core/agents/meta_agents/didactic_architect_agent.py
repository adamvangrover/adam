from __future__ import annotations
from typing import Any, Dict, List
import logging
import os
from jinja2 import Template
from core.agents.agent_base import AgentBase
from core.agents.mixins.audit_mixin import AuditMixin
from core.schemas.meta_agent_schemas import (
    DidacticArchitectInput,
    DidacticArchitectOutput,
    TutorialSection,
    PortableConfig,
    AudienceLevel
)

class DidacticArchitectAgent(AgentBase, AuditMixin):
    """
    The Didactic Architect Agent is a meta-agent designed to build modular,
    self-contained, portable, and complementary tutorials and setups.
    It bridges the gap between code and comprehension.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)
        AuditMixin.__init__(self)
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

        # Log completion
        self.log_decision(
            activity_type="TutorialGeneration",
            details={"topic": input_data.topic, "audience": input_data.target_audience.value},
            outcome={"title": title, "sections_count": len(sections)}
        )

        return output

    async def plan_curriculum(self, audit_path: str, architecture_path: str) -> List[str]:
        """
        Generates a learning path based on system failures (Audit) and new features (Architecture).
        """
        lessons = []

        # 1. Analyze Failures
        audit_analysis = await self.analyze_audit_logs(audit_path)
        if "Failures Detected" in audit_analysis:
            lessons.append("Troubleshooting Common Failures")

        # 2. Analyze Architecture
        try:
            with open(architecture_path, 'r') as f:
                arch_content = f.read()
            if "vertical_risk_agent" in arch_content:
                lessons.append("Mastering Vertical Risk Analysis")
            if "infrastructure" in arch_content:
                lessons.append("System Observability & Capacity Planning")
        except FileNotFoundError:
            pass

        self.log_decision(
            activity_type="CurriculumPlanning",
            details={"audit_source": audit_path},
            outcome={"lessons": lessons}
        )
        return lessons

    async def analyze_audit_logs(self, audit_file_path: str = "core/libraries_and_archives/reports/Audit_Defense_File.md") -> str:
        """
        Analyzes the Audit Defense File to identify recurring failures or patterns
        that require new documentation.
        """
        try:
            with open(audit_file_path, 'r') as f:
                content = f.read()

            # Simple heuristic analysis
            failure_count = content.count("Status: FAILURE") + content.count("status': 'FAILURE'")
            warning_count = content.count("WARNING")

            report = f"# Self-Improvement Analysis\n\n"
            report += f"Analyzed {audit_file_path}\n"
            report += f"- Failures Detected: {failure_count}\n"
            report += f"- Warnings Detected: {warning_count}\n\n"

            if failure_count > 0:
                report += "## Recommendation\n"
                report += "Generate a troubleshooting guide for recent failures.\n"

            # Log this meta-analysis
            self.log_decision(
                activity_type="SelfImprovementAnalysis",
                details={"source": audit_file_path},
                outcome={"failures": failure_count, "recommendation": "See report"}
            )

            return report

        except FileNotFoundError:
            logging.warning(f"Audit file not found: {audit_file_path}")
            return "Audit file not found."

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
