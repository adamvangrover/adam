from __future__ import annotations
from typing import Dict, Any, List
import re
import json
from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata
from core.schemas.skeleton_inject import SkeletonInput, SkeletonOutput, SynthesisInput, SynthesisOutput
from core.schemas.critique import CritiqueInput, CritiqueOutput
from core.prompting.loader import PromptLoader

class SkeletonPlugin(BasePromptPlugin[SkeletonOutput]):
    def __init__(self, **kwargs):
        # Load from file if not provided explicitly
        if not kwargs.get("system_template") and not kwargs.get("user_template"):
            metadata_dict, sys_tmpl, user_tmpl = PromptLoader.load_markdown_with_frontmatter("skeleton_generation")

            # Construct Metadata object from dictionary
            meta = PromptMetadata(
                prompt_id=metadata_dict.get("prompt_id", "AOPL-OS-001"),
                author=metadata_dict.get("author", "system"),
                version=metadata_dict.get("version", "1.0.0"),
                model_config=metadata_dict.get("model_config", {}),
                tags=metadata_dict.get("tags", [])
            )

            super().__init__(
                metadata=meta,
                system_template=sys_tmpl,
                user_template=user_tmpl
            )
        else:
            super().__init__(**kwargs)

    def get_input_schema(self):
        return SkeletonInput

    def get_output_schema(self):
        return SkeletonOutput

    def parse_response(self, raw_response: str) -> SkeletonOutput:
        return SkeletonOutput(skeleton_text=raw_response.strip())


class SynthesisPlugin(BasePromptPlugin[SynthesisOutput]):
    def __init__(self, **kwargs):
        if not kwargs.get("system_template") and not kwargs.get("user_template"):
            metadata_dict, sys_tmpl, user_tmpl = PromptLoader.load_markdown_with_frontmatter("synthesis_audit")

            meta = PromptMetadata(
                prompt_id=metadata_dict.get("prompt_id", "AOPL-OS-002"),
                author=metadata_dict.get("author", "system"),
                version=metadata_dict.get("version", "1.0.0"),
                model_config=metadata_dict.get("model_config", {}),
                tags=metadata_dict.get("tags", [])
            )

            super().__init__(
                metadata=meta,
                system_template=sys_tmpl,
                user_template=user_tmpl
            )
        else:
            super().__init__(**kwargs)

    def get_input_schema(self):
        return SynthesisInput

    def get_output_schema(self):
        return SynthesisOutput

    def parse_response(self, raw_response: str) -> SynthesisOutput:
        return SynthesisOutput(final_text=raw_response.strip())

class CritiquePlugin(BasePromptPlugin[CritiqueOutput]):
    def __init__(self, **kwargs):
        if not kwargs.get("system_template") and not kwargs.get("user_template"):
            metadata_dict, sys_tmpl, user_tmpl = PromptLoader.load_markdown_with_frontmatter("critique")

            meta = PromptMetadata(
                prompt_id=metadata_dict.get("prompt_id", "AOPL-OS-003"),
                author=metadata_dict.get("author", "system"),
                version=metadata_dict.get("version", "1.0.0"),
                model_config=metadata_dict.get("model_config", {}),
                tags=metadata_dict.get("tags", [])
            )

            super().__init__(
                metadata=meta,
                system_template=sys_tmpl,
                user_template=user_tmpl
            )
        else:
            super().__init__(**kwargs)

    def get_input_schema(self):
        return CritiqueInput

    def get_output_schema(self):
        return CritiqueOutput

    def parse_response(self, raw_response: str) -> CritiqueOutput:
        try:
            # Handle potential markdown code blocks
            clean_json = raw_response.strip().replace("```json", "").replace("```", "")
            data = json.loads(clean_json)
            return CritiqueOutput(**data)
        except json.JSONDecodeError:
            # Fallback for non-JSON response
            return CritiqueOutput(
                status="REJECTED",
                score=0,
                feedback="Failed to parse critique JSON.",
                red_flags=["Parse Error"]
            )
