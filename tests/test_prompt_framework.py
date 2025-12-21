import json
import os
import sys
import tempfile
from typing import List

import yaml
from pydantic import BaseModel

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.prompting.base_prompt_plugin import BasePromptPlugin, PromptMetadata
from core.prompting.registry import PromptRegistry

# --- Plugin Implementation ---

class AnalysisInput(BaseModel):
    company_name: str
    required_metrics: List[str]
    transcript_segment: str

class AnalysisOutput(BaseModel):
    EBITDA: str
    Revenue: str

class FinancialAnalysisPlugin(BasePromptPlugin[AnalysisOutput]):
    def get_input_schema(self) -> type[BaseModel]:
        return AnalysisInput

    def get_output_schema(self) -> type[AnalysisOutput]:
        return AnalysisOutput

# --- Verification Logic ---

def test_framework():
    print("Testing Prompt-as-Code Framework...")

    # 1. Existing Test: Manual Instantiation and Render
    yaml_config = """
prompt_id: "fin_statement_analyzer"
version: "2.1.0"
author: "system_architect_bot"
model_config:
  temperature: 0.2
  model: "gpt-4-turbo"
  stop_sequences: ["###"]
template_body: |
  SYSTEM: You are a senior financial analyst.

  USER: Analyze the following earnings call transcript for {{ company_name }}.
  Focus strictly on the following metrics:
  {% for metric in required_metrics %}
  - {{ metric }}
  {% endfor %}

  Context:
  {{ transcript_segment }}

  Output strictly in JSON format matching the schema.
"""
    config = yaml.safe_load(yaml_config)

    metadata = PromptMetadata(
        prompt_id=config['prompt_id'],
        version=config['version'],
        author=config['author'],
        model_config=config['model_config']
    )

    plugin = FinancialAnalysisPlugin(metadata, template_string=config['template_body'])
    print("[OK] Plugin Instantiated (Manual)")

    inputs = {
        "company_name": "Acme Corp",
        "required_metrics": ["EBITDA", "Revenue"],
        "transcript_segment": "We saw a 20% increase..."
    }

    rendered_prompt = plugin.render(inputs)
    assert "Acme Corp" in rendered_prompt
    assert "- EBITDA" in rendered_prompt
    print("[OK] Render Verification Passed")

    mock_llm_response = '{"EBITDA": "20M", "Revenue": "100M"}'
    parsed_output = plugin.parse_response(mock_llm_response)
    assert parsed_output.EBITDA == "20M"
    print("[OK] Response Parsing Verification Passed")

    audit_log = plugin.to_audit_log(inputs, mock_llm_response)
    log_data = json.loads(audit_log)
    assert log_data['prompt_id'] == "fin_statement_analyzer"
    print("[OK] Audit Log Verification Passed")

    # --- New Features ---

    # 2. Test Registry
    PromptRegistry.register(FinancialAnalysisPlugin)
    retrieved_cls = PromptRegistry.get("FinancialAnalysisPlugin")
    assert retrieved_cls == FinancialAnalysisPlugin
    print("[OK] Registry Verification Passed")

    # 3. Test render_messages (Chat API)
    metadata_chat = PromptMetadata(
        prompt_id="test_chat",
        author="test",
        model_config={}
    )
    plugin_chat = FinancialAnalysisPlugin(
        metadata_chat,
        system_template="You are a helper.",
        user_template="Analyze {{ company_name }}"
    )

    messages = plugin_chat.render_messages(inputs)
    print(f"\nRendered Messages: {messages}")
    assert len(messages) == 2
    assert messages[0]['role'] == 'system'
    assert "helper" in messages[0]['content']
    assert messages[1]['role'] == 'user'
    assert "Acme Corp" in messages[1]['content']
    print("[OK] render_messages Verification Passed")

    # 4. Test from_yaml
    yaml_content = """
prompt_id: "fin_yaml"
version: "1.0.0"
author: "bot"
model_config:
    temperature: 0.5
template_body: "Analyze {{ company_name }}"
"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        tmp.write(yaml_content)
        tmp_path = tmp.name

    try:
        plugin_yaml = FinancialAnalysisPlugin.from_yaml(tmp_path)
        assert plugin_yaml.metadata.prompt_id == "fin_yaml"
        assert plugin_yaml.metadata.llm_config['temperature'] == 0.5
        print("[OK] from_yaml Verification Passed")
    finally:
        os.remove(tmp_path)

if __name__ == "__main__":
    test_framework()
