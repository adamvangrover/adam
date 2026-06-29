import sys
import os

# Add root to python path to resolve modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from core.schemas.qpf_schema import QPFInput, QPFOutput
from core.prompting.registry import PromptRegistry
from core.prompting.plugins.qpf_plugin import QPFPlugin

def main():
    print("Testing QPFPlugin and Registry")
    plugin_cls = PromptRegistry.get("QPFPlugin")
    assert plugin_cls == QPFPlugin, "Plugin not found in registry!"

    print("Success: QPFPlugin found in registry!")

    # Test initialization (basic check to see if we can render)
    from core.prompting.base_prompt_plugin import PromptMetadata
    meta = PromptMetadata(prompt_id="test_qpf", author="system")
    with open("prompt_library/AOPL-v1.0/quantitative_analysis/QUANT_PROMPT_FRAMEWORK.md", "r") as f:
        template_body = f.read()

    plugin = plugin_cls(metadata=meta, user_template=template_body)

    inputs = {
        "objective": "Develop a mean-reversion strategy",
        "universe": "S&P 500 tech stocks",
        "data_frequency": "1-hour bars",
        "methodology": "Cointegration and Bollinger Bands",
        "deliverable": "Python code using VectorBT",
        "risk_metrics": "Sharpe Ratio and Max Drawdown",
        "constraints": "0.1% slippage and $10k starting capital",
    }
    msgs = plugin.render_messages(inputs)
    print("Rendered Output:")
    print(msgs[0]["content"])
    print("\nTest passed!")

if __name__ == "__main__":
    main()
