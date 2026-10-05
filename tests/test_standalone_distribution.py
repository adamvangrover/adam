import os
import re
import json
import pytest

def test_standalone_distribution_html_exists():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    html_path = os.path.join(root_dir, "adam_standalone_distribution.html")
    assert os.path.exists(html_path), "adam_standalone_distribution.html must exist at repository root"

    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "<!DOCTYPE html>" in content or "<html" in content
    assert "ADAM Repository — Standalone Interactive Distribution" in content
    assert "Loss Landscape Arena" in content
    assert "Live Neural Net Trainer" in content

def test_standalone_distribution_in_omni_index():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    index_html_path = os.path.join(root_dir, "index.html")
    assert os.path.exists(index_html_path)

    with open(index_html_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"const omniIndex = (\[.*?\]);", re.DOTALL)
    match = pattern.search(content)
    assert match is not None, "omniIndex must be defined in index.html"

    items = json.loads(match.group(1))
    urls = [item.get("u") for item in items]
    assert "adam_standalone_distribution.html" in urls
