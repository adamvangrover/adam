import pytest
from src.orchestrator.router import CognitiveRouter

def test_cognitive_router():
    router = CognitiveRouter()
    assert router.route("High-context synthesis for SEC Edgar") == "Gemini 1.5 Pro"
    assert router.route("Meticulous formatting required") == "Claude 3.5 Sonnet"
    assert router.route("Rapid data extraction and high-speed RPC tool calling") == "GPT-4o"
