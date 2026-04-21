# Static Mock Mode Fallback Contract

This directory (`config/mocks/`) contains lightweight Python static proxies utilized when the application operates in 'Static Mock Mode'. This ecosystem allows the UI and Swarm logic to be developed, demonstrated, and tested even when heavy execution layers or paid APIs are offline.

## Engaging Mock Mode
A 'Static Mock Mode' fallback can be engaged via the environment variables `MOCK_MODE=true` or `ENV=demo`. When engaged, the system safely routes application logic away from the heavy Rust execution layer and live APIs toward these synthetic responses.

## Mock Construction Rules
To maintain the "Mock Contract", all new mock implementations must adhere to the following rules:

1. **No Generic Text Stubs:** Avoid implementing empty or generic text stubs for mock fallbacks (e.g., returning "Mock Data").
2. **Real Functional Logic:** Ensure all mock components feature real functional logic, data models (e.g., Pydantic), and dynamic calculations to provide robust graceful degradation in static environments. The mock should simulate the behavior of the real system as closely as possible.
3. **MockLLM Standard:** The `MockLLM` class uses `generate_text()` for standard text generation, and `generate_structured()` for returning mock schema instances.

## Proxy Location
Proxies like `mock_edgar.py` and `mock_llm_generator.py` are segregated in this directory to clearly separate static mock data from production logic.
