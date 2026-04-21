# Mock Ecosystem & Static Fallback

The `config/mocks/` directory contains the lightweight Python static proxies used for safe execution when the live environments or external integrations are unavailable.

## Engaging Static Mock Mode
The fallback is engaged by setting the environment variables `MOCK_MODE=true` or `ENV=demo`. This safely routes application logic away from the heavy Rust execution layer or live API integrations to these synthetic proxies.

## The Mock Contract
When building new mock stubs, developers must adhere to the following rules:
- **No Empty Stubs**: Avoid generic text stubs.
- **Functional Logic**: Ensure all mock components feature real functional logic and perform calculations to provide robust graceful degradation in static environments.
- **Pydantic Models**: Ensure data structures passed out of mocks use the same Pydantic validation models as the live data, maintaining interface parity.
