# Developer Notes: Patterns & Anti-Patterns

Guidelines for contributing to the Adam v26.0 codebase.

## 🏆 Golden Patterns

### 1. The "State" Pattern
Always pass a typed `State` object (TypedDict or Pydantic) between functions in a workflow. Never pass raw unstructured dicts if possible.

```python
# Good
def node(state: ResearchState): ...

# Bad
def node(data: dict): ...
```

### 2. The "Tool" Pattern
Agents should not perform math or side effects directly. They should call **Tools**.
*   *Why?* Tools are deterministic and testable. LLMs are probabilistic.

### 3. The "Fallback" Pattern
Every external API call must have a fallback.
*   If `FMP_API` fails -> Try `YahooFinance`.
*   If both fail -> Use `MockData` (if in dev/test) or raise `GracefulError`.

## 🚫 Anti-Patterns (The "Pheromones" of Failure)

### 1. "God Agents"
Do not create one agent that does everything (Search + Calc + Write).
*   *Fix:* Break it down. One agent per cognitive function.

### 2. Hardcoded Prompts
Never write prompt strings in Python files.
*   *Fix:* Use `prompt_library/` and `load_prompt()`.

### 3. The `pickle` Trap
**NEVER** use `pickle` for serialization. It is a security vulnerability.
*   *Fix:* Use `json` or `msgpack`.

### 4. Global State
Avoid global variables for session data. Adam is designed to be stateless/async.
*   *Fix:* Pass context explicitly or use Redis for persistent session state.
