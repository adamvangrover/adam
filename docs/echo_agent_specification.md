# EchoAgent Specification for LLM-Based Generation

## 1. Overview

**Purpose of an EchoAgent:**
The EchoAgent acts as a specialized analytical agent within the "Adam" ecosystem. Its primary function is to process output data from the World Simulation Model (WSM), apply a defined persona or set of analytical guidelines (derived from "Adam System Prompts"), and utilize a Language Model (LLM) via an `LLMPlugin` to draw conclusions, generate insights, or provide analyses based on this simulation data.

**Role in the "Adam" Ecosystem:**
A user typically interacts with a Chatbot Command Line Interface (CLI). This CLI can trigger various workflows, including running the WSM. Once the WSM completes its simulation and produces output data, the EchoAgent is invoked to analyze this data and provide a summarized interpretation or specific insights as requested by the user or a higher-level orchestrator.

## 2. Core Architecture and Components it Interacts With

The EchoAgent is designed to be modular and integrate with several key components of the Adam system:

*   **Inheritance**:
    *   EchoAgent inherits from the `AgentBase` class, defined in `core/agents/agent_base.py`. This ensures it adheres to a common agent interface within the system.

*   **Model Context Protocol (MCP)**:
    *   As a subclass of `AgentBase`, EchoAgent is expected to implement and utilize MCP methods:
        *   `set_context(context_dict: Dict[str, Any])`: Called by the orchestrator or another agent to provide contextual information relevant to the current task (e.g., the user's high-level query objective, simulation ID, timestamps, or specific parameters guiding the analysis).
        *   `get_context() -> Dict[str, Any]`: Returns the agent's current operational context.
        *   `get_skill_schema() -> Dict[str, Any]`: Describes the agent's capabilities, parameters it accepts, and what it returns. For the current EchoAgent, the primary skill is:
            ```json
            {
                "name": "EchoAgent",
                "description": "Agent that analyzes simulation data using an LLM and system guidelines.",
                "skills": [
                    {
                        "name": "analyze_simulation_data",
                        "description": "Processes data from a World Simulation Model, consults an LLM based on system instructions, and returns conclusions.",
                        "parameters": [
                            {"name": "wsm_data", "type": "DataFrame", "description": "Data output from WSM (Pandas DataFrame expected)."}
                        ],
                        "returns": {"type": "str", "description": "Analysis conclusion from the LLM."}
                    }
                ]
            }
            ```
        *   `async def execute(wsm_data: Any, **kwargs: Any) -> Any`: The main asynchronous entry point for the agent to perform its task. It receives WSM data as a primary argument.

*   **World Simulation Model (WSM)**:
    *   EchoAgent is a consumer of data produced by the WSM.
    *   This data is typically a Pandas DataFrame containing time-series data of simulated market variables, economic indicators, geopolitical risk factors, agent portfolio values, etc.

*   **LLMPlugin**:
    *   An instance of `LLMPlugin` (from `core/llm_plugin.py`) is provided to EchoAgent during its initialization.
    *   EchoAgent uses the `llm_plugin.generate_content(context: str, prompt_text: str)` method to interact with an LLM.
    *   The `LLMPlugin` abstracts the actual LLM interaction, which could be a call to a real LLM API (like OpenAI) or a simulated one (like the current mock LLM service, which uses a probability map).

*   **Adam System Prompts / Instructions**:
    *   EchoAgent receives a list of `adam_instructions` (strings) during its initialization. These instructions are typically derived from a master "Adam System Prompt" file (e.g., `docs/Adam v19.2 system prompt.txt`).
    *   These instructions guide EchoAgent's analytical approach, the persona it should reflect in its output, and the style/tone of its generated content. They are formatted and incorporated into the `context` string passed to the `LLMPlugin`.

## 3. Key Methods (Illustrative - based on current EchoAgent)

*   **`__init__(self, app_config: Dict[str, Any], llm_plugin_instance: Any, adam_instructions: Optional[List[str]] = None)`**:
    *   Initializes the agent. It receives the main application configuration (`app_config`), an instance of `LLMPlugin`, and the `adam_instructions`.
    *   It extracts its own agent-specific configuration from `app_config` (e.g., from `app_config['agents']['EchoAgent']`) and passes it to `super().__init__()`.
    *   Stores `llm_plugin_instance` and `adam_instructions`.
    *   Sets up its own logger.

*   **`analyze_simulation_data(self, wsm_data: Any) -> str`**:
    *   This is currently the synchronous core logic method, typically called by the `async execute` method.
    *   **Input**: `wsm_data` (e.g., Pandas DataFrame from WSM).
    *   **Processing**:
        1.  Checks for the availability of the `LLMPlugin` and the `wsm_data`.
        2.  Formats the `wsm_data` into a string representation (e.g., `wsm_data.to_string()`). This forms the primary data context.
        3.  Selects and formats relevant `adam_instructions` to create a `guidelines_section`.
        4.  Combines the `guidelines_section` and the WSM data summary into a `full_context_for_llm`. This context is truncated if it exceeds a predefined length to manage prompt size.
        5.  Formulates a `user_query` (the actual prompt/question for the LLM) that directs the LLM to analyze the data based on the provided context and guidelines. This query is designed to be open-ended yet directive (e.g., "Provide a comprehensive analysis... focusing on key observations, potential implications, and actionable insights...").
        6.  Calls `self.llm_plugin.generate_content(context=full_context_for_llm, prompt_text=user_query)`.
    *   **Output**: Returns the string response received from the `LLMPlugin`.

*   **`detect_environment(self) -> Dict[str, Any]`**:
    *   (Optional, but present in the current EchoAgent). An example of a method that allows the agent to be aware of its operational settings (e.g., configured LLM engine, resource availability). This is not directly part of the core WSM data analysis workflow but contributes to the agent's self-awareness.

## 4. Workflow Example

1.  **Trigger**: The Chatbot CLI receives a user command (e.g., "run wsm and analyze SYMB1 performance").
2.  **WSM Execution**: The system triggers a WSM run, possibly configured with parameters influenced by the user query or system defaults. The WSM produces `wsm_output_data` (e.g., a Pandas DataFrame).
3.  **EchoAgent Invocation**: An instance of `EchoAgent` is created or retrieved.
    *   It is initialized with the application config, an `LLMPlugin` instance, and relevant `adam_instructions` (loaded from the Adam system prompt file).
4.  **Context Setting (MCP)**: `echo_agent.set_context({...})` is called to provide high-level context, such as:
    *   `user_query_objective`: "Analyze SYMB1 performance based on WSM data."
    *   `simulation_id`: A unique identifier for the WSM run.
    *   `timestamp`: Time of the analysis request.
5.  **Execution (MCP)**: `asyncio.run(echo_agent.execute(wsm_data=wsm_output_data))` is called.
6.  **Internal Processing** (within `execute`, which calls `analyze_simulation_data`):
    *   The `wsm_output_data` is converted to a string.
    *   Relevant `adam_instructions` (e.g., focusing on "Actionable Intelligence", "Transparency") are formatted.
    *   A combined context (instructions + WSM data summary) is created.
    *   A specific query is formulated (e.g., "Provide a comprehensive analysis...").
    *   This context and query are passed to `self.llm_plugin.generate_content()`.
7.  **LLM Interaction**: The `LLMPlugin` sends the structured prompt to the configured LLM (which could be the mock service). The mock service uses its `PROBABILITY_MAP` and the content of the prompt (including keywords from WSM data and guidelines) to select a persona-aligned conclusion.
8.  **Response Handling**: The response from `LLMPlugin` (the selected conclusion) is returned up the chain.
9.  **Output to User**: The Chatbot CLI displays this analysis result to the user.

## 5. Customization Points for a New EchoAgent Variant

An LLM tasked with generating a new EchoAgent variant could focus on innovating in the following areas:

*   **`analyze_simulation_data` Logic**:
    *   **WSM Data Interpretation**: How the raw WSM data (DataFrame) is pre-processed, summarized, or transformed before being included in the prompt for the LLM. This could involve statistical summaries, trend detection, or feature extraction.
    *   **Prompt Engineering**: The strategy for combining WSM data, `adam_instructions`, and the specific `user_query` to elicit the best possible response from the LLM. This includes how context is structured and truncated.
    *   **LLM Response Interpretation**: If the LLM's response is structured (e.g., JSON), the new agent might have logic to parse it and format it further before returning.
*   **`adam_instructions` Interaction**:
    *   More sophisticated ways to select, interpret, or even dynamically adjust `adam_instructions` based on the WSM data or MCP context.
*   **Skill Schema (`get_skill_schema`)**:
    *   If the new variant has different analytical capabilities (e.g., can compare two WSM runs, or focus only on risk assessment), its skill schema would need to reflect this.
*   **New Methods**:
    *   Introduction of new methods to handle different types of WSM data (e.g., agent-level data vs. model-level data from the WSM datacollector).
    *   Methods to produce different kinds of outputs (e.g., generating a structured JSON report instead of just a string conclusion).
*   **Error Handling**: Custom error handling related to WSM data anomalies or unexpected LLM responses.

## 6. Goal for LLM-Generated Variant

The primary goals for an LLM generating a new EchoAgent variant based on this specification are:

*   **Adherence to `AgentBase`**: The generated agent class *must* inherit from `AgentBase` and correctly implement all its abstract methods, especially `async execute()` and `get_skill_schema()`.
*   **Component Interaction**: It must correctly initialize and utilize the provided `LLMPlugin` instance and `adam_instructions`. It should be designed to accept WSM data (typically a Pandas DataFrame).
*   **Core Functionality**: The agent should fulfill the fundamental purpose of analyzing WSM data through the lens of system instructions and LLM interaction.
*   **Innovative Contribution**: The LLM's unique contribution would lie in the *internal logic* of how it achieves the analysis – its specific techniques for:
    *   Pre-processing or featurizing WSM data.
    *   Crafting highly effective prompts for its `LLMPlugin` call.
    *   Interpreting or post-processing the response from the `LLMPlugin`.
    *   This could reflect the generating LLM's own "style," embody a novel analytical methodology, or specialize in a particular type of insight extraction from WSM data.

This specification provides the structural and functional skeleton. The generating LLM is expected to "flesh out" the core analytical intelligence within the `analyze_simulation_data` method (or its equivalent in the new variant).
