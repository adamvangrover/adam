# Semantic Library for Explainable Narrative Generation

## Overview

The `semantic_library` is the declarative core of the Narrative Library project. It serves as a centralized, self-contained knowledge base and configuration hub that defines the "what" and "how" of understanding financial and market dynamics and generating explainable narratives.

This library is designed to be machine-readable (primarily using YAML and JSON) and human-editable. It allows domain experts, developers, and potentially automated systems to define, extend, and refine the logic, concepts, and content used by the `ReasoningEngine` and Large Language Models (LLMs) to produce insightful explanations and analyses.

The ultimate goal is for this library to support:
*   **Deep Domain Understanding:** Capturing complex relationships, driver characteristics, and impact models.
*   **Configurable Narrative Generation:** Allowing fine-grained control over the style, content, and analytical depth of generated narratives.
*   **Probabilistic Impact Assessment:** Defining models to assess potential first, second, and third-order impacts of drivers and events.
*   **Scenario and Simulation Analysis:** Providing definitions for different environments to enable "what-if" analysis.
*   **Potential for Dynamic Refinement:** (Future Goal) Structuring the library so it can be updated or refined based on new data, simulation outcomes, or feedback.

## Structure of the Semantic Library

The `semantic_library` folder is organized as follows:

```
semantic_library/
├── README.md                       # This file: Overview and guide to the library.
├── drivers_knowledge_base.yaml     # Catalog of drivers, their impact models, and narrative logic.
├── narrative_strategies.yaml       # Defines strategies for different types of narrative generation.
├── domain_ontology.yaml            # (Conceptual) Formal definition of entities, properties, relationships.
├── simulation_environment_definitions.yaml # (Conceptual) Definitions for scenarios and world models.
└── schemas/                        # JSON Schemas for validating core data entities.
    ├── company_schema.json
    ├── driver_schema.json
    ├── industry_schema.json
    └── macro_factor_schema.json
└── index.html                      # (Conceptual) A potential entry point or visualization for the library.
```

## Key Configuration Files

### 1. `drivers_knowledge_base.yaml`

*   **Purpose:** Provides a comprehensive, machine-readable catalog of known financial, economic, industry-specific, and company-specific "drivers." A driver is any factor that can influence an entity's performance, valuation, or market perception.
*   **Key Fields per Driver:**
    *   `id`: Unique identifier (e.g., `DRV001`).
    *   `name`: Human-readable name.
    *   `description`: Detailed explanation.
    *   `type`: Category (e.g., `Macroeconomic`, `Fundamental`).
    *   `tags`: Keywords for categorization.
    *   `detection_patterns`: (Conceptual/Future) Formal rules or queries to detect driver activation from data. The `ReasoningEngine` would use these to determine which drivers are relevant for a given context.
    *   `impact_model`: Defines how the driver quantitatively affects target variables. This is a cornerstone for generating analytical insights.
        *   `target_variables`: A list of specific metrics or entity attributes that this driver can influence (e.g., `company.financials.revenue_growth_qoq`, `company.valuation.pe_ratio`).
        *   `first_order_impacts`: A list detailing the direct, quantifiable effects of the driver. Each impact typically specifies:
            *   `target`: The specific variable path (from `target_variables`) affected by this particular impact logic.
            *   `effect_description`: A human-readable summary of the impact.
            *   `effect_function`: The name of a predefined Python function (registered in `impact_calculator.py`) that calculates the impact (e.g., `percentage_change`, `additive_change`). This promotes safety and reusability over embedding raw code/lambda strings in YAML.
            *   `parameters`: A dictionary of parameters required by the `effect_function`. Values can be direct numbers, strings referencing data from the `current_context` (e.g., "company.financials.pe_ratio" to get the current P/E), or simple distribution strings like "Normal(mean,stddev)" or "Uniform(min,max)" (from which the `ImpactCalculator` currently extracts the mean).
            *   `probability_of_occurrence`: A float (0.0 to 1.0) indicating the likelihood of this specific impact manifesting if the driver is active and conditions are met.
            *   `time_horizon`: A string describing the expected timeframe for this impact (e.g., "0-6 months", "1Y").
            *   `conditions`: (Optional) A list of conditions (as strings, e.g., "macro.interest_rate_trend == 'Increasing'") that must be true in the `current_context` for this specific impact to be considered. The `ImpactCalculator` performs basic evaluation of these.
        *   `second_order_effects`, `third_order_effects`: (Conceptual) For defining potential ripple effects or longer-term consequences, currently placeholders for future development.
    *   `narrative_logic`: Guidance for how the LLM should explain this driver and its impacts.
        *   `key_insights_to_extract`.
        *   `data_points_for_explanation`.
        *   `explanation_patterns_llm`: Structured LLM guidance.
    *   `refinement_rules`: (Future) For dynamic library updates.

### 2. `narrative_strategies.yaml`

*   **Purpose:** Defines various strategies for generating different types of narratives (e.g., company impact overview, risk analysis). Each strategy dictates the analytical process and the structure of the output.
*   **Key Fields per Strategy:**
    *   `strategy_id`: Unique identifier (e.g., `STRAT_COMPANY_FINANCIAL_IMPACT_OVERVIEW`).
    *   `description`: Purpose of the strategy.
    *   `target_audience`: Intended reader.
    *   `information_gathering_plan`: Steps for the `ReasoningEngine` to collect necessary data.
    *   `analytical_steps`: Sequence of analyses to perform (e.g., calculate probabilistic impacts).
    *   `narrative_flow`: Ordered sections for the generated narrative, each with specific `llm_instructions`.
    *   `llm_persona_and_tone`: Guidance on LLM style.
    *   `probabilistic_language_rules`: Rules for how the LLM should phrase uncertainty.
    *   `overall_llm_instructions`: General LLM guidance for the entire strategy.

### 3. `domain_ontology.yaml` (Conceptual Placeholder)

*   **Purpose:** Intended to provide a formal definition of all core concepts, entities (Company, Industry, Driver, Event, etc.), their properties (with data types, constraints), and the allowed relationships between them.
*   **Current Status:** This is a placeholder. A full implementation might use a formal ontology language (like OWL) or a detailed YAML/JSON structure.
*   **Future Use:** Would serve as the foundational schema for validating all other definitions in the library and could enable more advanced semantic reasoning.

### 4. `simulation_environment_definitions.yaml` (Conceptual Placeholder)

*   **Purpose:** Designed to define parameters and base states for different simulation environments or "world models." This facilitates "what-if" analysis and scenario testing.
*   **Current Status:** This is a placeholder.
*   **Future Use:** Would allow the `ReasoningEngine` to:
    *   Set up initial states for simulations (e.g., a "recession scenario").
    *   Define scenario-specific events or modifications to driver behaviors.
    *   Generate narratives based on simulation outcomes.

### 5. `schemas/` (JSON Schemas)

*   **Purpose:** Contains JSON Schema files that define the structure and validation rules for the raw data entities (like those found in the CSV sample data or expected from external data sources).
*   **Usage:**
    *   Can be used to validate input data during an ingestion phase.
    *   Provides a clear, machine-readable definition of the data that the Knowledge Graph and `ReasoningEngine` operate on.
    *   Helps ensure consistency between data producers and consumers within the system.
    *   While `domain_ontology.yaml` aims for a richer semantic definition, these JSON schemas offer concrete validation for data instances.

## How the Semantic Library is Used (Intended Architecture)

The `ReasoningEngine` (located in `backend/src/main/python/reasoning_engine.py`) is the primary consumer of this library. The intended workflow is:

1.  **Loading:** Upon initialization, the `ReasoningEngine` loads and parses the YAML configuration files from the `semantic_library`.
2.  **Data Input & Context Building:** The engine fetches relevant data for a target entity (e.g., a company) from the Knowledge Graph. This includes company financials, linked drivers, industry information, and relevant macro factors. This data forms a `current_context` dictionary.
3.  **Driver Identification:** The engine identifies "active" or relevant drivers for the entity (currently based on KG links, future enhancements could use `detection_patterns`).
4.  **Impact Calculation:** For each active driver, the `ReasoningEngine` uses the `ImpactCalculator` module (`backend/src/main/python/impact_calculator.py`).
    *   The `ImpactCalculator` takes the driver's `impact_model` definition (from `drivers_knowledge_base.yaml`) and the `current_context`.
    *   It evaluates any `conditions` specified for each `first_order_impact`.
    *   If conditions are met, it executes the specified `effect_function` (e.g., `percentage_change`) using `parameters` from the YAML (which can be direct values, context references, or parsed distribution strings) to compute a `calculated_impact_value`.
    *   The result is a list of `calculated_impacts`, each including the target variable, calculated value, probability, time horizon, etc.
5.  **Narrative Strategy Selection:** Based on the user's request (e.g., via API or CLI, which can specify a `strategy_id`), an appropriate strategy is selected from `narrative_strategies.yaml`.
6.  **Information Orchestration & Prompt Generation:** The `ReasoningEngine` follows the selected strategy's `information_gathering_plan` (which now includes fetching the `calculated_impacts`) and `analytical_steps`. It then constructs a detailed, context-rich prompt for an LLM:
    *   It follows the `narrative_flow` defined in the strategy, populating sections like `COMPANY_CONTEXT`, `KEY_ACTIVE_DRIVERS_IDENTIFIED`, and importantly, `QUANTITATIVE_IMPACT_ANALYSIS`.
    *   The `QUANTITATIVE_IMPACT_ANALYSIS` section of the prompt is now populated with the actual `calculated_impacts` (formatted values, probabilities, time horizons).
    *   It uses `llm_instructions` from both the strategy (overall and block-specific) and the `narrative_logic.explanation_patterns_llm` from `drivers_knowledge_base.yaml` for each relevant driver and its calculated impacts.
    *   Applying `probabilistic_language_rules` to guide the LLM's expression of uncertainty.
7.  **LLM Interaction:** The generated prompt is sent to an LLM (currently simulated).
8.  **Output:** The LLM's response (the narrative) is returned, structured according to the strategy.

## Extending the Library

To extend the capabilities of the narrative generation system:

*   **Add New Drivers:** Define new entries in `drivers_knowledge_base.yaml` with their impact models and narrative logic.
*   **Refine Existing Drivers:** Update impact parameters, detection patterns, or narrative guidance for existing drivers.
*   **Create New Narrative Strategies:** Define new YAML blocks in `narrative_strategies.yaml` for different analytical tasks or output styles.
*   **Expand the Ontology:** (Future) Add new entity types, properties, or relationships to `domain_ontology.yaml`.
*   **Define New Scenarios:** (Future) Add new environments to `simulation_environment_definitions.yaml`.

This declarative, configuration-driven approach aims to make the system adaptable and scalable.

## Generating Reports from the Library

The rich data defined within this semantic library, particularly in `drivers_knowledge_base.yaml` (which includes probabilistic impact models), can be used to generate structured reports.

Currently, the system supports generating a JSONL report of *calculated* high-probability driver impacts for specified companies. This report leverages the `ImpactCalculator` and the `impact_model` definitions.

It can be generated via the CLI:
```bash
# Ensure PYTHONPATH is set to project root: export PYTHONPATH=.
python backend/src/main/python/cli.py report driver-impacts --company-ids AAPL,MSFT --output-file reports/calculated_impacts.jsonl --min-probability 0.65
```
This command uses the `generate_driver_impact_report.py` tool. The script now:
1. Takes company IDs as input (or defaults to processing companies from the KG).
2. For each company, it uses the `ReasoningEngine` to get `calculated_impacts`. This involves building a context for the company and then applying the `impact_model` for its active drivers via the `ImpactCalculator`.
3. Filters these *calculated* impacts by the `--min-probability`.
4. Outputs a JSONL file where each line contains details of a calculated impact, including the `calculated_impact_value`, `probability`, `target_variable`, `source_driver_id`, etc.

This provides a structured way to see the quantitative outputs of the impact assessment system.
