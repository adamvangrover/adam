```mermaid
graph TD
    subgraph User Interaction Layer
        UI[User Interface (Conversational Agent / IDE Plugin / Web)]
    end

    subgraph CACM-ADK Core Engine
        Orchestrator(CACM Authoring Orchestrator)
        OntologyNav[Ontology Navigator & Expert]
        TemplateEngine[Template Engine]
        WorkflowAssist[Workflow Assistant]
        MetricAdvisor[Metric & Factor Advisor]
        ParamHelper[Parameterization Helper]
        Validator[Semantic & Structural Validator]
        ModularPrompter[Modular Design Prompter]
        DocGen(Documentation Generator - Conceptual)
    end

    subgraph External Dependencies & Services
        LLM_Service[LLM Service (e.g., Vertex AI)]
        OntologyStore[Credit Analysis Ontology Store/Service]
        TemplateRepo[Template Library (e.g., Git Repo)]
        SchemaValidator[Schema Validation Service]
        SemanticValidator[Semantic Validation Service]
        ComputeCatalog[Compute Capability Catalog API]
        CACM_Registry[CACM Registry & Storage API]
    end

    subgraph Developer/Analyst
        User(User: Credit Analyst / Developer)
    end

    %% Interactions
    User -- User Input / Prompts --> UI
    UI -- Requests / User Context --> Orchestrator
    Orchestrator -- LLM Queries / Context --> LLM_Service
    LLM_Service -- LLM Responses / Suggestions --> Orchestrator
    Orchestrator -- Manages Interaction --> UI
    UI -- Generated CACM / Feedback --> User

    %% Core Engine Interactions
    Orchestrator -- Uses --> OntologyNav
    Orchestrator -- Uses --> TemplateEngine
    Orchestrator -- Uses --> WorkflowAssist
    Orchestrator -- Uses --> MetricAdvisor
    Orchestrator -- Uses --> ParamHelper
    Orchestrator -- Uses --> Validator
    Orchestrator -- Uses --> ModularPrompter
    Orchestrator -- Uses --> DocGen

    %% Dependency Interactions
    OntologyNav -- Queries --> OntologyStore
    TemplateEngine -- Fetches Templates --> TemplateRepo
    WorkflowAssist -- Queries Available Capabilities --> ComputeCatalog
    MetricAdvisor -- References --> OntologyStore
    Validator -- Validates Against --> SchemaValidator
    Validator -- Validates Against --> SemanticValidator
    Orchestrator -- Saves/Registers CACM --> CACM_Registry

    %% Data Flow (High Level)
    User -- High-Level Goal --> Orchestrator
    Orchestrator -- Guided Interaction & Suggestions --> User
    Orchestrator -- Compiles --> CACM_Definition(Generated CACM Definition - JSON-LD/YAML)
    CACM_Definition -- Validated by --> Validator
    CACM_Definition -- Stored/Registered --> CACM_Registry
