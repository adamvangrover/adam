"""
Prompts for the 7-phase Systematic Upgrade Agent workflow.
"""

PROMPTS = {
    "Monday": {
        "name": "The Pruning (Deterministic Integrity)",
        "goal": "Remove bloat and non-deterministic \"ghost\" logic to simplify the context window.",
        "prompt": "Analyze this code. Remove all dead code, unused imports, and legacy logic that obfuscates the data flow. Specifically, identify any 'black box' logic that lacks clear input/output types and simplify it. Return the cleaned code and a summary of what was removed to reduce cognitive load on the system."
    },
    "Tuesday": {
        "name": "The Refactor (Architecture & Modularity)",
        "goal": "Decouple data ingestion from logic to facilitate horizontal/vertical scalability.",
        "prompt": "Refactor this module into a clean, reusable architecture. Abstract logic into modular components, prioritizing the separation of raw data ingestion and model-based processing. Apply structural patterns that ensure this module can be independently tested and easily integrated into the broader Adam framework."
    },
    "Wednesday": {
        "name": "The PDIL Optimizer (Performance & Safety)",
        "goal": "Audit the probabilistic-to-deterministic transition.",
        "prompt": "Audit this code for bottlenecks and potential 'stochastic drift'—where model output might violate system invariants. Implement strict schema validation (e.g., Pydantic) to secure the PDIL bridge. Optimize performance for high-throughput and ensure robust error handling that maintains state consistency."
    },
    "Thursday": {
        "name": "The Modernizer (Syntactic & Type Safety)",
        "goal": "Upgrade to the latest idioms and ensure rigid type checking.",
        "prompt": "Modernize this codebase to align with the latest language standards (e.g., Python 3.12+ features, strict typing, async patterns). Replace any anti-patterns that hinder static analysis or AI-readability. Ensure that all type hints are explicit, supporting high-assurance engineering."
    },
    "Friday": {
        "name": "The Innovator (Agentic Enhancement)",
        "goal": "Inject autonomous logic that adheres to the established framework.",
        "prompt": "Analyze this code's core utility. Propose 3 high-value enhancements (e.g., autonomous self-healing, recursive RAG pipelines, or smarter parsing). Implement the most high-value feature, ensuring it includes a 'ProvenanceHeader' to maintain W3C PROV-O compliance and trace decision-making."
    },
    "Saturday": {
        "name": "The Documenter (Context-First Documentation)",
        "goal": "Make the code 'AI-native' for future context windows.",
        "prompt": "Read through this updated code. Generate high-density docstrings that prioritize **Provenance Metadata** (Input/Output schemas, intended state, and derivation paths). Create an 'Architecture & Usage' snippet for the README that allows another AI to fully reconstruct the module's behavior from its documentation."
    },
    "Sunday": {
        "name": "The Validator (Deterministic Test Suite)",
        "goal": "Secure the PDIL bridge with property-based testing.",
        "prompt": "Generate a comprehensive test suite using a modern framework (e.g., `pytest` with `hypothesis`). Focus on property-based testing to stress-test the probabilistic inputs against deterministic boundaries. Ensure all external calls are mocked and that failure states explicitly log the provenance of the erroneous input."
    }
}
