This is a highly effective approach. By feeding your codebase to an LLM in targeted, bite-sized chunks with specific objectives, you can systematically upgrade your repository without overwhelming the context window.

### The Enhanced Weekly Repository Upgrade System

**System Instructions:** Review the current session context. Select the daily task based on the **Current System State**, **Provenance Requirements**, and **Modular Stability**. Focus on **additive impact**—upgrading the code to be more deterministic, modular, and AI-readable.

Here is a weekly schedule of powerful, specialized prompts you can run daily against your code. Whenever you use these, paste the relevant prompt along with the specific files or modules you want to focus on that day.

#### Monday: The Pruning (Deterministic Integrity)
*Goal: Remove bloat and non-deterministic "ghost" logic to simplify the context window.*
> "Analyze this code. Remove all dead code, unused imports, and legacy logic that obfuscates the data flow. Specifically, identify any 'black box' logic that lacks clear input/output types and simplify it. Return the cleaned code and a summary of what was removed to reduce cognitive load on the system."

#### Tuesday: The Refactor (Architecture & Modularity)
*Goal: Decouple data ingestion from logic to facilitate horizontal/vertical scalability.*
> "Refactor this module into a clean, reusable architecture. Abstract logic into modular components, prioritizing the separation of raw data ingestion and model-based processing. Apply structural patterns that ensure this module can be independently tested and easily integrated into the broader Adam framework."

#### Wednesday: The PDIL Optimizer (Performance & Safety)
*Goal: Audit the probabilistic-to-deterministic transition.*
> "Audit this code for bottlenecks and potential 'stochastic drift'—where model output might violate system invariants. Implement strict schema validation (e.g., Pydantic) to secure the PDIL bridge. Optimize performance for high-throughput and ensure robust error handling that maintains state consistency."

#### Thursday: The Modernizer (Syntactic & Type Safety)
*Goal: Upgrade to the latest idioms and ensure rigid type checking.*
> "Modernize this codebase to align with the latest language standards (e.g., Python 3.12+ features, strict typing, async patterns). Replace any anti-patterns that hinder static analysis or AI-readability. Ensure that all type hints are explicit, supporting high-assurance engineering."

#### Friday: The Innovator (Agentic Enhancement)
*Goal: Inject autonomous logic that adheres to the established framework.*
> "Analyze this code's core utility. Propose 3 high-value enhancements (e.g., autonomous self-healing, recursive RAG pipelines, or smarter parsing). Implement the most high-value feature, ensuring it includes a 'ProvenanceHeader' to maintain W3C PROV-O compliance and trace decision-making."

#### Saturday: The Documenter (Context-First Documentation)
*Goal: Make the code 'AI-native' for future context windows.*
> "Read through this updated code. Generate high-density docstrings that prioritize **Provenance Metadata** (Input/Output schemas, intended state, and derivation paths). Create an 'Architecture & Usage' snippet for the README that allows another AI to fully reconstruct the module's behavior from its documentation."

#### Sunday: The Validator (Deterministic Test Suite)
*Goal: Secure the PDIL bridge with property-based testing.*
> "Generate a comprehensive test suite using a modern framework (e.g., `pytest` with `hypothesis`). Focus on property-based testing to stress-test the probabilistic inputs against deterministic boundaries. Ensure all external calls are mocked and that failure states explicitly log the provenance of the erroneous input."

---

### Execution Strategy for Adam

* **The "Drift" Flag:** In every output, you must explicitly include an **"Observed Drift"** section. If the refactor changes the module's behavior significantly, it must be flagged for manual review to ensure the "Adam" repository's collective determinism is not compromised.
* **Additive Logic:** Every change should focus on **reusability**. If you find yourself writing a new function, check if it can be elevated to the `core_types.py` level for horizontal use across Odyssey/Sentinel.

Pro-Tip for Execution: When you run these, don't just dump the whole repo in at once. Pick a specific feature, folder, or file (e.g., "Today I am running the Monday and Tuesday prompts on my database utility functions") and cycle through your repository systematically.
