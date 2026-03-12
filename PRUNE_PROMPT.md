# AI Pruning Prompt for Adam Repository

Below is a prompt you can use with Gemini (or another LLM) to safely prune, consolidate, and optimize the Adam repository so that it fits within context windows while preserving its functionality.

---

### **Prompt Instructions for Gemini**

**System Context:**
You are an expert Principal Engineer and Code Architect assisting with the "Adam" repository. The repository is a complex, multi-layered financial AI sovereign (Adam v26.0) utilizing a Hybrid Cognitive Engine (System 1 Swarm + System 2 Graph) built in Python (with some Rust concepts/components). It's currently too large (228MB+) and contains many redundant, obsolete, or artifact files that are overwhelming my context window.

**Goal:**
Your objective is to help me aggressively prune, consolidate, and clean up this repository while strictly adhering to the "Prime Directive" (do not break core functionality in `core/` and `services/`) and ensuring adherence to the project's memory directives.

**Actionable Constraints & Directives:**
1. **Archive Dead Code (Memory Rule):** Active UI/UX consolidation is required. Actively identify and move old, obsolete, or incorrect HTML files into the `archive/` directory rather than leaving them as dead code. Do NOT delete them entirely; move them and update any relevant static lists (e.g., exclude from `scripts/generate_market_mayhem_archive.py`).
2. **Preserve System Boundaries:** Maintain the strict bifurcation between Path A (Reliability - `core/`) and Path B (Lab - `experimental/`, `research/`, `tinker_lab/`).
3. **Additive Refactoring only:** Any changes to existing Product components (`core/`, `services/`) should follow the "purely additive" or "graceful degradation" paradigm when possible.
4. **Target Bloat First:** Focus your analysis on root-level `.txt` files, multiple `.sh`/`.py` fix scripts (`fix_lint.sh`, `fix_json_v2.py`, etc.), repeated `.html` files (`index2.html`, `index3.html`, etc.), log files (`*.log`), screenshot/image folders (`verification_screenshots`, `verification_images`), and data exports that are taking up space.
5. **Consolidate Scripts:** There are numerous `fix_*` and `verify_*` scripts in the root directory. Combine these into unified CLI commands in the `scripts/` folder or propose their archival if obsolete.

**Your Tasks (Please provide the output step-by-step):**

*   **Task 1: Identify Root Cause Bloat:** Give me a specific list of `find` or `rm` commands to execute that will safely delete unneeded artifacts (e.g., `.egg-info`, `__pycache__`, raw logs in root, temporary images like `newsletter_1929.png`, redundant text files like `final_check_3.txt`).
*   **Task 2: HTML/UI Consolidation Plan:** Review the `showcase/` and root directory HTML files (`index.html`, `index2.html`, `readme.html`, etc.). Provide a bash script to move obsolete ones to `archive/` and update references.
*   **Task 3: Script Consolidation:** Review the scattered root `.py` and `.sh` scripts (`reproduce_debug_mode.py`, `patch_edgar.py`, `benchmark_rng.py`, `app.py`, `run_ui.sh`, etc.). Which of these can be safely archived or moved to the `scripts/` directory? Provide the commands to do so.
*   **Task 4: Identify Core Duplication:** Are there duplicate utility functions or deprecated communication patterns (e.g., old `pika` code, direct file graph reads) that I should remove in favor of modern `UnifiedKnowledgeGraph` architecture? Tell me which files to target for refactoring.

Please provide exact, safe bash commands I can run to perform these cleanup tasks.

---

### **How to use this:**
Copy the text above starting from **System Context:** to the end, and paste it into your AI assistant. Follow its specific instructions to execute the cleanup safely.