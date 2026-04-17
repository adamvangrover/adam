# Monday Execution Plan: The Pruning (Comprehensive System Purge)

**Objective**: Apply "The Pruning" sequentially across all major repository domains over a 12-week timeline. Strip bloat, unused imports, dead code, and legacy artifacts to improve clarity across the entire swarm architecture.

## Phase 1: Core Engine and Backend Purge (Weeks 1-3)
* **Week 1 (Backend API & Controller)**: Run static analysis tools (`uv run ruff`) on `backend/api.py` and `backend/controller.py`. Remove deprecated API endpoints, unused LLM logic gate variants, and obsolete telemetry trackers.
* **Week 2 (Core v30 Architecture)**: Target `core/v30_architecture/python_intelligence/`. Prune unreferenced agent definitions and orphaned helper functions. Remove legacy `sys.path.append` hacks.
* **Week 3 (Legacy Agent Purge)**: Target `core/agents/`. Identify and purge deprecated pre-v29 agent logic that has been fully superseded by the v30 architecture. Ensure `EngineFactory` fallback logic remains intact.

## Phase 2: Operations, Scripts, and Automation (Weeks 4-6)
* **Week 4 (Operational Scripts)**: Scan the `scripts/` directory. Delete iterative LLM generation artifacts (e.g., temporary `patch_*.py` files). Consolidate redundant daily generation scripts (e.g., merging overlapping `generate_daily_index.py` tasks).
* **Week 5 (Repo Maintenance & Docker)**: Prune unused or redundant containers from `docker/docker-compose.yml`. Clean up legacy `.sh` startup scripts that conflict with the modern `uv` pipeline.
* **Week 6 (Archive Cleanup)**: Systematically review `archive/` and temporary `.log` files in the root. Permanently delete outdated UI backups (e.g., `index2.html`) and enforce `.gitignore` rules against future clutter.

## Phase 3: Data, Mocks, and Prompt Libraries (Weeks 7-9)
* **Week 7 (Mock Ecosystem)**: Audit `config/mocks/` and mock connectors (e.g., `MockLLM` in `core/llm_plugin.py`). Ensure all empty text stubs are removed and replaced with functional, typed Pydantic fallback logic to maintain graceful degradation.
* **Week 8 (Data Artifacts)**: Target `showcase/` and `verification_images/`. Identify orphaned JSON data blobs or screenshots that are no longer referenced by the generated HTML dashboards.
* **Week 9 (Prompt Library Consolidation)**: Scan `prompt_library/AOPL-v2.0/`. Remove redundant or outdated prompt variations. Consolidate overlapping prompts for the `DynamicSearchAgent`.

## Phase 4: Frontend, Tests, and Validation (Weeks 10-12)
* **Week 10 (Webapp UI Pruning)**: Target `services/webapp/client/`. Remove unused React components, obsolete CSS classes (ensure adherence to the 'Cyberpunk' theme), and dead TypeScript interfaces.
* **Week 11 (Test Suite Trimming)**: Scan the `tests/` directory. Remove obsolete tests for purged pre-v29 agents. Address or safely ignore known unfixable test collection errors as specified in memory.
* **Week 12 (Global Verification)**: Run the full `uv run bandit` and `uv run pytest` suites. Ensure the aggressive pruning across all domains has not introduced regressions or violated the 'Static Mock Mode' fallback contract.
