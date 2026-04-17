# Thursday Execution Plan: The Modernizer (Syntax & Framework Upgrades)

**Objective**: Apply "The Modernizer" sequentially across all major repository domains over a 12-week timeline. Bring code up to date with the absolute latest language features, idioms, and tooling standards.

## Phase 1: Python 3.10+ and Tooling Alignment (Weeks 1-3)
* **Week 1 (Syntax Upgrades)**: Run `uv run ruff check --fix .` globally. Manually upgrade complex conditional chains in `core/` and `backend/` to utilize Python 3.10 Structural Pattern Matching (`match`/`case`).
* **Week 2 (Type Hinting Modernization)**: Sweep all Python domains. Replace outdated `typing` imports (e.g., `List`, `Dict`, `Optional`) with modern built-in equivalents (`list`, `dict`, `str | None`). Enforce strict `mypy` compliance.
* **Week 3 (Dependency Management)**: Eradicate legacy `setup.py` and `requirements.txt` logic. Ensure all deployment scripts and CI pipelines strictly use `uv` and `pyproject.toml` as the sole source of truth.

## Phase 2: Framework Upgrades (Weeks 4-6)
* **Week 4 (Pydantic V2 Migration)**: Review all data models in `backend/` and `core/`. Ensure full compliance with Pydantic V2 (replacing `.dict()` with `.model_dump()`, updating validators, etc.).
* **Week 5 (FastAPI Idioms)**: Update `backend/api.py`. Adopt the latest FastAPI Dependency Injection idioms (e.g., using `Annotated`) and ensure complete OpenAPI schema generation for all endpoints.
* **Week 6 (LLM Framework Alignment)**: Review integration with the `litellm` framework (e.g., in `scripts/daily_ritual.py`). Ensure dynamic multi-model fallbacks are using the most current, non-deprecated API methods.

## Phase 3: Frontend and UI Modernization (Weeks 7-9)
* **Week 7 (React & TypeScript Upgrades)**: Target `services/webapp/client/`. Ensure the project uses modern React functional components and Hooks exclusively. Address legacy peer dependencies (using `--legacy-peer-deps` during `npm install` if necessary).
* **Week 8 (CSS & Styling)**: Modernize the styling approach. Ensure the 'Bloomberg Terminal meets Cyberpunk' aesthetic is implemented using modern CSS variables or utility classes (e.g., Tailwind, if applicable) rather than inline styles.
* **Week 9 (Build Tools)**: Review the frontend build pipeline. If using legacy tools (like an old Webpack config), evaluate the feasibility of upgrading to modern, faster bundlers (like Vite), provided it doesn't break the static HTML generation requirements.

## Phase 4: Scripting and Orchestration Modernization (Weeks 10-12)
* **Week 10 (Bash Script Modernization)**: Review `.sh` scripts in the root and `docker/`. Ensure they use modern POSIX-compliant syntax, handle errors gracefully (`set -e`), and utilize `uv run` prefixes correctly.
* **Week 11 (Docker Configuration)**: Update `docker/docker-compose.yml` and related Dockerfiles. Ensure they use modern multi-stage builds, current base images, and optimal caching strategies.
* **Week 12 (Global Idiom Review)**: Conduct a final pass to ensure modern Python idioms (e.g., walrus operator `:=`, advanced comprehensions) are utilized where they genuinely improve readability without adding undue complexity.
