# How-To: Deploying JULES CI/CD and the Daily Ritual

This guide provides comprehensive instructions on maintaining, modifying, and understanding the GitHub Actions workflow that powers ADAM's automated intelligence lifecycle, specifically the Daily Ritual.

## Prerequisites

Before modifying the CI/CD pipeline, ensure you understand the following core architectural components:
*   **Adam OS & Kernel:** The underlying orchestration layer that coordinates agent interactions.
*   **DAG Orchestration:** The Directed Acyclic Graph used to manage the sequence and dependencies of automated tasks.
*   **Model Context Protocol (MCP) & Knowledge Base:** The centralized data ingestion and search agent functionality that scrapes, simulates, and populates the swarm's memory.
*   **YAML Schema Fine-tuning:** Configuration is driven by strict YAML definitions that align with `pydantic` schemas for type safety.

## Understanding the Pipeline Architecture

The CI/CD pipeline (defined in `.github/workflows/daily_ritual.yml`) serves as the heartbeat of the system. It executes on a scheduled basis (Cron `0 8 * * *`) and performs several critical functions:

1.  **Environment Setup:** Initializes a Python 3.11 environment.
2.  **Dependency Management:** Strictly utilizes `uv` (`uv sync`) for rapid, reliable virtual environment resolution.
3.  **MCP Ingestion & Knowledge Base Update:** Executes `scripts/run_daily_ingestion.py` to ingest new market data, update system context, and allow full observability.
4.  **Security & Risk Assessments:** Runs `scripts/generate_sentinel_assessments.py` (Fortress & Hunt module) to evaluate potential vulnerabilities.
5.  **Market Mayhem Simulations:** Executes `scripts/generate_market_mayhem.py` to simulate tail-risk market events for the Swarm to react to.
6.  **Core Protocol Execution:** Runs `daily_ritual/daily_ritual.py` to facilitate Protocol ARCHITECT_INFINITE, allowing the system to expand itself.
7.  **Multiformat Reporting:** Executes `scripts/generate_human_reports.py` to output the results in a variety of accessible formats (JSON, JSONL, JSON-LD, JSON-RPC, Markdown, TXT, HTML).
8.  **Testing & Verification:** Enforces stability by running the pytest suite (`PYTHONPATH=.:src:core:adam_v3 uv run pytest tests/`) and requiring >80% coverage.
9.  **Provenance Enforcement:** Evaluates outputs via `check_provenance.py` to guarantee W3C PROV-O compliance (e.g., verifying `context_provenance` fields).
10. **Automated Merging:** Submits a Pull Request to the `daily-ritual` branch for review.

## Modifying the Workflow

If you need to change the cron schedule or adjust environment variables, edit `.github/workflows/daily_ritual.yml`.

### Best Practices
*   **Fault Tolerance:** Use `|| true` for generative tasks (like simulations or report generation) to prevent temporary API outages or data sparsity from crashing the entire pipeline.
*   **Strict Dependencies:** Do not revert to `pip` or `poetry`. All new scripts must be executable via `uv run python`.

## Adding a New Step

To add a new processing step (e.g., a new simulation vector), follow this pattern:

```yaml
      - name: Your Custom Simulation Step
        run: uv run python scripts/your_custom_script.py || true
```

Ensure the script outputs data consistently and logs its operations clearly so that downstream tasks (like report generation) can access the results.

## Verifying Testing

Any modifications to the core agents or pipeline scripts must not degrade test coverage.
The pipeline safeguards the integrity of our System 2 logic by executing:

```bash
PYTHONPATH=.:src:core:adam_v3 uv run pytest tests/ --cov=src --cov=adam_v3 --cov-fail-under=80
```

If coverage drops below 80%, the PR will fail. To test locally before committing:
`uv run pytest tests/ --cov=src --cov=adam_v3`

## Troubleshooting

*   **Dependency Resolution Errors:** If a task fails complaining about missing modules, ensure `uv.lock` is up-to-date and that the step prefix uses `uv run`.
*   **Provenance Check Failures:** Ensure that any JSONL data generated explicitly includes the `context_provenance` field.
*   **Coverage Drops:** If you added a new script or agent, you must also provide corresponding unit tests in the `tests/` directory to satisfy the 80% threshold.
*   **Pipeline Hanging:** Long-running generative AI scripts should have appropriate timeouts or fallbacks (e.g., using Mock mode) if the external LLM providers are rate-limiting.