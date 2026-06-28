# How-To: Deploying JULES CI/CD

This guide provides instructions on modifying the GitHub Actions workflow for the daily automated intelligence ritual (e.g., Market Mayhem).

## The Daily Ritual
The `daily_ritual.py` process is integrated into a GitHub Actions CI/CD workflow (Cron `0 8 * * *`). It requires the test suite to pass with 80%+ coverage and the `ProvenanceHeader` validator to be satisfied before merging.

## Step 1: Locate the Workflow File
The CI/CD pipeline is defined in `.github/workflows/daily_ritual.yml`.

## Step 2: Modify the Pipeline
If you need to add a new intelligence report or change the cron schedule, edit the YAML file. Ensure that tests are always run before executing the final ritual script.

```yaml
jobs:
  daily-intelligence:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Install dependencies
        run: uv sync
      - name: Run Tests
        run: PYTHONPATH=.:src:core:adam_v3 uv run pytest tests/unit/
      - name: Execute Daily Ritual
        run: uv run python scripts/daily_ritual.py
```

## Step 3: Verify Test Coverage
Any changes must not degrade test coverage. If coverage drops below 80%, the pipeline will fail, safeguarding the integrity of our System 2 logic.
