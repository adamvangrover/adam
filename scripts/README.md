# Scripts Directory

This directory contains utility scripts for the Adam v24 system.

## Benchmarking

### `benchmark_adam.py`

This script establishes performance baselines for the Unified Knowledge Graph (UKG) by measuring both data ingestion and graph traversal speeds. It logs results to `logs/swarm_telemetry.jsonl`.

**Usage:**

```bash
python3 scripts/benchmark_adam.py
```

**Benchmarks Run:**

1.  **UKG Ingestion:** Measures the time to ingest a risk state containing 5000 covenants.
2.  **UKG Path Query:** Measures the time to find a symbolic path between the root Legal Entity and a deeply nested Covenant node.

**Output:**

- Console output showing the commit hash and execution time for each benchmark.
- Appends structured JSON log entries to `logs/swarm_telemetry.jsonl` with `BENCHMARK_START` and `BENCHMARK_RESULT` events.

**Metrics Tracked:**

- `git_hash`: Commit hash for historical tracking.
- `benchmark_name`: Name of the specific test (e.g., `UKG_Ingestion`, `UKG_PathQuery`).
- `duration_seconds`: Execution time.
- `iterations`: Workload size (e.g., number of covenants).
- `status`: `SUCCESS` or `FAILED`.
