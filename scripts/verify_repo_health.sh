#!/usr/bin/env bash
# Golden-path repo health check. Runs, in order:
#   1. uv sync --frozen --inexact --extra data-science --extra orchestration  (the blueprint's extras; never hand-edit
#      uv.lock; --all-extras would pull the Linux-only triton wheel via ml-tensors)
#      + requirements/base.txt, as the test-unit CI job installs it
#   2. ruff check .
#   3. pytest tests/ --strict-markers           (same selection as test-unit CI; tests/quarantine.txt = non-strict xfail)
#   4. cargo test core/experimental/adamos_kernel (as ci-rust.yml) + cargo check rust_ext (PyO3 extension;
#      its test binary can't link libpython with the extension-module feature)
#   5. scripts/golden_path_smoke.py              (System 1, System 2, Nexus, JSONLogic governance + PROV-O)
#
# Usage: scripts/verify_repo_health.sh [--skip-sync] [--skip-rust] [--quick]
#   --quick  runs tests/unit + tests/regression instead of the full suite.
set -uo pipefail
cd "$(dirname "$0")/.."

SKIP_SYNC=0 SKIP_RUST=0 QUICK=0
for arg in "$@"; do
  case "$arg" in
    --skip-sync) SKIP_SYNC=1 ;;
    --skip-rust) SKIP_RUST=1 ;;
    --quick) QUICK=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if command -v uv >/dev/null 2>&1; then RUN=(uv run --no-sync); else RUN=(); fi
[[ -x .venv/bin/python && ${#RUN[@]} -eq 0 ]] && export PATH="$PWD/.venv/bin:$PATH"

declare -a RESULTS=()
FAILED=0
step() {
  local name="$1"; shift
  printf '\n==> %s\n' "$name"
  local t0=$SECONDS
  if "$@"; then RESULTS+=("PASS  $name ($((SECONDS - t0))s)")
  else RESULTS+=("FAIL  $name ($((SECONDS - t0))s)"); FAILED=1; fi
}
skip() { RESULTS+=("SKIP  $1"); }

if [[ $SKIP_SYNC -eq 1 ]]; then skip "uv sync --frozen"
elif command -v uv >/dev/null 2>&1; then step "uv sync --frozen" uv sync --frozen --inexact --extra data-science --extra orchestration
else skip "uv sync --frozen (uv not installed)"; fi
if [[ $SKIP_SYNC -eq 0 ]] && command -v uv >/dev/null 2>&1 && [[ -f requirements/base.txt ]]; then
  step "install requirements/base.txt (test-unit CI deps)" \
    uv pip install --python .venv/bin/python -q -r requirements/base.txt pytest-timeout hypothesis aiosqlite
fi

step "ruff check ." "${RUN[@]}" ruff check .

if [[ $QUICK -eq 1 ]]; then TESTS=(tests/unit tests/regression); else TESTS=(tests/); fi
TIMEOUT=()
"${RUN[@]}" python -c 'import pytest_timeout' 2>/dev/null && TIMEOUT=(--timeout=120) \
  || echo "note: pytest-timeout not installed; running without the 120 s per-test timeout"
step "pytest ${TESTS[*]} --strict-markers" env PYTHONPATH=. "${RUN[@]}" python -m pytest "${TESTS[@]}" -q \
  --strict-markers -p no:cacheprovider -m "not integration" --ignore=tests/unit/test_v23_5_pipeline.py "${TIMEOUT[@]}"

if [[ $SKIP_RUST -eq 1 ]]; then skip "cargo"
elif ! command -v cargo >/dev/null 2>&1; then skip "cargo (cargo not installed)"
else
  # PyO3 builds against the project interpreter (abi3-py311 needs >= 3.11).
  PYO3_PYTHON=$("${RUN[@]}" python -c 'import sys; print(sys.executable)'); export PYO3_PYTHON
  step "cargo test core/experimental/adamos_kernel" cargo test --quiet --manifest-path core/experimental/adamos_kernel/Cargo.toml
  step "cargo check rust_ext" cargo check --quiet --manifest-path rust_ext/Cargo.toml
fi

step "golden-path smoke (S1, S2, Nexus, governance, PROV-O)" "${RUN[@]}" python scripts/golden_path_smoke.py

printf '\n==> Summary\n'
printf '  %s\n' "${RESULTS[@]}"
exit $FAILED
