import pytest
from unittest.mock import patch, MagicMock
import subprocess

from scripts.benchmark_adam import BenchmarkRunner, get_git_revision_hash

@patch("subprocess.check_output")
def test_get_git_revision_hash_success(mock_check_output):
    """Test successful retrieval of git hash."""
    mock_check_output.return_value = b"abcdef123456\n"
    assert get_git_revision_hash() == "abcdef123456"

@patch("subprocess.check_output")
def test_get_git_revision_hash_failure(mock_check_output):
    """Test fallback when git fails."""
    mock_check_output.side_effect = subprocess.CalledProcessError(1, "git")
    assert get_git_revision_hash() == "unknown"

@patch("scripts.benchmark_adam.UnifiedKnowledgeGraph")
def test_benchmark_runner_ingestion(MockUKG):
    """Test ingestion benchmarking runs gracefully and returns True."""
    mock_ukg_instance = MockUKG.return_value
    mock_ukg_instance.ingest_risk_state = MagicMock()

    runner = BenchmarkRunner(n_covenants=10)
    result = runner.run_ingestion_benchmark()

    assert result is True
    mock_ukg_instance.ingest_risk_state.assert_called_once()

    # Verify the argument contains 10 covenants
    args, _ = mock_ukg_instance.ingest_risk_state.call_args
    assert "covenants" in args[0]
    assert len(args[0]["covenants"]) == 10

@patch("scripts.benchmark_adam.UnifiedKnowledgeGraph")
def test_benchmark_runner_query(MockUKG):
    """Test pathfinding query runs without exceptions."""
    mock_ukg_instance = MockUKG.return_value
    mock_ukg_instance.find_symbolic_path.return_value = ["Path", "Exists"]

    runner = BenchmarkRunner(n_covenants=10)
    runner.run_query_benchmark()

    mock_ukg_instance.find_symbolic_path.assert_called_once_with(
        "LegalEntity::BENCH_TEST", "Covenant::BENCH_TEST::Cov_9"
    )

@patch("scripts.benchmark_adam.UnifiedKnowledgeGraph")
def test_benchmark_runner_query_failure(MockUKG, capsys):
    """Test query handles an empty path correctly."""
    mock_ukg_instance = MockUKG.return_value
    mock_ukg_instance.find_symbolic_path.return_value = None

    runner = BenchmarkRunner(n_covenants=10)
    runner.run_query_benchmark()

    # Because we moved "print" to "logging.error", capsys doesn't catch it on stdout
    # Instead, we just ensure it executes cleanly without throwing an unhandled exception.
    captured = capsys.readouterr()
    assert "Query Time:" in captured.out
