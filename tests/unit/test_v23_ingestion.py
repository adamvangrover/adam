import unittest
import shutil
import tempfile
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import logging
import json

from core.financial_data.realtime_pipe import MockRealTimePipe
from core.system.agent_improvement_pipeline import AgentImprovementPipeline
from core.data_processing.universal_ingestor_v2 import UniversalIngestor, ArtifactType

# Disable logging during tests
logging.disable(logging.CRITICAL)


class TestV23Ingestion(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.lakehouse_path = Path(self.temp_dir) / "daily"

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        logging.disable(logging.NOTSET)

    def test_realtime_pipe_mock(self):
        pipe = MockRealTimePipe(lakehouse_path=str(self.lakehouse_path), flush_threshold=10)
        pipe.listen()  # Should generate data and flush

        # Check files
        self.assertTrue(self.lakehouse_path.exists())
        # Should have regions and years
        regions = list(self.lakehouse_path.glob("region=*"))
        self.assertTrue(len(regions) > 0)

        # Read a file
        parquet_files = list(self.lakehouse_path.rglob("*.parquet"))
        self.assertTrue(len(parquet_files) > 0)

        df = pd.read_parquet(parquet_files[0])
        self.assertTrue('symbol' in df.columns)
        self.assertTrue('price' in df.columns)

    def test_sensitivity_generator(self):
        # 1. Seed data with a drop
        # Create a parquet file with a drop > 5%
        region_dir = self.lakehouse_path / "region=US" / "year=2023"
        region_dir.mkdir(parents=True, exist_ok=True)

        data = {
            'symbol': ['TEST', 'TEST'],
            'Open': [100.0, 100.0],
            'Close': [101.0, 90.0],  # 2nd day drop 10%
            'Volume': [1000, 2000],
            'timestamp': [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')]
        }
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)

        df.to_parquet(region_dir / "seed.parquet")

        # 2. Run Generator
        pipeline = AgentImprovementPipeline(None, None)
        output_path = Path(self.temp_dir) / "training" / "sensitivity.jsonl"
        pipeline.generate_sensitivity_dataset(
            lakehouse_path=str(self.lakehouse_path),
            output_path=str(output_path)
        )

        # 3. Verify Output
        self.assertTrue(output_path.exists())
        with open(output_path, 'r') as f:
            lines = f.readlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertIn("DETECTED SIGNIFICANT DROP", entry['response'])

    def test_universal_ingestor_parquet(self):
        # Create a parquet file
        file_path = Path(self.temp_dir) / "test.parquet"
        df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
        df.to_parquet(file_path)

        # Run Ingestor
        ingestor = UniversalIngestor(state_file=str(Path(self.temp_dir)/"state.json"))
        # scan_and_process expects a list of paths
        ingestor.scan_and_process([str(self.temp_dir)])

        self.assertEqual(len(ingestor.artifacts), 1)
        self.assertEqual(ingestor.artifacts[0].artifact_type, ArtifactType.DATA.value)
        self.assertIn("col1", ingestor.artifacts[0].metadata['columns'])


if __name__ == '__main__':
    unittest.main()
