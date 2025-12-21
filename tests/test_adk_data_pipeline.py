import json
import os
import unittest

from core.v23_graph_engine.data_pipeline.graph import create_sequential_data_pipeline


class TestADKDataPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = create_sequential_data_pipeline()
        # Create dummy files for testing
        self.valid_json = "test_data.json"
        with open(self.valid_json, "w") as f:
            json.dump({"title": "Test Report", "content": "Some data", "type": "report"}, f)

        self.invalid_file = "test_bad.unknown"
        with open(self.invalid_file, "w") as f:
            f.write("Some random content")

    def tearDown(self):
        if os.path.exists(self.valid_json):
            os.remove(self.valid_json)
        if os.path.exists(self.invalid_file):
            os.remove(self.invalid_file)

    def test_valid_json_pipeline(self):
        initial_state = {
            "file_path": self.valid_json,
            "pipeline_status": "processing",
            "verification_status": "pending"
        }

        result = self.pipeline.invoke(initial_state)

        self.assertEqual(result["pipeline_status"], "success")
        self.assertEqual(result["verification_status"], "verified")
        self.assertIsNotNone(result["formatted_artifact"])
        self.assertEqual(result["formatted_artifact"]["title"], "Test Report")

    def test_invalid_extension_failure(self):
        initial_state = {
            "file_path": self.invalid_file,
            "pipeline_status": "processing"
        }

        result = self.pipeline.invoke(initial_state)

        self.assertEqual(result["pipeline_status"], "failed")
        self.assertIn("Unsupported file extension", result["error_message"])

    def test_verifier_rejection(self):
        # Create a report missing title to trigger verifier error
        bad_report = "bad_report.json"
        with open(bad_report, "w") as f:
            json.dump({"content": "Missing title", "type": "report"}, f)

        initial_state = {
            "file_path": bad_report,
            "pipeline_status": "processing"
        }

        try:
            result = self.pipeline.invoke(initial_state)
            self.assertEqual(result["pipeline_status"], "failed")
            self.assertEqual(result["verification_status"], "rejected")
            self.assertIn("Report missing 'title' field", result["verification_errors"])
        finally:
            if os.path.exists(bad_report):
                os.remove(bad_report)

if __name__ == "__main__":
    unittest.main()
