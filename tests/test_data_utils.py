# tests/test_data_utils.py

import csv
import json
import os
import tempfile
import unittest

import yaml

from core.system.error_handler import FileReadError, InvalidInputError
from core.utils.data_utils import load_data


class TestDataUtils(unittest.TestCase):

    def test_load_data_json_success(self):
        # Create a temporary JSON file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmpfile:
            json.dump({"key": "value"}, tmpfile)
            tmpfile_path = tmpfile.name

        data = load_data({"type": "json", "path": tmpfile_path})
        self.assertEqual(data, {"key": "value"})
        os.remove(tmpfile_path)

    def test_load_data_json_file_not_found(self):
        with self.assertRaises(FileReadError) as context:
            load_data({"type": "json", "path": "nonexistent_file.json"})
        self.assertEqual(context.exception.code, 105)

    def test_load_data_json_invalid_json(self):
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json") as tmpfile:
            tmpfile.write("{invalid json")  # Missing closing brace
            tmpfile_path = tmpfile.name
        with self.assertRaises(FileReadError) as context:
            load_data({"type": "json", "path": tmpfile_path})
        self.assertEqual(context.exception.code, 105)
        os.remove(tmpfile_path)

    def test_load_data_csv_success(self):
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".csv") as tmpfile:
            writer = csv.writer(tmpfile)
            writer.writerow(["header1", "header2"])
            writer.writerow(["value1", "value2"])
            tmpfile_path = tmpfile.name

        data = load_data({"type": "csv", "path": tmpfile_path})
        self.assertEqual(data, [{"header1": "value1", "header2": "value2"}])
        os.remove(tmpfile_path)

    def test_load_data_csv_file_not_found(self):
         with self.assertRaises(FileReadError) as context:
            load_data({"type": "csv", "path": "nonexistent_file.csv"})
         self.assertEqual(context.exception.code, 105)

    def test_load_data_yaml_success(self):
        # Create a temporary YAML file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as tmpfile:
            yaml.dump({"key": "value"}, tmpfile)
            tmpfile_path = tmpfile.name

        data = load_data({"type": "yaml", "path": tmpfile_path})
        self.assertEqual(data, {"key": "value"})
        os.remove(tmpfile_path)

    def test_load_data_yaml_file_not_found(self):
        with self.assertRaises(FileReadError) as context:
            load_data({"type": "yaml", "path": "nonexistent.yaml"})
        self.assertEqual(context.exception.code, 105)

    def test_load_data_yaml_invalid_yaml(self):
        # Create temp file with invalid YAML.
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".yaml") as tmpfile:
            tmpfile.write("invalid yaml: -") # Missing closing
            tmpfile_path = tmpfile.name

        with self.assertRaises(FileReadError) as context:
            load_data({"type": "yaml", "path": tmpfile_path})
        self.assertEqual(context.exception.code, 105)
        os.remove(tmpfile_path)

    def test_load_data_unsupported_type(self):
        with self.assertRaises(InvalidInputError) as context:
            load_data({"type": "unsupported", "path": "some_file.xyz"})
        self.assertEqual(context.exception.code, 103)

if __name__ == '__main__':
    unittest.main()
