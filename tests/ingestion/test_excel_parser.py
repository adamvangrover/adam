import io
import pandas as pd
from src.ingestion.plugins.excel_parser import ExcelParser

def test_excel_parser_parse_csv():
    # Setup mock CSV
    csv_data = "id,name,value\n1,Alice,10.5\n2,Bob,20.0"
    file_content = io.BytesIO(csv_data.encode("utf-8"))

    # Init Parser
    parser = ExcelParser()

    # Parse
    results = parser.parse(file_content, file_extension=".csv")

    # Verify
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["name"] == "Alice"
    assert results[1]["value"] == 20.0

def test_excel_parser_serialize_markdown():
    # Setup list of dicts
    records = [
        {"id": 1, "name": "Alice", "value": 10.5},
        {"id": 2, "name": "Bob", "value": 20.0}
    ]

    # Init Parser
    parser = ExcelParser()

    # Serialize
    markdown = parser.serialize_to_markdown(records)

    # Verify markdown string contains expected table structures
    assert "| id | name  | value |" in markdown.replace(" ", "") or "|id|name|value|" in markdown.replace(" ", "")
    assert "| 1 | Alice | 10.5  |" in markdown.replace(" ", "") or "|1|Alice|10.5|" in markdown.replace(" ", "")
