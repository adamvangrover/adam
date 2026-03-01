from typing import List, Dict, Any, Union
import io
import pandas as pd
from ..base import IngestionStrategy
from src.core.logging import logger

class ExcelParser(IngestionStrategy):
    """
    Parses Excel and CSV files using Pandas.
    Extracts tabular data into a structured list of dictionaries.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [".xlsx", ".xls", ".csv"]

    def parse(self, file_content: Union[str, bytes, io.BytesIO], **kwargs) -> List[Dict[str, Any]]:
        """
        Parses the Excel/CSV file content.

        Args:
            file_content: The raw bytes or a BytesIO object representing the file.
            **kwargs: Additional parameters like 'file_extension' to determine format.

        Returns:
            A list of dictionaries representing the rows in the tabular data.
        """
        extension = kwargs.get("file_extension", ".xlsx").lower()

        try:
            if extension == ".csv":
                if isinstance(file_content, bytes):
                    df = pd.read_csv(io.BytesIO(file_content))
                else:
                    df = pd.read_csv(file_content)
            else:
                if isinstance(file_content, bytes):
                    df = pd.read_excel(io.BytesIO(file_content))
                else:
                    df = pd.read_excel(file_content)

            # Fill NaN values with empty string for cleaner output
            df = df.fillna("")

            # Convert to list of dicts for universal ingestion format
            records = df.to_dict(orient="records")
            logger.info(f"Successfully parsed {len(records)} rows from {extension} file.")

            return records

        except Exception as e:
            logger.error(f"Failed to parse {extension} file: {e}")
            raise ValueError(f"Error parsing {extension} file: {e}")

    def serialize_to_markdown(self, records: List[Dict[str, Any]]) -> str:
        """
        Converts the extracted records into a delimited Markdown table.
        This structured encoding significantly improves LLM comprehension
        and reduces token overhead compared to massive JSON arrays.

        Args:
            records: List of dictionaries representing tabular rows.

        Returns:
            A string containing the formatted Markdown table.
        """
        if not records:
            return ""

        # Reconstruct DataFrame
        df = pd.DataFrame(records)
        return df.to_markdown(index=False)
