from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
import io

class IngestionStrategy(ABC):
    """
    Abstract Base Class for all ingestion strategies.
    Ensures that any new format parsing plugin implements the parse method.
    """

    @abstractmethod
    def parse(self, file_content: Union[str, bytes, io.BytesIO], **kwargs) -> List[Dict[str, Any]]:
        """
        Parses the incoming file content and returns a structured representation.

        Args:
            file_content: The raw content of the file (bytes, text, or file-like object).
            **kwargs: Additional parameters for parsing.

        Returns:
            A list of dictionaries representing the parsed data. For tabular data,
            this is typically a list of rows, where each row is a dict mapping
            column names to values.
        """
        pass

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """
        Returns a list of file extensions supported by this strategy.
        Example: ['.xlsx', '.xls']
        """
        pass
