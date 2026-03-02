from typing import List, Dict, Any, Union
import io
import fitz
from src.ingestion.base import IngestionStrategy
from src.core.logging import logger


class PDFParser(IngestionStrategy):
    """
    Parses PDF files using PyMuPDF (fitz).
    Extracts text from PDF documents.
    """

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def parse(self, file_content: Union[str, bytes, io.BytesIO], **kwargs) -> List[Dict[str, Any]]:
        """
        Parses the PDF file content.

        Args:
            file_content: The raw bytes, string filepath, or a BytesIO object representing the file.
            **kwargs: Additional parameters.

        Returns:
            A list containing a dictionary with the extracted text.
            For now, returning the whole text as a single row to match the return signature,
            but in practice we might want to split it by pages or paragraphs.
        """
        try:
            if isinstance(file_content, bytes):
                doc = fitz.open(stream=file_content, filetype="pdf")
            elif isinstance(file_content, io.BytesIO):
                doc = fitz.open(stream=file_content.read(), filetype="pdf")
            elif isinstance(file_content, str):
                doc = fitz.open(file_content)
            else:
                raise ValueError("Unsupported file_content type for PDFParser")

            full_text = []
            with doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text.strip():
                        full_text.append({"page": page_num + 1, "text": text})

            logger.info(f"Successfully extracted text from {len(full_text)} pages in PDF.")
            return full_text

        except Exception as e:
            logger.error(f"Failed to parse PDF file: {e}")
            raise ValueError(f"Error parsing PDF file: {e}")
