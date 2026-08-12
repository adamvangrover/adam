class DataParser:
    """Universal data handler."""
    @staticmethod
    def parse_document(text: str) -> dict:
        return {"parsed_content": text, "metadata": {"source": "SEC_EDGAR"}}
