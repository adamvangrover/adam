from typing import List, Optional
from pydantic import BaseModel, Field

class ProcessedRow(BaseModel):
    """
    Structured validation schema representing a successfully processed tabular row.
    Maps directly back to the original spreadsheet format.
    """

    # Using generic fields allowing adaptation based on specific prompts
    # In a real-world scenario, you might dynamically generate schemas
    # based on the expected columns, but this provides a strong foundation.

    original_id: Optional[str] = Field(None, description="The exact row ID from the source spreadsheet, if applicable")

    # Generic entity extraction and transformations
    extracted_entities: List[str] = Field(default_factory=list, description="Names, organizations, or key entities identified in the row")
    translated_text: Optional[str] = Field(None, description="The translated text, if requested in the system prompt")
    sentiment_score: Optional[int] = Field(None, description="Sentiment score from 1 to 10")

    # Dictionary of transformed columns and their new values
    transformed_columns: dict[str, str] = Field(default_factory=dict, description="A mapping of original column names to their new transformed values")

class SpreadsheetBatchOutput(BaseModel):
    """
    Schema for validating the batch output from the LLM.
    Ensures that the LLM returns structured rows that can be reassembled.
    """
    processed_rows: List[ProcessedRow] = Field(description="A list of processed rows mapping back to the input batch")
