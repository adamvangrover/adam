import io

import fitz
import pytest

from src.ingestion.plugins.pdf_parser import PDFParser


def test_pdf_parser_supported_extensions():
    parser = PDFParser()
    assert ".pdf" in parser.supported_extensions

def test_pdf_parser_parse_valid_pdf_bytes():
    parser = PDFParser()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 50), "Test PDF Content")
    pdf_bytes = doc.write()
    doc.close()

    result = parser.parse(pdf_bytes)
    assert len(result) == 1
    assert result[0]["page"] == 1
    assert "Test PDF Content" in result[0]["text"]

def test_pdf_parser_parse_valid_pdf_bytes_io():
    parser = PDFParser()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(fitz.Point(50, 50), "Test PDF Content BytesIO")
    pdf_bytes = doc.write()
    doc.close()

    pdf_io = io.BytesIO(pdf_bytes)

    result = parser.parse(pdf_io)
    assert len(result) == 1
    assert result[0]["page"] == 1
    assert "Test PDF Content BytesIO" in result[0]["text"]

def test_pdf_parser_invalid_content_type():
    parser = PDFParser()
    with pytest.raises(ValueError):
         parser.parse("this is a string, not bytes")
