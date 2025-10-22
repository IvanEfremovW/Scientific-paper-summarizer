import pymupdf
import tempfile
from src.ingestion import exctract_text_from_document


def test_extract_text_from_document():
    # Создаём временный PDF с текстом
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        document_path = tmp.name

    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((50, 50), "Abstract: This is a test scientific paper.")
        doc.save(document_path)

    text = exctract_text_from_document(document_path)

    assert "Abstract" in text
    assert "scientific paper" in text
