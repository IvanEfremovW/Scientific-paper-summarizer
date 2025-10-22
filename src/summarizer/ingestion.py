import pymupdf
from pathlib import Path


def exctract_text_from_document(document_path: str | Path) -> str:
    with pymupdf.open(document_path) as doc:
        text = "".join(page.get_text("text") for page in doc)  # type: ignore

    return text
