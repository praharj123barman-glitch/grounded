"""Load raw documents (PDF, Markdown, text) into a common Document form.

PDFs are loaded page by page so page numbers survive into citations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    text: str
    source: str
    page: int | None = None


def load_dir(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for path in sorted(Path(directory).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            docs.extend(_load_pdf(path))
        elif suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append(Document(text=text, source=path.name))
    return docs


def _load_pdf(path: Path) -> list[Document]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    out: list[Document] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            out.append(Document(text=text, source=path.name, page=i + 1))
    return out
