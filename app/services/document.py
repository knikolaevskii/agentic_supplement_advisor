"""PDF and text extraction utilities.

Handles ingestion of user-uploaded documents: extracts raw text,
splits it into overlapping chunks, and returns chunk dicts ready
for insertion into the vector store.
"""

from __future__ import annotations

import io
import logging

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def extract_text(file_bytes: bytes, filename: str) -> str:
    """Extract plain text from a PDF or TXT file.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used to determine format).

    Returns:
        Extracted text as a single string.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    ext = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n".join(pages)

    if ext == "txt":
        return file_bytes.decode("utf-8")

    raise ValueError(f"Unsupported file type: .{ext}")


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into fixed-size character chunks with overlap.

    Args:
        text: The full document text.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def process_document(
    file_bytes: bytes,
    filename: str,
    doc_id: str,
) -> list[dict]:
    """Extract, chunk, and annotate a document for the vector store.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename.
        doc_id: Unique identifier assigned to this document.

    Returns:
        List of chunk dicts, each containing:
        - text: the chunk content
        - metadata: {doc_id, chunk_id, title, source, page}
    """
    ext = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""

    # For PDFs we chunk per page to preserve page numbers.
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        chunks: list[dict] = []
        chunk_idx = 0
        for page_num, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for piece in chunk_text(page_text):
                chunks.append({
                    "text": piece,
                    "metadata": {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                        "title": filename,
                        "source": filename,
                        "page": page_num,
                    },
                })
                chunk_idx += 1
        return chunks

    # For plain text there is no page concept.
    full_text = extract_text(file_bytes, filename)
    return [
        {
            "text": piece,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{idx}",
                "title": filename,
                "source": filename,
                "page": 1,
            },
        }
        for idx, piece in enumerate(chunk_text(full_text))
    ]
