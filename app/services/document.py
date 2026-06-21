"""PDF and text extraction utilities.

Handles ingestion of user-uploaded documents: extracts raw text,
splits it into overlapping chunks, and returns chunk dicts ready
for insertion into the vector store.

PDFs are parsed with opendataloader-pdf, which returns structured
elements (headings, paragraphs, tables, …) with page numbers and
bounding boxes. These are grouped into semantic chunks that respect
document structure rather than slicing at fixed character counts.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

_MAX_CHUNK_CHARS = 1000


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[str]:
    """Split *text* into fixed-size character chunks with overlap (used for TXT files)."""
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


def _semantic_chunks(elements: list[dict], doc_id: str, filename: str) -> list[dict]:
    """Group opendataloader-pdf elements into semantic chunks.

    Starts a new chunk at every heading or when the running character
    count exceeds _MAX_CHUNK_CHARS. Page number and bounding box are
    taken from the first element in each chunk.
    """
    chunks: list[dict] = []
    chunk_idx = 0
    current_parts: list[str] = []
    current_page: int = 1
    current_bbox: list[float] | None = None

    def _flush() -> None:
        nonlocal chunk_idx, current_parts, current_page, current_bbox
        text = "\n".join(current_parts).strip()
        if text:
            chunks.append({
                "text": text,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{chunk_idx}",
                    "title": filename,
                    "source": filename,
                    "page": current_page,
                    "bbox": current_bbox,
                },
            })
            chunk_idx += 1
        current_parts = []
        current_bbox = None

    for elem in elements:
        content = (elem.get("content") or "").strip()
        if not content:
            continue

        elem_type = elem.get("type", "paragraph")
        page = elem.get("page number", 1)
        bbox = elem.get("bounding box")

        if elem_type == "heading":
            _flush()
            current_page = page
            current_bbox = bbox
            current_parts = [content]
        else:
            running_len = sum(len(p) for p in current_parts)
            if current_parts and running_len + len(content) > _MAX_CHUNK_CHARS:
                _flush()
                current_page = page
                current_bbox = bbox
            if not current_parts:
                current_page = page
                current_bbox = bbox
            current_parts.append(content)

    _flush()
    return chunks


def _process_pdf(file_bytes: bytes, filename: str, doc_id: str) -> list[dict]:
    """Parse a PDF with opendataloader-pdf and return semantic chunks."""
    import opendataloader_pdf

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_file = tmp_path / filename
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        input_file.write_bytes(file_bytes)

        opendataloader_pdf.convert(
            input_path=[str(input_file)],
            output_dir=str(output_dir),
            format="json",
        )

        stem = Path(filename).stem
        elements: list[dict] = json.loads((output_dir / f"{stem}.json").read_text())["kids"]

    logger.info("opendataloader-pdf extracted %d elements from %s", len(elements), filename)
    return _semantic_chunks(elements, doc_id, filename)


def process_document(
    file_bytes: bytes,
    filename: str,
    doc_id: str,
) -> list[dict]:
    """Extract, chunk, and annotate a document for the vector store.

    Returns:
        List of chunk dicts, each containing:
        - text: the chunk content
        - metadata: {doc_id, chunk_id, title, source, page, bbox}
    """
    ext = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""

    if ext == "pdf":
        return _process_pdf(file_bytes, filename, doc_id)

    if ext == "txt":
        full_text = file_bytes.decode("utf-8")
        return [
            {
                "text": piece,
                "metadata": {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_chunk_{idx}",
                    "title": filename,
                    "source": filename,
                    "page": 1,
                    "bbox": None,
                },
            }
            for idx, piece in enumerate(chunk_text(full_text))
        ]

    raise ValueError(f"Unsupported file type: .{ext}")
