"""Tests for document extraction and chunking."""

from __future__ import annotations

import io

import pytest
from pypdf import PdfWriter

from app.services.document import chunk_text, extract_text, process_document


# ── Helpers ──────────────────────────────────────────────────────────

def _make_pdf(pages: list[str]) -> bytes:
    """Create a minimal in-memory PDF with the given page texts."""
    writer = PdfWriter()
    for text in pages:
        writer.add_blank_page(width=72, height=72)
        page = writer.pages[-1]
        # Inject text via a simple content stream.
        from pypdf.generic import ArrayObject, NameObject

        resources = page.get("/Resources")
        if resources is None:
            from pypdf.generic import DictionaryObject
            resources = DictionaryObject()
            page[NameObject("/Resources")] = resources

        # Use a proper font resource so the text renders.
        from pypdf.generic import DictionaryObject as DO
        font_dict = DO(
            {NameObject("/Type"): NameObject("/Font"),
             NameObject("/Subtype"): NameObject("/Type1"),
             NameObject("/BaseFont"): NameObject("/Helvetica")}
        )
        font_res = DO({NameObject("/F1"): font_dict})
        resources[NameObject("/Font")] = font_res  # type: ignore[index]

        from pypdf.generic import ContentStream, DecodedStreamObject
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 50 Td ({text}) Tj ET".encode())
        page[NameObject("/Contents")] = stream

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ── extract_text ─────────────────────────────────────────────────────

class TestExtractText:
    def test_txt_extraction(self) -> None:
        content = "Hello, supplement world!"
        result = extract_text(content.encode(), "notes.txt")
        assert result == content

    def test_pdf_extraction(self) -> None:
        pdf_bytes = _make_pdf(["Page one content"])
        result = extract_text(pdf_bytes, "report.pdf")
        assert "Page one content" in result

    def test_unsupported_extension_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported file type"):
            extract_text(b"data", "image.png")


# ── chunk_text ───────────────────────────────────────────────────────

class TestChunkText:
    def test_empty_string(self) -> None:
        assert chunk_text("") == []

    def test_text_shorter_than_chunk_size(self) -> None:
        chunks = chunk_text("short", chunk_size=100, overlap=10)
        assert chunks == ["short"]

    def test_exact_chunk_size_no_overlap(self) -> None:
        text = "a" * 500
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_produces_correct_count(self) -> None:
        # 1000 chars, chunk_size=500, overlap=50 → step=450
        # chunks start at 0, 450, 900 → 3 chunks
        text = "x" * 1000
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 3

    def test_chunks_overlap_content(self) -> None:
        text = "abcdefghij" * 100  # 1000 chars
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        # The last 50 chars of chunk 0 should equal the first 50 chars of chunk 1.
        assert chunks[0][-50:] == chunks[1][:50]


# ── process_document ─────────────────────────────────────────────────

class TestProcessDocument:
    def test_txt_document(self) -> None:
        text = "a" * 1200
        chunks = process_document(text.encode(), "data.txt", "doc_001")
        assert len(chunks) >= 2
        first = chunks[0]
        assert first["metadata"]["doc_id"] == "doc_001"
        assert first["metadata"]["title"] == "data.txt"
        assert first["metadata"]["chunk_id"] == "doc_001_chunk_0"

    def test_pdf_document_preserves_pages(self) -> None:
        pdf_bytes = _make_pdf(["First page text", "Second page text"])
        chunks = process_document(pdf_bytes, "two_pages.pdf", "doc_002")
        assert len(chunks) >= 2
        pages = {c["metadata"]["page"] for c in chunks}
        assert 1 in pages
        assert 2 in pages
