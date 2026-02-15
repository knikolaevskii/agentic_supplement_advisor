"""Citation formatting helpers."""

from __future__ import annotations

from app.models.schemas import Citation


def format_citations(
    chunks: list[dict],
) -> tuple[str, list[Citation]]:
    """Build a numbered context string and matching :class:`Citation` list.

    Each chunk dict is expected to have ``text`` and ``metadata`` keys
    (as returned by :meth:`VectorStoreService.search`).

    Returns:
        A tuple of ``(context_string, citations)`` where *context_string*
        is meant to be injected into the LLM prompt and each citation
        carries structured metadata for the API response.

        Example context line::

            [1] Vitamin D aids calcium absorption (Source: nutrition_guide.pdf)
    """
    if not chunks:
        return "", []

    lines: list[str] = []
    citations: list[Citation] = []

    for idx, chunk in enumerate(chunks, start=1):
        text = chunk["text"]
        meta = chunk["metadata"]
        title = meta.get("title", "unknown")

        lines.append(f"[{idx}] {text} (Source: {title})")
        citations.append(
            Citation(
                doc_id=meta.get("doc_id", ""),
                chunk_id=meta.get("chunk_id", ""),
                title=title,
                source=meta.get("source", ""),
                snippet=text[:200],
                full_text=text,
            ),
        )

    return "\n\n".join(lines), citations
