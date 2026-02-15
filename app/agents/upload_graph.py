"""LangGraph upload/ingest agent.

A linear pipeline that extracts text from an uploaded document, chunks it,
classifies it (general / personal / ambiguous), and ingests the chunks
into the appropriate vector-store collection.

If the classification is **AMBIGUOUS** the graph stops early and returns
``classification=AMBIGUOUS`` so the API layer can ask the user to choose
before calling a second ingestion endpoint.

Graph structure::

    START ─> extract ─> chunk ─> classify ─┬─ GENERAL  ─> ingest ─> END
                                           ├─ PERSONAL ─> ingest ─> END
                                           └─ AMBIGUOUS ──────────> END
"""

from __future__ import annotations

import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.router import classify_document
from app.models.schemas import DocClassification
from app.services.document import extract_text, process_document
from app.services.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)

# ── State schema ─────────────────────────────────────────────────────

_vectorstore: VectorStoreService | None = None


def _get_vectorstore() -> VectorStoreService:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStoreService()
    return _vectorstore


class UploadState(TypedDict, total=False):
    user_id: str
    file_bytes: bytes
    filename: str
    doc_id: str
    extracted_text: str
    chunks: list[dict]
    classification: DocClassification | None
    final_classification: str | None
    chunks_created: int


# ── Node functions ───────────────────────────────────────────────────


def extract_node(state: UploadState) -> dict:
    """Extract raw text from the uploaded file."""
    text = extract_text(state["file_bytes"], state["filename"])
    logger.info("Extracted %d chars from %s", len(text), state["filename"])
    return {"extracted_text": text}


def chunk_node(state: UploadState) -> dict:
    """Chunk the document and produce metadata-annotated dicts."""
    chunks = process_document(
        state["file_bytes"],
        state["filename"],
        state["doc_id"],
    )
    return {"chunks": chunks}


def classify_node(state: UploadState) -> dict:
    """Classify the document as general, personal, or ambiguous."""
    classification = classify_document(
        state.get("extracted_text", ""),
        state["filename"],
    )
    logger.info("Document %s classified as %s", state["filename"], classification.value)
    return {"classification": classification}


def ingest_node(state: UploadState) -> dict:
    """Write chunks into the correct vector-store collection.

    Uses ``final_classification`` if the user overrode an ambiguous result,
    otherwise falls back to the LLM-assigned ``classification``.
    """
    classification = state.get("final_classification") or state.get("classification")
    chunks = state.get("chunks", [])
    store = _get_vectorstore()

    if classification == DocClassification.PERSONAL or classification == "personal":
        collection_name = f"personal_{state['user_id']}"
        print(f"[INGEST] Storing {len(chunks)} chunks in collection={collection_name} (personal)")
        store.add_documents(chunks, collection_type="personal", user_id=state["user_id"])
    else:
        print(f"[INGEST] Storing {len(chunks)} chunks in collection=general_kb (general)")
        store.add_documents(chunks, collection_type="general")

    logger.info("Ingested %d chunks as %s", len(chunks), classification)
    return {"chunks_created": len(chunks)}


# ── Routing ──────────────────────────────────────────────────────────


def _route_by_classification(state: UploadState) -> str:
    classification = state.get("classification")
    if classification is DocClassification.AMBIGUOUS:
        return "end"
    return "ingest"


# ── Graph construction ───────────────────────────────────────────────


def build_upload_graph() -> CompiledStateGraph:
    """Construct and compile the upload/ingest graph."""
    graph = StateGraph(UploadState)

    graph.add_node("extract", extract_node)
    graph.add_node("chunk", chunk_node)
    graph.add_node("classify", classify_node)
    graph.add_node("ingest", ingest_node)

    graph.add_edge(START, "extract")
    graph.add_edge("extract", "chunk")
    graph.add_edge("chunk", "classify")
    graph.add_conditional_edges(
        "classify",
        _route_by_classification,
        {
            "ingest": "ingest",
            "end": END,
        },
    )
    graph.add_edge("ingest", END)

    return graph.compile()


upload_graph = build_upload_graph()
