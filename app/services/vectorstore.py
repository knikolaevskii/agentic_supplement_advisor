"""ChromaDB wrapper for general and personal collections.

Provides two logical knowledge bases:
- **General KB**: a single shared collection (``general_kb`` by default).
- **Personal KB**: one collection per user, named ``personal_{user_id}``.

Personal data is strictly isolated — each user's documents live in their
own collection and are never mixed into the general store.
"""

from __future__ import annotations

import logging

import chromadb

from app.config import settings

logger = logging.getLogger(__name__)


def _join_chunks_without_overlap(chunks: list[str]) -> str:
    """Join chunk texts, removing overlapping regions between consecutive chunks."""
    if not chunks:
        return ""
    result = chunks[0]
    for i in range(1, len(chunks)):
        max_overlap = min(len(result), len(chunks[i]), 200)
        overlap = 0
        for j in range(1, max_overlap + 1):
            if result[-j:] == chunks[i][:j]:
                overlap = j
        result += chunks[i][overlap:]
    return result


class VectorStoreService:
    """Thin wrapper around ChromaDB that enforces general / personal separation."""

    def __init__(self, persist_dir: str | None = None) -> None:
        path = persist_dir or settings.chroma_persist_dir
        self._client = chromadb.PersistentClient(path=path)
        self._general_name = settings.general_collection_name

    # ── internal helpers ─────────────────────────────────────────────

    def _collection_name(
        self,
        collection_type: str,
        user_id: str | None = None,
    ) -> str:
        if collection_type == "personal":
            if not user_id:
                raise ValueError("user_id is required for personal collections")
            return f"personal_{user_id}"
        return self._general_name

    def _get_collection(
        self,
        collection_type: str,
        user_id: str | None = None,
    ) -> chromadb.Collection:
        name = self._collection_name(collection_type, user_id)
        return self._client.get_or_create_collection(name=name)

    # ── public API ───────────────────────────────────────────────────

    def add_documents(
        self,
        chunks: list[dict],
        collection_type: str,
        user_id: str | None = None,
    ) -> None:
        """Insert document chunks into the appropriate collection.

        Args:
            chunks: Each dict must contain ``text`` and ``metadata``
                    (with keys doc_id, chunk_id, title, source, page).
            collection_type: ``"general"`` or ``"personal"``.
            user_id: Required when *collection_type* is ``"personal"``.
        """
        if not chunks:
            return

        collection = self._get_collection(collection_type, user_id)
        collection.add(
            ids=[c["metadata"]["chunk_id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        logger.info(
            "Added %d chunks to %s collection", len(chunks), collection.name,
        )

    def search(
        self,
        query: str,
        collection_type: str,
        user_id: str | None = None,
        k: int = 5,
    ) -> list[dict]:
        """Semantic search within a single collection.

        Returns:
            List of ``{text, metadata, score}`` dicts sorted by relevance
            (lower distance = more relevant).  Returns ``[]`` when the
            collection is empty.
        """
        collection = self._get_collection(collection_type, user_id)

        if collection.count() == 0:
            return []

        n = min(k, collection.count())
        results = collection.query(query_texts=[query], n_results=n)

        out: list[dict] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"text": doc, "metadata": meta, "score": dist})
        return out

    def search_both(
        self,
        query: str,
        user_id: str,
        k: int = 3,
    ) -> list[dict]:
        """Search general + personal collections, merge and sort by relevance.

        Returns the top *k* results across both stores.
        """
        general = self.search(query, "general", k=k)
        personal = self.search(query, "personal", user_id=user_id, k=k)
        merged = general + personal
        merged.sort(key=lambda r: r["score"])
        return merged[:k]

    def list_user_documents(self, user_id: str) -> list[dict]:
        """Return unique documents in a user's personal collection.

        Returns:
            List of ``{doc_id, title, source}`` dicts (deduplicated).
        """
        collection = self._get_collection("personal", user_id)

        if collection.count() == 0:
            return []

        all_meta = collection.get(include=["metadatas"])["metadatas"]

        seen: set[str] = set()
        docs: list[dict] = []
        for meta in all_meta:
            doc_id = meta["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                docs.append({
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                })
        return docs

    def list_general_documents(self) -> list[dict]:
        """Return unique documents in the general knowledge base.

        Returns:
            List of ``{doc_id, title, source}`` dicts (deduplicated).
        """
        collection = self._get_collection("general")

        if collection.count() == 0:
            return []

        all_meta = collection.get(include=["metadatas"])["metadatas"]

        seen: set[str] = set()
        docs: list[dict] = []
        for meta in all_meta:
            doc_id = meta["doc_id"]
            if doc_id not in seen:
                seen.add(doc_id)
                docs.append({
                    "doc_id": doc_id,
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                })
        return docs

    def get_document_preview(
        self,
        doc_id: str,
        collection_type: str,
        user_id: str | None = None,
        max_chunks: int = 5,
    ) -> str:
        """Return a text preview by joining the first *max_chunks* chunks."""
        collection = self._get_collection(collection_type, user_id)

        if collection.count() == 0:
            return ""

        results = collection.get(
            where={"doc_id": doc_id},
            include=["documents"],
        )
        texts = results.get("documents", [])[:max_chunks]
        return _join_chunks_without_overlap(texts)

    def clear_collection(
        self,
        collection_type: str,
        user_id: str | None = None,
    ) -> int:
        """Delete and recreate a collection. Returns the chunk count before clearing."""
        name = self._collection_name(collection_type, user_id)
        try:
            col = self._client.get_collection(name=name)
            count = col.count()
        except Exception:
            return 0
        self._client.delete_collection(name=name)
        self._client.get_or_create_collection(name=name)
        return count

    def list_all_personal_collections(self) -> list[str]:
        """Return user_ids for all personal collections."""
        prefix = "personal_"
        return [
            c.name[len(prefix):]
            for c in self._client.list_collections()
            if c.name.startswith(prefix)
        ]

    def delete_document(
        self,
        doc_id: str,
        collection_type: str,
        user_id: str | None = None,
    ) -> None:
        """Delete all chunks belonging to *doc_id* from a collection."""
        collection = self._get_collection(collection_type, user_id)

        if collection.count() == 0:
            return

        # Fetch chunk IDs that belong to this document.
        results = collection.get(
            where={"doc_id": doc_id},
            include=[],
        )
        if results["ids"]:
            collection.delete(ids=results["ids"])
