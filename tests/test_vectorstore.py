"""Tests for vectorstore service."""

from __future__ import annotations

import pytest

from app.services.vectorstore import VectorStoreService


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture()
def store(tmp_path) -> VectorStoreService:
    """VectorStoreService backed by a temporary directory."""
    return VectorStoreService(persist_dir=str(tmp_path / "chroma"))


def _make_chunks(doc_id: str, texts: list[str]) -> list[dict]:
    """Build chunk dicts matching the process_document output format."""
    return [
        {
            "text": text,
            "metadata": {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_chunk_{i}",
                "title": f"{doc_id}.txt",
                "source": f"{doc_id}.txt",
                "page": 1,
            },
        }
        for i, text in enumerate(texts)
    ]


# ── General collection ───────────────────────────────────────────────

class TestGeneralCollection:
    def test_add_and_search(self, store: VectorStoreService) -> None:
        chunks = _make_chunks("doc1", [
            "Vitamin D helps with calcium absorption",
            "Omega-3 fatty acids reduce inflammation",
        ])
        store.add_documents(chunks, collection_type="general")

        results = store.search("calcium and bones", collection_type="general", k=2)
        assert len(results) == 2
        assert all("text" in r and "metadata" in r and "score" in r for r in results)
        # The vitamin D chunk should rank first for a calcium query.
        assert "Vitamin D" in results[0]["text"]

    def test_search_empty_collection(self, store: VectorStoreService) -> None:
        results = store.search("anything", collection_type="general")
        assert results == []

    def test_add_empty_chunks_is_noop(self, store: VectorStoreService) -> None:
        store.add_documents([], collection_type="general")
        results = store.search("anything", collection_type="general")
        assert results == []


# ── Personal collection ──────────────────────────────────────────────

class TestPersonalCollection:
    def test_add_and_search(self, store: VectorStoreService) -> None:
        chunks = _make_chunks("blood_test", [
            "Patient hemoglobin level 14.2 g/dL",
            "Vitamin B12 level 450 pg/mL within normal range",
        ])
        store.add_documents(chunks, collection_type="personal", user_id="alice")

        results = store.search(
            "hemoglobin", collection_type="personal", user_id="alice", k=2,
        )
        assert len(results) == 2
        assert "hemoglobin" in results[0]["text"].lower()

    def test_personal_requires_user_id(self, store: VectorStoreService) -> None:
        chunks = _make_chunks("doc", ["some text"])
        with pytest.raises(ValueError, match="user_id is required"):
            store.add_documents(chunks, collection_type="personal")


# ── Cross-user isolation ─────────────────────────────────────────────

class TestCrossUserIsolation:
    def test_no_cross_user_leakage(self, store: VectorStoreService) -> None:
        alice_chunks = _make_chunks("alice_doc", [
            "Alice's private blood test results show low iron",
        ])
        bob_chunks = _make_chunks("bob_doc", [
            "Bob's cholesterol panel looks normal",
        ])
        store.add_documents(alice_chunks, collection_type="personal", user_id="alice")
        store.add_documents(bob_chunks, collection_type="personal", user_id="bob")

        # Alice should only see her own data.
        alice_results = store.search(
            "blood test", collection_type="personal", user_id="alice",
        )
        assert all("alice" in r["metadata"]["doc_id"] for r in alice_results)

        # Bob should only see his own data.
        bob_results = store.search(
            "blood test", collection_type="personal", user_id="bob",
        )
        assert all("bob" in r["metadata"]["doc_id"] for r in bob_results)

    def test_personal_does_not_leak_into_general(
        self, store: VectorStoreService,
    ) -> None:
        chunks = _make_chunks("private", ["Sensitive personal health data"])
        store.add_documents(chunks, collection_type="personal", user_id="alice")

        results = store.search("health data", collection_type="general")
        assert results == []


# ── search_both ──────────────────────────────────────────────────────

class TestSearchBoth:
    def test_merges_general_and_personal(self, store: VectorStoreService) -> None:
        gen = _make_chunks("gen", ["Magnesium supports muscle function"])
        per = _make_chunks("per", ["Patient magnesium level is low"])
        store.add_documents(gen, collection_type="general")
        store.add_documents(per, collection_type="personal", user_id="alice")

        results = store.search_both("magnesium", user_id="alice", k=2)
        assert len(results) == 2
        sources = {r["metadata"]["doc_id"] for r in results}
        assert sources == {"gen", "per"}

    def test_search_both_respects_k(self, store: VectorStoreService) -> None:
        gen = _make_chunks("g", ["Zinc boosts immunity", "Iron carries oxygen"])
        per = _make_chunks("p", ["Low zinc detected", "Ferritin normal"])
        store.add_documents(gen, collection_type="general")
        store.add_documents(per, collection_type="personal", user_id="u1")

        results = store.search_both("zinc immunity", user_id="u1", k=2)
        assert len(results) == 2


# ── list_user_documents & delete_document ────────────────────────────

class TestDocumentManagement:
    def test_list_user_documents(self, store: VectorStoreService) -> None:
        c1 = _make_chunks("doc_a", ["chunk one", "chunk two"])
        c2 = _make_chunks("doc_b", ["chunk three"])
        store.add_documents(c1, collection_type="personal", user_id="alice")
        store.add_documents(c2, collection_type="personal", user_id="alice")

        docs = store.list_user_documents("alice")
        doc_ids = {d["doc_id"] for d in docs}
        assert doc_ids == {"doc_a", "doc_b"}

    def test_list_empty(self, store: VectorStoreService) -> None:
        assert store.list_user_documents("nobody") == []

    def test_delete_document(self, store: VectorStoreService) -> None:
        chunks = _make_chunks("to_delete", ["will be removed", "also removed"])
        store.add_documents(chunks, collection_type="personal", user_id="alice")

        store.delete_document("to_delete", collection_type="personal", user_id="alice")

        results = store.search(
            "removed", collection_type="personal", user_id="alice",
        )
        assert results == []

    def test_delete_from_empty_collection(self, store: VectorStoreService) -> None:
        # Should not raise.
        store.delete_document("nope", collection_type="general")
