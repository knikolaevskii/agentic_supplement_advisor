"""End-to-end tests that run the full LangGraph pipelines.

Each scenario invokes a compiled graph with a real (temp-dir) ChromaDB
vector store while mocking only the LLM and Tavily external calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.models.schemas import DocClassification, IntentType, PurchaseLink
from app.services.vectorstore import VectorStoreService


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture()
def store(tmp_path) -> VectorStoreService:
    """VectorStoreService backed by a temporary directory."""
    return VectorStoreService(persist_dir=str(tmp_path / "chroma"))


def _make_chunks(doc_id: str, texts: list[str]) -> list[dict]:
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


def _mock_llm(content: str) -> MagicMock:
    """Create a mock LLM whose ``.invoke()`` returns *content*."""
    llm = MagicMock()
    resp = MagicMock()
    resp.content = content
    llm.invoke.return_value = resp
    return llm


def _mock_llm_multi(*contents: str) -> MagicMock:
    """Mock LLM returning different content on successive .invoke() calls."""
    llm = MagicMock()
    responses = [MagicMock(content=c) for c in contents]
    llm.invoke.side_effect = responses
    return llm


# ── 1. Health-general question with citation ──────────────────────────


class TestHealthGeneralFlow:
    def test_answer_with_citation(self, store: VectorStoreService) -> None:
        # Seed general KB.
        chunks = _make_chunks("vitd", [
            "Vitamin D supports calcium absorption and bone health",
            "The recommended daily intake of Vitamin D is 600-800 IU",
        ])
        store.add_documents(chunks, collection_type="general")

        router_llm = _mock_llm("health_general")
        # Pipeline: plan → filter → gap_detect → generate
        nodes_llm = _mock_llm_multi(
            '[{"query": "vitamin D benefits and health effects", '
            '"collection": "general", "reason": "general vitamin D info"}]',  # plan
            "KEEP\nKEEP",                               # filter
            "SUFFICIENT",                               # gap detection
            "Vitamin D supports calcium absorption [1]. "
            "The recommended intake is 600-800 IU [2].",  # generate
        )

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.nodes._get_llm", return_value=nodes_llm),
            patch("app.agents.nodes._get_vectorstore", return_value=store),
        ):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "user_id": "test_user",
                "message": "What is vitamin D good for?",
                "conversation_history": [],
                "has_personal_data": False,
            })

        assert result["intent"] is IntentType.HEALTH_GENERAL
        assert "Vitamin D" in result["response"]
        assert len(result["citations"]) > 0
        assert result["citations"][0].doc_id == "vitd"


# ── 2. Upload personal document ──────────────────────────────────────


class TestUploadPersonalFlow:
    def test_personal_classification_and_ingest(
        self, store: VectorStoreService,
    ) -> None:
        router_llm = _mock_llm("personal")
        pdf_text = "Patient: Jane Doe\nHemoglobin: 12.1 g/dL\nIron: 45 mcg/dL"
        file_bytes = pdf_text.encode("utf-8")

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.upload_graph._get_vectorstore", return_value=store),
        ):
            from app.agents.upload_graph import build_upload_graph

            graph = build_upload_graph()
            result = graph.invoke({
                "user_id": "jane",
                "file_bytes": file_bytes,
                "filename": "lab_report.txt",
                "doc_id": "lab001",
            })

        assert result["classification"] is DocClassification.PERSONAL
        assert result["chunks_created"] > 0

        # Verify chunks were actually persisted in the personal collection.
        docs = store.list_user_documents("jane")
        assert any(d["doc_id"] == "lab001" for d in docs)


# ── 3. Health-personal question referencing uploaded doc ──────────────


class TestHealthPersonalFlow:
    def test_answer_from_personal_kb(self, store: VectorStoreService) -> None:
        # Seed personal KB for user "jane".
        personal_chunks = _make_chunks("lab001", [
            "Patient hemoglobin: 12.1 g/dL — slightly below normal",
            "Iron level: 45 mcg/dL — low, consider supplementation",
        ])
        store.add_documents(
            personal_chunks, collection_type="personal", user_id="jane",
        )

        router_llm = _mock_llm("health_personal")
        # Pipeline: plan → filter → gap_detect → generate
        nodes_llm = _mock_llm_multi(
            '[{"query": "iron level status from lab results", '
            '"collection": "personal", "reason": "personal lab data"}]',  # plan
            "KEEP\nKEEP",                               # filter
            "SUFFICIENT",                               # gap detection
            "Based on your lab results, your iron level is low [1]. "
            "You may want to discuss iron supplementation with your doctor.",
        )

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.nodes._get_llm", return_value=nodes_llm),
            patch("app.agents.nodes._get_vectorstore", return_value=store),
        ):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "user_id": "jane",
                "message": "Is my iron level okay?",
                "conversation_history": [],
                "has_personal_data": True,
            })

        assert result["intent"] is IntentType.HEALTH_PERSONAL
        assert "iron" in result["response"].lower()


# ── 4. Purchase question with Tavily links ───────────────────────────


class TestPurchaseFlow:
    def test_purchase_links_returned(self, store: VectorStoreService) -> None:
        router_llm = _mock_llm("purchase")
        # Pipeline: rephrase_purchase → filter_links → generate_purchase
        nodes_llm = _mock_llm_multi(
            "buy vitamin C supplements online",         # rephrase_purchase
            "KEEP",                                     # filter_links
            "Here are some options for buying Vitamin C:\n"
            "1. [Vitamin C 1000mg](https://example.com/vitc) — $12.99",
        )

        mock_tavily = MagicMock()
        mock_tavily.search_purchase_links.return_value = [
            PurchaseLink(
                title="Vitamin C 1000mg",
                url="https://example.com/vitc",
                price="$12.99",
            ),
        ]

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.nodes._get_llm", return_value=nodes_llm),
            patch("app.agents.nodes._get_tavily", return_value=mock_tavily),
        ):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "user_id": "test_user",
                "message": "Where can I buy vitamin C?",
                "conversation_history": [],
                "has_personal_data": False,
            })

        assert result["intent"] is IntentType.PURCHASE
        assert len(result["purchase_links"]) == 1
        assert result["purchase_links"][0].url == "https://example.com/vitc"


# ── 5. Out-of-scope question → refusal ───────────────────────────────


class TestOutOfScopeFlow:
    def test_refusal_response(self) -> None:
        router_llm = _mock_llm("out_of_scope")

        with patch("app.agents.router._get_llm", return_value=router_llm):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "user_id": "test_user",
                "message": "What is the weather today?",
                "conversation_history": [],
                "has_personal_data": False,
            })

        assert result["intent"] is IntentType.OUT_OF_SCOPE
        assert "vitamin" in result["response"].lower() or "supplement" in result["response"].lower()
        assert result["citations"] == []


# ── 6. Ambiguous upload stops early ───────────────────────────────────


class TestAmbiguousUploadFlow:
    def test_ambiguous_returns_without_ingesting(
        self, store: VectorStoreService,
    ) -> None:
        router_llm = _mock_llm("ambiguous")
        file_bytes = b"Some unclear document content"

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.upload_graph._get_vectorstore", return_value=store),
        ):
            from app.agents.upload_graph import build_upload_graph

            graph = build_upload_graph()
            result = graph.invoke({
                "user_id": "test_user",
                "file_bytes": file_bytes,
                "filename": "mystery.txt",
                "doc_id": "amb001",
            })

        assert result["classification"] is DocClassification.AMBIGUOUS
        # No chunks should have been ingested.
        assert result.get("chunks_created") is None or result["chunks_created"] == 0
        # But chunks should still be available for the UI to send to /upload/confirm.
        assert len(result.get("chunks", [])) > 0


# ── 7. Messages-based flow (new persistence path) ────────────────────


class TestMessagesBasedFlow:
    def test_messages_appends_ai_message(self, store: VectorStoreService) -> None:
        """When invoked with `messages`, the graph appends an AIMessage."""
        chunks = _make_chunks("vitd", [
            "Vitamin D supports calcium absorption and bone health",
        ])
        store.add_documents(chunks, collection_type="general")

        router_llm = _mock_llm("health_general")
        # Pipeline: plan → filter → gap_detect → generate
        nodes_llm = _mock_llm_multi(
            '[{"query": "vitamin D health benefits", '
            '"collection": "general", "reason": "general info"}]',  # plan
            "KEEP",                                     # filter
            "SUFFICIENT",                               # gap detection
            "Vitamin D is great for bone health [1].",  # generate
        )

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.nodes._get_llm", return_value=nodes_llm),
            patch("app.agents.nodes._get_vectorstore", return_value=store),
        ):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "messages": [HumanMessage(content="Tell me about vitamin D")],
                "user_id": "test_user",
                "has_personal_data": False,
            })

        assert result["intent"] is IntentType.HEALTH_GENERAL
        assert "Vitamin D" in result["response"]
        # Messages list should contain HumanMessage + AIMessage.
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][0], HumanMessage)
        assert isinstance(result["messages"][1], AIMessage)
        assert result["messages"][1].content == result["response"]

    def test_out_of_scope_messages_path(self) -> None:
        """Refuse node also appends AIMessage when using messages path."""
        router_llm = _mock_llm("out_of_scope")

        with patch("app.agents.router._get_llm", return_value=router_llm):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "messages": [HumanMessage(content="What is 2+2?")],
                "user_id": "test_user",
                "has_personal_data": False,
            })

        assert result["intent"] is IntentType.OUT_OF_SCOPE
        assert len(result["messages"]) == 2
        assert isinstance(result["messages"][1], AIMessage)


# ── 8. Checkpointer flow — multi-turn persistence ────────────────────


class TestCheckpointerFlow:
    def test_messages_persist_across_turns(self, store: VectorStoreService) -> None:
        """Using MemorySaver, messages accumulate across invocations."""
        chunks = _make_chunks("vitd", [
            "Vitamin D supports calcium absorption and bone health",
        ])
        store.add_documents(chunks, collection_type="general")

        router_llm = _mock_llm("health_general")
        # Each turn: plan → filter → gap_detect → generate (8 calls for 2 turns)
        nodes_llm = _mock_llm_multi(
            '[{"query": "vitamin D health benefits", '
            '"collection": "general", "reason": "general info"}]',  # turn 1: plan
            "KEEP",                             # turn 1: filter
            "SUFFICIENT",                       # turn 1: gap detection
            "Vitamin D helps with bones.",      # turn 1: generate
            '[{"query": "vitamin D recommended daily dose", '
            '"collection": "general", "reason": "dosage info"}]',   # turn 2: plan
            "KEEP",                             # turn 2: filter
            "SUFFICIENT",                       # turn 2: gap detection
            "Vitamin D helps with bones.",      # turn 2: generate
        )

        checkpointer = MemorySaver()

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.nodes._get_llm", return_value=nodes_llm),
            patch("app.agents.nodes._get_vectorstore", return_value=store),
        ):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": "thread_1"}}

            # Turn 1.
            result1 = graph.invoke(
                {
                    "messages": [HumanMessage(content="What is vitamin D?")],
                    "user_id": "test_user",
                    "has_personal_data": False,
                },
                config=config,
            )
            assert len(result1["messages"]) == 2  # Human + AI

            # Turn 2 — only send the new user message.
            result2 = graph.invoke(
                {
                    "messages": [HumanMessage(content="How much should I take?")],
                    "user_id": "test_user",
                    "has_personal_data": False,
                },
                config=config,
            )
            # Should now have 4 messages: H1, A1, H2, A2.
            assert len(result2["messages"]) == 4
            assert isinstance(result2["messages"][0], HumanMessage)
            assert isinstance(result2["messages"][1], AIMessage)
            assert isinstance(result2["messages"][2], HumanMessage)
            assert isinstance(result2["messages"][3], AIMessage)

        # Verify via get_state.
        snapshot = graph.get_state(config)
        assert len(snapshot.values["messages"]) == 4


# ── 9. Health-combined question (both KBs) ───────────────────────────


class TestHealthCombinedFlow:
    def test_combined_two_phase_retrieval(self, store: VectorStoreService) -> None:
        """HEALTH_COMBINED uses two-phase retrieval: personal first, then targeted general."""
        # Seed general KB.
        general_chunks = _make_chunks("vitd_guide", [
            "Vitamin D recommended intake is 600-800 IU for most adults",
            "Severe deficiency (<20 ng/mL): 5,000-10,000 IU daily for 8-12 weeks",
        ])
        store.add_documents(general_chunks, collection_type="general")

        # Seed personal KB.
        personal_chunks = _make_chunks("lab001", [
            "Patient vitamin D level: 22 ng/mL — LOW",
        ])
        store.add_documents(
            personal_chunks, collection_type="personal", user_id="jane",
        )

        router_llm = _mock_llm("health_combined")
        # Pipeline: plan → execute → analyze_and_replan → execute_targeted → filter → gap_detect → generate
        nodes_llm = _mock_llm_multi(
            '[{"query": "vitamin D lab results level", '
            '"collection": "personal", "reason": "personal vitamin D data"}]',  # plan (personal only)
            '["Vitamin D: 22 ng/mL (LOW)"]',               # analyze: extract findings
            "KEEP\nKEEP",                                   # filter
            "SUFFICIENT",                                   # gap detection
            "Your vitamin D is low at 22 ng/mL [1]. "
            "The recommended intake is 600-800 IU [2].",    # generate
        )

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.nodes._get_llm", return_value=nodes_llm),
            patch("app.agents.nodes._get_vectorstore", return_value=store),
        ):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "user_id": "jane",
                "message": "Based on my lab results, how much vitamin D should I take?",
                "conversation_history": [],
                "has_personal_data": True,
            })

        assert result["intent"] is IntentType.HEALTH_COMBINED
        assert "vitamin D" in result["response"].lower() or "vitamin d" in result["response"].lower()
        assert len(result["citations"]) > 0
        # Phase 2 should have extracted findings.
        assert result.get("personal_findings") == ["Vitamin D: 22 ng/mL (LOW)"]
        # Targeted plan should have created a general query for vitamin D.
        targeted = result.get("targeted_plan", [])
        assert len(targeted) >= 1
        assert targeted[0]["collection"] == "general"


# ── 10. Clarification flow ──────────────────────────────────────────


class TestClarificationFlow:
    def test_health_clarification(self) -> None:
        """Vague health question triggers clarification instead of retrieval."""
        router_llm = _mock_llm("health_general")
        # Plan retrieval detects ambiguity
        nodes_llm = _mock_llm(
            "CLARIFY: Could you tell me which supplement or health topic you're interested in?"
        )

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.nodes._get_llm", return_value=nodes_llm),
        ):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "user_id": "test_user",
                "message": "supplements",
                "conversation_history": [],
                "has_personal_data": False,
            })

        assert result["intent"] is IntentType.HEALTH_GENERAL
        assert "which supplement" in result["response"].lower()
        assert result["citations"] == []

    def test_purchase_clarification(self) -> None:
        """Vague purchase question triggers clarification."""
        router_llm = _mock_llm("purchase")
        nodes_llm = _mock_llm(
            "CLARIFY: Which supplement are you looking to purchase?"
        )

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.nodes._get_llm", return_value=nodes_llm),
        ):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "user_id": "test_user",
                "message": "buy something",
                "conversation_history": [],
                "has_personal_data": False,
            })

        assert result["intent"] is IntentType.PURCHASE
        assert "which supplement" in result["response"].lower()
        assert result["citations"] == []


# ── 11. Filter relevance behavior ───────────────────────────────────


class TestFilterRelevanceFlow:
    def test_filter_drops_irrelevant_chunks(self, store: VectorStoreService) -> None:
        """Filter node drops chunks marked as DROP by the LLM."""
        chunks = _make_chunks("vitd", [
            "Vitamin D supports calcium absorption and bone health",
            "The weather today is sunny and warm",  # irrelevant
            "Recommended daily intake of Vitamin D is 600-800 IU",
        ])
        store.add_documents(chunks, collection_type="general")

        router_llm = _mock_llm("health_general")
        # Pipeline: plan → filter → gap_detect → generate
        # ChromaDB returns chunks in similarity order (unpredictable with
        # real embeddings), so we drop the last chunk regardless of content.
        nodes_llm = _mock_llm_multi(
            '[{"query": "vitamin D benefits and dosage", '
            '"collection": "general", "reason": "general info"}]',  # plan
            "KEEP\nKEEP\nDROP",                # filter: drop last chunk
            "SUFFICIENT",                       # gap detection
            "Vitamin D supports bones [1]. Recommended intake is 600-800 IU [2].",
        )

        with (
            patch("app.agents.router._get_llm", return_value=router_llm),
            patch("app.agents.nodes._get_llm", return_value=nodes_llm),
            patch("app.agents.nodes._get_vectorstore", return_value=store),
        ):
            from app.agents.chat_graph import build_chat_graph

            graph = build_chat_graph()
            result = graph.invoke({
                "user_id": "test_user",
                "message": "What is vitamin D good for?",
                "conversation_history": [],
                "has_personal_data": False,
            })

        assert result["intent"] is IntentType.HEALTH_GENERAL
        # Filter should have kept 2 out of 3 chunks.
        filtered = result.get("filtered_chunks", [])
        assert len(filtered) == 2
        # All 3 were retrieved, but only 2 survived filtering.
        retrieved = result.get("retrieved_chunks", [])
        assert len(retrieved) == 3
