"""Tests for intent and document routing.

All LLM calls are mocked — no real API key is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import DocClassification, IntentType


# ── Helpers ──────────────────────────────────────────────────────────

def _mock_llm_response(label: str):
    """Return a patched ``_get_llm`` whose ``.invoke()`` returns *label*."""
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = label
    mock_llm.invoke.return_value = mock_response
    return patch("app.agents.router._get_llm", return_value=mock_llm)


# ── classify_intent ──────────────────────────────────────────────────

class TestClassifyIntent:
    def test_health_general(self) -> None:
        with _mock_llm_response("health_general"):
            from app.agents.router import classify_intent

            result = classify_intent("What vitamin D dosage is recommended?")
        assert result is IntentType.HEALTH_GENERAL

    def test_health_personal(self) -> None:
        with _mock_llm_response("health_personal"):
            from app.agents.router import classify_intent

            result = classify_intent(
                "Based on my lab results, what should I take?",
                has_personal_data=True,
            )
        assert result is IntentType.HEALTH_PERSONAL

    def test_purchase(self) -> None:
        with _mock_llm_response("purchase"):
            from app.agents.router import classify_intent

            result = classify_intent("Where can I buy vitamin C?")
        assert result is IntentType.PURCHASE

    def test_out_of_scope(self) -> None:
        with _mock_llm_response("out_of_scope"):
            from app.agents.router import classify_intent

            result = classify_intent("What's the weather today?")
        assert result is IntentType.OUT_OF_SCOPE

    def test_personal_intent_preserved_without_data(self) -> None:
        """Intent reflects user request even when no personal data exists.

        Data availability is handled downstream by knowledge-gap detection.
        """
        with _mock_llm_response("health_personal"):
            from app.agents.router import classify_intent

            result = classify_intent(
                "Based on my lab results, what should I take?",
                has_personal_data=False,
            )
        assert result is IntentType.HEALTH_PERSONAL

    def test_llm_failure_defaults_to_general(self) -> None:
        with patch("app.agents.router._get_llm", side_effect=RuntimeError("boom")):
            from app.agents.router import classify_intent

            result = classify_intent("anything")
        assert result is IntentType.HEALTH_GENERAL


# ── classify_document ────────────────────────────────────────────────

class TestClassifyDocument:
    def test_personal_document(self) -> None:
        with _mock_llm_response("personal"):
            from app.agents.router import classify_document

            result = classify_document(
                "Patient: Jane Doe\nHemoglobin: 12.1 g/dL",
                "Lab Report - Jan 2025.pdf",
            )
        assert result is DocClassification.PERSONAL

    def test_general_document(self) -> None:
        with _mock_llm_response("general"):
            from app.agents.router import classify_document

            result = classify_document(
                "A meta-analysis of 25-hydroxyvitamin D supplementation ...",
                "Vitamin D meta-analysis.pdf",
            )
        assert result is DocClassification.GENERAL

    def test_ambiguous_document(self) -> None:
        with _mock_llm_response("ambiguous"):
            from app.agents.router import classify_document

            result = classify_document("Some unclear text", "document.pdf")
        assert result is DocClassification.AMBIGUOUS

    def test_llm_failure_defaults_to_ambiguous(self) -> None:
        with patch("app.agents.router._get_llm", side_effect=RuntimeError("boom")):
            from app.agents.router import classify_document

            result = classify_document("text", "file.pdf")
        assert result is DocClassification.AMBIGUOUS

    def test_truncates_to_1000_chars(self) -> None:
        """Verify only the first 1000 chars are sent to the LLM."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "general"
        mock_llm.invoke.return_value = mock_response

        with patch("app.agents.router._get_llm", return_value=mock_llm):
            from app.agents.router import classify_document

            long_text = "z" * 5000
            classify_document(long_text, "big.pdf")

        # Inspect the user message passed to invoke.
        call_args = mock_llm.invoke.call_args[0][0]
        user_msg = call_args[1].content
        # The text sample portion should be at most 1000 'z' chars.
        assert user_msg.count("z") == 1000
