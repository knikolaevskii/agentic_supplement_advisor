"""Intent and document routing logic.

Uses lightweight LLM calls (gpt-4o-mini) to classify user messages and
uploaded documents so the rest of the pipeline knows which agent to invoke.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.models.schemas import DocClassification, IntentType

logger = logging.getLogger(__name__)

# ── Prompts ──────────────────────────────────────────────────────────

_INTENT_SYSTEM_PROMPT = """\
You are a classifier for a vitamin and supplement advisor chatbot.
You will receive the conversation history (if any) followed by the
latest user message. Classify the LATEST message into EXACTLY one of:
  health_general, health_personal, health_combined, purchase, out_of_scope

Definitions:
- health_general: general vitamin/supplement/nutrition questions that do NOT reference the user's personal data
- health_personal: questions about the user's OWN health data, lab results, blood work, personal profile,
  prescriptions, medications, doctor notes, diagnoses, or health conditions
- health_combined: questions that need BOTH the user's personal records AND general supplement knowledge
- purchase: questions about buying, ordering, pricing, or where to find supplements
- out_of_scope: the question has absolutely NO connection to health, vitamins, supplements, nutrition,
  medications, lab results, or personal health data. Examples: weather, travel, coding, sports.
  If the question is about medications, prescriptions, doctor recommendations, or health conditions,
  these ARE in scope because they relate to supplement recommendations and interactions.

Key rules:
- If the user says "my", "my results", "my labs", "my levels", "my blood work",
  "my records" → classify as health_personal (NOT health_general)
- If the user references "my" personal health data — prescriptions, medications,
  doctor notes, health records, diagnoses, conditions — classify as health_personal.
  Prescriptions and medications are directly relevant to supplement advice because
  of drug-supplement interactions.
- If the user asks about their personal data AND also wants general supplement
  advice or recommendations → classify as health_combined
- Words like "buy", "purchase", "order", "price", "where to get", "where to find" → purchase
- Use conversation history to resolve ambiguous follow-ups (e.g. "where can I
  get some?" after discussing vitamin D → purchase)

Examples:
- "What are the benefits of Vitamin D?" → health_general
- "What vitamins help with energy?" → health_general
- "Recommended dosage for omega-3?" → health_general
- "What do my lab results show?" → health_personal
- "Based on my blood work, am I deficient in anything?" → health_personal
- "What are my vitamin levels?" → health_personal
- "What are my doctor's prescriptions?" → health_personal
- "What medications am I taking?" → health_personal
- "Given my low Vitamin D, what supplements and foods can help?" → health_combined
- "Based on my results, what dosage of B12 do you recommend?" → health_combined
- "My iron is low, what are the best iron supplements?" → health_combined
- "Do any of my medications interact with supplements?" → health_combined
- "What supplements should I avoid given my prescriptions?" → health_combined
- "My doctor prescribed metformin, does it affect my vitamins?" → health_combined
- "Where can I buy Vitamin C?" → purchase
- "I'd like to order some Omega-3, where can I find it?" → purchase
- "What's the best price for magnesium supplements?" → purchase
- "What's the weather?" → out_of_scope
- "Help me write code" → out_of_scope
- "What's the capital of France?" → out_of_scope
- "Book me a flight" → out_of_scope

Respond with the label only. No explanation."""

_DOC_SYSTEM_PROMPT = """\
You are a document classifier for a health supplement advisor.
Given the filename and a text sample from a document, respond with EXACTLY one of:
  general, personal, ambiguous

Rules:
- general: medical literature, research papers, supplement guides, general health info
  Examples: meta-analyses, review articles, supplement fact sheets
- personal: lab results, personal health records, prescriptions, blood work, patient reports
  Examples: files named "Lab Report", "Blood Panel", text with patient-specific values
- ambiguous: cannot confidently determine from the available information

Respond with the label only. No explanation."""


# ── LLM helper ───────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )


# ── Public API ───────────────────────────────────────────────────────

def classify_intent(
    message: str,
    has_personal_data: bool = False,
    conversation_history: list[dict] | None = None,
) -> IntentType:
    """Classify a user message into an :class:`IntentType`.

    Includes recent *conversation_history* so the LLM can resolve
    ambiguous follow-ups (e.g. "yes, buy that" after a vitamin D answer).

    The returned intent always reflects what the user *asked for*, not
    what data is available.  Data-availability handling (e.g. the user
    asks about "my labs" but has no uploads) is managed downstream by the
    knowledge-gap detection in ``filter_relevance_node``.

    Falls back to ``HEALTH_GENERAL`` on any LLM error.
    """
    # Build context: last few turns + current message.
    history = conversation_history or []
    recent = history[-6:]  # keep token cost low

    context_parts: list[str] = []
    for turn in recent:
        role = turn.get("role", "user")
        context_parts.append(f"{role}: {turn.get('content', '')}")
    context_parts.append(f"user: {message}")
    user_content = "\n".join(context_parts)

    print(f"[CLASSIFY_INTENT] has_personal_data={has_personal_data}")
    print(f"[CLASSIFY_INTENT] prompt_to_llm:\n  system: {_INTENT_SYSTEM_PROMPT[:120]}...")
    print(f"  user: {user_content}")

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_INTENT_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        label = response.content.strip().lower()
        print(f"[CLASSIFY_INTENT] raw_llm_response={response.content!r} -> label={label!r}")
        intent = IntentType(label)
    except Exception:
        logger.exception("Intent classification failed, defaulting to HEALTH_GENERAL")
        return IntentType.HEALTH_GENERAL

    return intent


def classify_document(text_sample: str, filename: str) -> DocClassification:
    """Classify an uploaded document as general, personal, or ambiguous.

    Uses the first 1000 characters of the document text together with the
    filename to make the determination.

    Falls back to ``AMBIGUOUS`` on any LLM error.
    """
    snippet = text_sample[:1000]
    user_content = f"Filename: {filename}\n\nText sample:\n{snippet}"

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_DOC_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        label = response.content.strip().lower()
        return DocClassification(label)
    except Exception:
        logger.exception(
            "Document classification failed, defaulting to AMBIGUOUS"
        )
        return DocClassification.AMBIGUOUS
