"""Individual node functions for LangGraph agents.

Each function receives the full graph state dict and returns a partial
update dict that LangGraph merges back into state.
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import add_messages

from app.agents.router import classify_intent
from app.config import settings
from app.models.schemas import Citation, IntentType, PurchaseLink
from app.services.tavily_client import TavilyService
from app.services.vectorstore import VectorStoreService
from app.utils.citations import format_citations

logger = logging.getLogger(__name__)


# ── State schema ─────────────────────────────────────────────────────

class ChatState(TypedDict, total=False):
    messages: Annotated[list, add_messages]  # LangGraph-managed message list
    user_id: str
    message: str                       # legacy — used by tests / LangGraph Studio
    conversation_history: list[dict]   # legacy
    intent: IntentType | None
    retrieved_chunks: list[dict]
    purchase_links: list[PurchaseLink]
    response: str
    citations: list[Citation]
    has_personal_data: bool
    # ── New fields for rephrase / filter / clarify pipeline ──
    rephrased_query: str | None
    filtered_chunks: list[dict]
    filtered_links: list[dict]
    needs_clarification: bool
    clarification_question: str | None
    user_location: str | None
    knowledge_gap: str | None


# ── Backward-compat helpers ──────────────────────────────────────────

def _current_message(state: ChatState) -> str:
    """Return the latest user message from either messages or legacy field."""
    msgs = state.get("messages")
    if msgs:
        return msgs[-1].content
    return state.get("message", "")


def _history_as_dicts(state: ChatState) -> list[dict]:
    """Return prior conversation turns as [{role, content}] for the classifier."""
    msgs = state.get("messages")
    if msgs and len(msgs) > 1:
        return [
            {"role": "assistant" if isinstance(m, AIMessage) else "user", "content": m.content}
            for m in msgs[:-1]
        ]
    return state.get("conversation_history", [])


# ── Shared resources (lazily initialised) ────────────────────────────

_vectorstore: VectorStoreService | None = None
_tavily: TavilyService | None = None


def _get_vectorstore() -> VectorStoreService:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = VectorStoreService()
    return _vectorstore


def _get_tavily() -> TavilyService:
    global _tavily
    if _tavily is None:
        _tavily = TavilyService()
    return _tavily


def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=0.3,
        api_key=settings.openai_api_key,
    )


# ── Prompts ──────────────────────────────────────────────────────────

_REPHRASE_SYSTEM_PROMPT = """\
You are a query optimizer for a vitamin and supplement knowledge base.
Rewrite the user's message into a standalone, search-optimized query.

The user's question has been classified as: {intent}
The user has personal documents uploaded: {has_personal_data}

Rules:
- Resolve pronouns and references using the conversation history.
- Expand abbreviations (e.g. "vit D" → "vitamin D").
- Keep the query concise (1-2 sentences).

Intent-specific rules:
- For health_personal questions: the user is asking about their own uploaded
  health data. Do NOT ask for clarification about what they mean by "my results"
  or "my labs". Rephrase the query to search their personal document collection
  effectively.
  Examples:
    "what are my lab results?" → "lab results vitamin levels blood test values"
    "what did my doctor recommend?" → "doctor recommendations treatment plan"
    "what are my dietary restrictions?" → "dietary restrictions allergies diet"
- For health_combined questions: rephrase to cover both personal data retrieval
  and general knowledge lookup.
- Only use CLARIFY for genuinely vague messages like "help" or "supplements"
  with no context at all. Questions that reference "my", "mine", or personal
  data should NEVER trigger clarification if intent is health_personal or
  health_combined.
- If the message is too vague to search (e.g. just "supplements" or "help"),
  respond with EXACTLY: CLARIFY: followed by a clarifying question.
  Example: CLARIFY: Could you tell me which supplement or health topic you're interested in?

Respond with the rewritten query only, or CLARIFY: followed by a question."""

_REPHRASE_PURCHASE_SYSTEM_PROMPT = """\
You are a query optimizer for a supplement product search engine.
Rewrite the user's message into a standalone product search query.

Rules:
- Resolve pronouns and references using the conversation history.
- Include the product/supplement name explicitly.
- Add "buy" or "purchase" context if not already present.
- Check conversation history for any mention of the user's location,
  city, or country. If found, include it in the search query.
- If the user's location is provided as context below, always include it.
- When the user provides a location, always include BOTH the city AND
  the country in the search query to avoid ambiguity. If the user only
  says a city name, infer the most likely country:
    Amsterdam → Amsterdam, Netherlands
    Paris → Paris, France
    London → London, UK
    Dublin → Dublin, Ireland
    Portland → Portland, USA
  Always format location as "City, Country" in the search query.
  Example: "buy Vitamin C supplements Amsterdam, Netherlands"
- If the user already specified a country (e.g., "Amsterdam, Netherlands"
  or "I'm in the Netherlands"), use it as-is — do not change it.
- If no location is known and the product is clear, respond with EXACTLY:
  LOCATION: followed by the rewritten query (without location).
  This signals that we should ask the user for their location.
  Example: LOCATION: buy vitamin D supplements
- If the message is too vague to search (e.g. just "buy something"),
  respond with EXACTLY: CLARIFY: followed by a clarifying question.
  Example: CLARIFY: Which supplement are you looking to purchase?

Respond with one of:
1. The rewritten query (with location as "City, Country" if known)
2. LOCATION: followed by the query (when location is unknown)
3. CLARIFY: followed by a question (when the product is unclear)"""

_FILTER_RELEVANCE_SYSTEM_PROMPT = """\
You are a strict relevance judge for a vitamin and supplement advisor.
Given a user question and retrieved text chunks, decide if each chunk
directly addresses the SPECIFIC topic in the user's question.

A chunk is relevant ONLY if it directly addresses the specific topic asked about.
Being about supplements or vitamins in general is NOT enough.

Example: if the user asks about Selenium, a chunk about Iron supplementation
is NOT relevant even though both are minerals. If the user asks about Vitamin D
dosage, a chunk about Vitamin C benefits is NOT relevant.

Be strict: when in doubt, mark as NOT relevant.

Respond with EXACTLY one word per chunk: KEEP or DROP (one per line)."""

_GAP_DETECTION_SYSTEM_PROMPT = """\
You are a knowledge gap detector for a vitamin and supplement advisor.
Given the user's question and the available reference chunks, assess whether
the chunks contain enough information to answer the question.

Respond with EXACTLY one of:
- SUFFICIENT — the chunks contain relevant information to answer the question
  (even if not perfect, as long as there's something useful to work with)
- GAP: [description] — the chunks do NOT contain the specific information needed.
  Describe concisely what information is missing.

Be specific about the gap. Examples:
- GAP: information about Vitamin C benefits and recommended dosage
- GAP: the user's personal lab results or blood work
- GAP: information about the user's dietary restrictions
- GAP: clinical data on interactions between iron supplements and calcium

Important: if the chunks contain SOME useful information even if incomplete,
respond with SUFFICIENT — the generation step will handle partial answers.
Only report a GAP when the chunks are truly irrelevant to the question."""

_FILTER_LINKS_SYSTEM_PROMPT = """\
You are a relevance judge for supplement purchase links.
Given a user question and a list of product links, decide if each link
is relevant to what the user wants to buy.

Respond with EXACTLY one word per link: KEEP or DROP (one per line)."""

_RESPONSE_SYSTEM_PROMPT = """\
You are a knowledgeable vitamin and supplement advisor. Use ONLY the
provided reference material to answer the user's question.

Rules:
- Cite sources using bracket notation, e.g. [1], [2].
- Place each citation IMMEDIATELY after the specific claim it supports,
  not at the end of a paragraph or sentence group.
- Be helpful, accurate, and concise.
- NEVER provide medical diagnoses or claim to replace professional advice.
- If the references don't contain enough information, say so honestly.
- When discussing personal lab results, note that interpretation should
  be confirmed with a healthcare provider.

Example of a GOOD answer:
  "Vitamin D supports bone health by aiding calcium absorption [1] and
   plays a role in immune function [2]. Adults typically need 600-800 IU
   daily [1]. Please consult your healthcare provider for personalized dosing."

Example of a BAD answer (citations lumped at the end):
  "Vitamin D supports bone health by aiding calcium absorption and plays
   a role in immune function. Adults typically need 600-800 IU daily. [1][2]"
  (Citations must follow each individual claim, not be grouped at the end.)

Example of a BAD answer (no citations, unsupported claims):
  "You should definitely take 5000 IU of vitamin D daily. It will cure your
   fatigue and boost your immune system."
  (Makes unsupported claims, no citations, gives specific medical advice.)

Reference material:
{context}"""

_PURCHASE_SYSTEM_PROMPT = """\
You are a helpful shopping assistant for vitamins and supplements.
Present the purchase options clearly with titles and links.
Do NOT make health claims — only help the user find where to buy."""


# ── Node functions ───────────────────────────────────────────────────

def route_intent_node(state: ChatState) -> dict:
    """Classify the user's message intent."""
    msg = _current_message(state)
    has_personal = state.get("has_personal_data", False)
    history = _history_as_dicts(state)

    print(f"[ROUTE_INTENT] user_id={state.get('user_id')}, has_personal_data={has_personal}")
    print(f"[ROUTE_INTENT] message={msg!r}")

    intent = classify_intent(
        msg,
        has_personal_data=has_personal,
        conversation_history=history,
    )

    print(f"[ROUTE_INTENT] classified_intent={intent.value}")
    logger.info("Classified intent: %s", intent.value)
    return {"intent": intent}


def retrieve_general_node(state: ChatState) -> dict:
    """Retrieve chunks from the general knowledge base."""
    store = _get_vectorstore()
    query = state.get("rephrased_query") or _current_message(state)
    chunks = store.search(query, collection_type="general", k=5)
    logger.info("Retrieved %d chunks from general KB", len(chunks))
    return {"retrieved_chunks": chunks}


def retrieve_personal_node(state: ChatState) -> dict:
    """Retrieve chunks from the user's personal knowledge base."""
    store = _get_vectorstore()
    query = state.get("rephrased_query") or _current_message(state)
    chunks = store.search(query, collection_type="personal", user_id=state["user_id"], k=5)
    logger.info("Retrieved %d chunks from personal KB", len(chunks))
    return {"retrieved_chunks": chunks}


def search_purchase_node(state: ChatState) -> dict:
    """Search for purchase links via Tavily."""
    tavily = _get_tavily()
    query = state.get("rephrased_query") or _current_message(state)
    location = state.get("user_location")
    links = tavily.search_purchase_links(query, location=location)
    logger.info("Tavily returned %d purchase links (location=%s)", len(links), location)
    return {"purchase_links": links}


def generate_response_node(state: ChatState) -> dict:
    """Generate an LLM answer grounded in retrieved chunks.

    Handles three information scenarios:

    1. **Case A — No relevant chunks** (filtered_chunks empty):
       Skip the LLM, return a gap message directing the user to upload
       docs. Empty citations list, no Sources section.
    2. **Case B — Partial** (knowledge_gap set but some chunks exist):
       Answer with what's available, append a note about the gap.
       Only include citations actually referenced in the response.
    3. **Case C — Sufficient** (no gap):
       Normal cited answer with all referenced citations.
    """
    chunks = state.get("filtered_chunks") or state.get("retrieved_chunks", [])
    knowledge_gap = state.get("knowledge_gap")
    intent = state.get("intent")
    use_messages = bool(state.get("messages"))

    # ── Case A: no relevant chunks at all ─────────────────────────────
    if not chunks:
        gap = knowledge_gap or "information about this topic in the knowledge base"
        reply = _build_gap_only_response(gap, intent)
        result: dict = {"response": reply, "citations": []}
        if use_messages:
            result["messages"] = [AIMessage(content=reply)]
        return result

    # ── Cases B & C: we have some chunks ──────────────────────────────
    context, all_citations = format_citations(chunks)

    system = _RESPONSE_SYSTEM_PROMPT.format(
        context=context if context else "(No reference material found.)",
    )

    # Build conversation for the LLM.
    llm_messages = [SystemMessage(content=system)]
    if use_messages:
        llm_messages.extend(state["messages"])
    else:
        for turn in state.get("conversation_history", []):
            if turn.get("role") == "user":
                llm_messages.append(HumanMessage(content=turn["content"]))
            else:
                llm_messages.append(AIMessage(content=turn["content"]))
        llm_messages.append(HumanMessage(content=state["message"]))

    try:
        llm = _get_llm()
        response = llm.invoke(llm_messages)
    except Exception:
        logger.exception("LLM generation failed")
        fallback = "I'm having trouble processing your request right now. Please try again in a moment."
        result = {"response": fallback, "citations": []}
        if use_messages:
            result["messages"] = [AIMessage(content=fallback)]
        return result

    reply = response.content

    # ── Case B: partial info — append gap notice ─────────────────────
    if knowledge_gap:
        reply += _build_gap_suffix(knowledge_gap, intent)

    # Keep only cited sources and renumber so text and list match.
    reply, used_citations = _renumber_citations(reply, all_citations)

    result = {"response": reply, "citations": used_citations}
    if use_messages:
        result["messages"] = [AIMessage(content=reply)]
    return result


def _build_gap_only_response(gap: str, intent: IntentType | None) -> str:
    """Build a full response when there are zero usable chunks."""
    is_personal = intent in (IntentType.HEALTH_PERSONAL, IntentType.HEALTH_COMBINED)

    if is_personal:
        return (
            f"I don't have {gap} on file yet. "
            "You can upload them in the **Upload Documents** tab "
            "(PDF or TXT format), and I'll be able to analyze your "
            "data and provide personalized recommendations."
        )
    return (
        f"I don't currently have {gap}. "
        "You can upload a relevant reference document in the "
        "**Upload Documents** tab, and I'll be able to answer "
        "your question using that information."
    )


def _build_gap_suffix(gap: str, intent: IntentType | None) -> str:
    """Build a paragraph to append when we answered partially."""
    is_personal = intent in (IntentType.HEALTH_PERSONAL, IntentType.HEALTH_COMBINED)

    if is_personal:
        return (
            "\n\n---\n"
            f"*Note: I could only partially answer because I'm missing {gap}. "
            "If you have a document with this information, you can upload it "
            "in the **Upload Documents** tab for a more complete answer. "
            "You can also share relevant details directly in this chat.*"
        )
    return (
        "\n\n---\n"
        f"*Note: My answer may be incomplete because I'm missing {gap}. "
        "You can upload a more detailed reference document in the "
        "**Upload Documents** tab for a better answer.*"
    )


def _renumber_citations(
    text: str,
    all_citations: list,
) -> tuple[str, list]:
    """Keep only referenced citations and renumber them sequentially.

    Returns the updated response text and the filtered citations list,
    both using sequential 1-based numbering.

    Example::

        text = "...immune function [1]...iron absorption [3]..."
        all_citations = [cite_a, cite_b, cite_c]
        # [1] and [3] are used → mapped to new [1] and [2]
        # returns ("...immune function [1]...iron absorption [2]...", [cite_a, cite_c])
    """
    # Find which original 1-based indices appear in the text.
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)]", text)})

    # Keep only citations whose index was actually referenced and exists.
    used = [(old_idx, all_citations[old_idx - 1])
            for old_idx in cited
            if 1 <= old_idx <= len(all_citations)]

    if not used:
        # LLM didn't cite anything recognisable — return text as-is.
        return text, []

    # Build old→new mapping and apply replacements.
    # Replace largest numbers first so [1] replacement doesn't clobber [10].
    old_to_new: dict[int, int] = {}
    renumbered_citations: list = []
    for new_idx, (old_idx, citation) in enumerate(used, 1):
        old_to_new[old_idx] = new_idx
        renumbered_citations.append(citation)

    for old_idx in sorted(old_to_new, reverse=True):
        new_idx = old_to_new[old_idx]
        # Use exact pattern \[N] to avoid matching inside \[10] etc.
        text = re.sub(rf"\[{old_idx}]", f"[{new_idx}]", text)

    return text, renumbered_citations


def refuse_node(state: ChatState) -> dict:
    """Return a polite refusal for out-of-scope queries."""
    reply = (
        "I'm a vitamin and supplement advisor, so I can only help with "
        "questions about vitamins, supplements, and nutrition. "
        "Could you rephrase your question in that context?"
    )
    result: dict = {"response": reply, "citations": [], "purchase_links": []}
    if state.get("messages"):
        result["messages"] = [AIMessage(content=reply)]
    return result


def generate_purchase_response_node(state: ChatState) -> dict:
    """Format purchase links into a user-friendly response."""
    links = state.get("filtered_links") or state.get("purchase_links", [])
    use_messages = bool(state.get("messages"))

    if not links:
        fallback = (
            "Purchase links are currently unavailable. This could be a "
            "temporary issue with the search service. Please try again "
            "in a moment, or be more specific about the supplement."
        )
        result: dict = {"response": fallback, "citations": []}
        if use_messages:
            result["messages"] = [AIMessage(content=fallback)]
        return result

    lines = ["Here are some options I found:\n"]
    for i, link in enumerate(links, start=1):
        price_info = f" — {link.price}" if link.price else ""
        lines.append(f"{i}. [{link.title}]({link.url}){price_info}")

    query = _current_message(state)
    try:
        llm = _get_llm()
        user_content = (
            f"User asked: {query}\n\n"
            f"Available products:\n" + "\n".join(lines)
        )
        response = llm.invoke([
            SystemMessage(content=_PURCHASE_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ])
        reply = response.content
    except Exception:
        logger.exception("Purchase response generation failed")
        reply = "\n".join(lines)

    result = {"response": reply, "citations": [], "purchase_links": links}
    if use_messages:
        result["messages"] = [AIMessage(content=reply)]
    return result


# ── New pipeline nodes (rephrase / filter / clarify / retrieve_combined) ──


def rephrase_query_node(state: ChatState) -> dict:
    """Rephrase the user's message into a search-optimized standalone query.

    The classified intent and ``has_personal_data`` flag are injected into
    the system prompt so the LLM understands the context and avoids
    unnecessary clarification for personal/combined queries.
    """
    user_msg = _current_message(state)
    history = _history_as_dicts(state)
    has_personal = state.get("has_personal_data", False)
    intent = state.get("intent")
    intent_label = intent.value if intent else "health_general"

    context_parts = [f"{t['role']}: {t['content']}" for t in history[-6:]]
    context_parts.append(f"user: {user_msg}")

    system_prompt = _REPHRASE_SYSTEM_PROMPT.format(
        intent=intent_label,
        has_personal_data=has_personal,
    )

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="\n".join(context_parts)),
        ])
        text = response.content.strip()
    except Exception:
        logger.exception("Rephrase failed, using original message")
        return {"rephrased_query": user_msg, "needs_clarification": False}

    if text.startswith("CLARIFY:"):
        # Hard guard: never clarify for personal/combined intents.
        # The user is clearly asking about their own data; knowledge gap
        # detection downstream handles missing documents.
        if intent in (IntentType.HEALTH_PERSONAL, IntentType.HEALTH_COMBINED):
            print(f"[REPHRASE] Suppressing clarification (intent={intent_label})")
            logger.info("Suppressing clarification for intent %s", intent_label)
            return {"rephrased_query": user_msg, "needs_clarification": False}

        question = text[len("CLARIFY:"):].strip()
        logger.info("Rephrase node requests clarification: %s", question)
        return {"needs_clarification": True, "clarification_question": question}

    logger.info("Rephrased query: %s", text)
    return {"rephrased_query": text, "needs_clarification": False}


def rephrase_purchase_query_node(state: ChatState) -> dict:
    """Rephrase the user's message into a product search query.

    Handles location awareness:
    - If ``user_location`` is already in state (from a prior turn), it is
      injected into the LLM context so the query includes it automatically.
    - If the LLM returns a ``LOCATION:`` prefix, it means the product is
      clear but location is unknown — we ask the user for their location.
    - On the next turn the user provides their city/country, the rephrase
      LLM picks it up from conversation history and embeds it in the query.
    """
    user_msg = _current_message(state)
    history = _history_as_dicts(state)
    known_location = state.get("user_location")

    context_parts = [f"{t['role']}: {t['content']}" for t in history[-6:]]
    context_parts.append(f"user: {user_msg}")

    # Inject known location so the LLM always includes it.
    if known_location:
        context_parts.append(f"[User location: {known_location}]")

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_REPHRASE_PURCHASE_SYSTEM_PROMPT),
            HumanMessage(content="\n".join(context_parts)),
        ])
        text = response.content.strip()
    except Exception:
        logger.exception("Purchase rephrase failed, using original message")
        return {"rephrased_query": user_msg, "needs_clarification": False}

    if text.startswith("CLARIFY:"):
        question = text[len("CLARIFY:"):].strip()
        logger.info("Purchase rephrase requests clarification: %s", question)
        return {"needs_clarification": True, "clarification_question": question}

    if text.startswith("LOCATION:"):
        # Product is clear but location unknown — ask the user.
        query_without_location = text[len("LOCATION:"):].strip()
        logger.info("Purchase rephrase needs location for query: %s", query_without_location)
        return {
            "needs_clarification": True,
            "clarification_question": (
                "I'd like to find the best purchase options for you. "
                "What city and country are you located in?"
            ),
            # Stash the partial query so it can be used after the user replies.
            "rephrased_query": query_without_location,
        }

    # Check if the LLM included a location we didn't know about yet.
    result: dict = {"rephrased_query": text, "needs_clarification": False}
    if not known_location:
        extracted = _extract_location(text)
        if extracted:
            result["user_location"] = extracted

    logger.info("Rephrased purchase query: %s", text)
    return result


def _extract_location(query: str) -> str | None:
    """Best-effort extraction of location from a rephrased purchase query.

    Looks for patterns like "buy ... in Eindhoven Netherlands" and returns
    the location portion, or None if not found.
    """
    lower = query.lower()
    for marker in (" in ", " near "):
        idx = lower.rfind(marker)
        if idx != -1:
            location = query[idx + len(marker):].strip().rstrip(".")
            if location and len(location) < 80:
                logger.info("Extracted user location from rephrase: %s", location)
                return location
    return None


def retrieve_combined_node(state: ChatState) -> dict:
    """Retrieve chunks from both general and personal knowledge bases."""
    store = _get_vectorstore()
    query = state.get("rephrased_query") or _current_message(state)
    chunks = store.search_both(query, user_id=state["user_id"], k=5)
    logger.info("Retrieved %d chunks from general+personal KB", len(chunks))
    return {"retrieved_chunks": chunks}


def filter_relevance_node(state: ChatState) -> dict:
    """Post-retrieval filter: LLM evaluates each chunk as KEEP or DROP.

    After filtering, runs a knowledge-gap check to determine whether
    the surviving chunks actually contain enough information to answer
    the user's question.  Sets ``knowledge_gap`` in state when they don't.
    """
    chunks = state.get("retrieved_chunks", [])
    user_msg = _current_message(state)

    if not chunks:
        # No chunks at all — detect what's missing based on intent.
        gap = _detect_empty_gap(state)
        return {"filtered_chunks": [], "knowledge_gap": gap}

    chunk_texts = []
    for i, c in enumerate(chunks, 1):
        text = c.get("text", c.get("page_content", ""))
        chunk_texts.append(f"Chunk {i}: {text[:300]}")

    prompt = f"User question: {user_msg}\n\n" + "\n".join(chunk_texts)

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_FILTER_RELEVANCE_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        verdicts = response.content.strip().upper().split("\n")
    except Exception:
        logger.exception("Filter relevance failed, keeping all chunks")
        return {"filtered_chunks": chunks, "knowledge_gap": None}

    kept = []
    for i, chunk in enumerate(chunks):
        verdict = verdicts[i].strip() if i < len(verdicts) else "KEEP"
        if verdict == "KEEP":
            kept.append(chunk)

    logger.info("Filter kept %d / %d chunks", len(kept), len(chunks))

    # ── Knowledge-gap detection ──────────────────────────────────────
    if not kept:
        # All chunks dropped by strict filter — treat as empty retrieval.
        gap = _detect_empty_gap(state)
        logger.info("All chunks filtered out — gap: %s", gap)
        return {"filtered_chunks": [], "knowledge_gap": gap}

    gap = _detect_knowledge_gap(user_msg, kept)
    if gap:
        logger.info("Knowledge gap detected: %s", gap)
    return {"filtered_chunks": kept, "knowledge_gap": gap}


def _detect_empty_gap(state: ChatState) -> str:
    """Produce a gap description when retrieval returned zero chunks."""
    intent = state.get("intent")
    if intent in (IntentType.HEALTH_PERSONAL, IntentType.HEALTH_COMBINED):
        return "your personal lab results or health records"
    return "information about this topic in the knowledge base"


def _detect_knowledge_gap(user_msg: str, kept_chunks: list[dict]) -> str | None:
    """Ask the LLM whether *kept_chunks* can answer *user_msg*.

    Returns a gap description string, or ``None`` if the chunks are sufficient.
    """
    chunk_summaries = []
    for i, c in enumerate(kept_chunks, 1):
        text = c.get("text", c.get("page_content", ""))
        chunk_summaries.append(f"Chunk {i}: {text[:300]}")

    prompt = f"User question: {user_msg}\n\nAvailable chunks:\n" + "\n".join(chunk_summaries)

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_GAP_DETECTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        text = response.content.strip()
    except Exception:
        logger.exception("Gap detection failed, assuming sufficient")
        return None

    if text.upper().startswith("SUFFICIENT"):
        return None

    if text.upper().startswith("GAP:"):
        return text[4:].strip()

    # Unexpected format — don't block the pipeline.
    logger.warning("Unexpected gap detection response: %s", text)
    return None


def filter_links_node(state: ChatState) -> dict:
    """Post-search filter: LLM evaluates each purchase link as KEEP or DROP."""
    links = state.get("purchase_links", [])
    if not links:
        return {"filtered_links": []}

    user_msg = _current_message(state)
    link_texts = []
    for i, link in enumerate(links, 1):
        price_info = f" — {link.price}" if link.price else ""
        link_texts.append(f"Link {i}: {link.title}{price_info}")

    prompt = f"User question: {user_msg}\n\n" + "\n".join(link_texts)

    try:
        llm = _get_llm()
        response = llm.invoke([
            SystemMessage(content=_FILTER_LINKS_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ])
        verdicts = response.content.strip().upper().split("\n")
    except Exception:
        logger.exception("Filter links failed, keeping all links")
        return {"filtered_links": links}

    kept = []
    for i, link in enumerate(links):
        verdict = verdicts[i].strip() if i < len(verdicts) else "KEEP"
        if verdict == "KEEP":
            kept.append(link)

    if not kept:
        logger.warning("Filter dropped all links — keeping originals")
        kept = links

    logger.info("Filter kept %d / %d links", len(kept), len(links))
    return {"filtered_links": kept}


def clarify_node(state: ChatState) -> dict:
    """Return a clarification question to the user."""
    question = state.get("clarification_question", "Could you please be more specific?")
    result: dict = {"response": question, "citations": [], "purchase_links": []}
    if state.get("messages"):
        result["messages"] = [AIMessage(content=question)]
    return result
