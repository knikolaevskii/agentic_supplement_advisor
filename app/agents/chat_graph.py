"""LangGraph chat agent.

Builds a compiled :class:`StateGraph` that routes user messages through
intent classification, query rephrasing, retrieval, relevance filtering,
and response generation.

Graph structure::

    START -> route_intent -> +-- HEALTH_GENERAL  --+
                             +-- HEALTH_PERSONAL --+--> rephrase_query --+--> retrieve_general  --+
                             +-- HEALTH_COMBINED --+                    +--> retrieve_personal --+--> filter_relevance --> generate_response --> END
                             |                                          +--> retrieve_combined --+
                             +-- PURCHASE --> rephrase_purchase_query --> search_purchase --> filter_links --> generate_purchase_response --> END
                             +-- OUT_OF_SCOPE --> refuse --> END

    (any rephrase node may set needs_clarification=True --> clarify --> END)
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes import (
    ChatState,
    clarify_node,
    filter_links_node,
    filter_relevance_node,
    generate_purchase_response_node,
    generate_response_node,
    refuse_node,
    rephrase_purchase_query_node,
    rephrase_query_node,
    retrieve_combined_node,
    retrieve_general_node,
    retrieve_personal_node,
    route_intent_node,
    search_purchase_node,
)
from app.models.schemas import IntentType

# -- Routing functions --------------------------------------------------------


def _route_by_intent(state: ChatState) -> str:
    """After intent classification: health intents -> rephrase, purchase -> rephrase_purchase."""
    intent = state.get("intent")
    if intent is IntentType.PURCHASE:
        return "rephrase_purchase_query"
    if intent is IntentType.OUT_OF_SCOPE:
        return "refuse"
    # HEALTH_GENERAL, HEALTH_PERSONAL, HEALTH_COMBINED all go to rephrase.
    return "rephrase_query"


def _route_after_rephrase(state: ChatState) -> str:
    """After rephrasing a health query: clarify, or route to the right retriever."""
    if state.get("needs_clarification"):
        return "clarify"
    intent = state.get("intent")
    if intent is IntentType.HEALTH_PERSONAL:
        return "retrieve_personal"
    if intent is IntentType.HEALTH_COMBINED:
        return "retrieve_combined"
    return "retrieve_general"


def _route_after_purchase_rephrase(state: ChatState) -> str:
    """After rephrasing a purchase query: clarify or search."""
    if state.get("needs_clarification"):
        return "clarify"
    return "search_purchase"


# -- Graph construction -------------------------------------------------------


def build_chat_graph(checkpointer=None) -> CompiledStateGraph:
    """Construct and compile the chat agent graph."""
    # langgraph dev passes server config as a dict via invoke_factory;
    # graph.compile() only accepts BaseCheckpointSaver | True | False | None.
    if isinstance(checkpointer, dict):
        checkpointer = None

    graph = StateGraph(ChatState)

    # Nodes (13 total)
    graph.add_node("route_intent", route_intent_node)
    graph.add_node("rephrase_query", rephrase_query_node)
    graph.add_node("rephrase_purchase_query", rephrase_purchase_query_node)
    graph.add_node("retrieve_general", retrieve_general_node)
    graph.add_node("retrieve_personal", retrieve_personal_node)
    graph.add_node("retrieve_combined", retrieve_combined_node)
    graph.add_node("filter_relevance", filter_relevance_node)
    graph.add_node("search_purchase", search_purchase_node)
    graph.add_node("filter_links", filter_links_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_node("generate_purchase_response", generate_purchase_response_node)
    graph.add_node("refuse", refuse_node)
    graph.add_node("clarify", clarify_node)

    # Edges
    graph.add_edge(START, "route_intent")

    graph.add_conditional_edges(
        "route_intent",
        _route_by_intent,
        {
            "rephrase_query": "rephrase_query",
            "rephrase_purchase_query": "rephrase_purchase_query",
            "refuse": "refuse",
        },
    )

    graph.add_conditional_edges(
        "rephrase_query",
        _route_after_rephrase,
        {
            "clarify": "clarify",
            "retrieve_general": "retrieve_general",
            "retrieve_personal": "retrieve_personal",
            "retrieve_combined": "retrieve_combined",
        },
    )

    graph.add_conditional_edges(
        "rephrase_purchase_query",
        _route_after_purchase_rephrase,
        {
            "clarify": "clarify",
            "search_purchase": "search_purchase",
        },
    )

    # Retrieval -> filter -> generate
    graph.add_edge("retrieve_general", "filter_relevance")
    graph.add_edge("retrieve_personal", "filter_relevance")
    graph.add_edge("retrieve_combined", "filter_relevance")
    graph.add_edge("filter_relevance", "generate_response")
    graph.add_edge("generate_response", END)

    # Purchase: search -> filter -> generate
    graph.add_edge("search_purchase", "filter_links")
    graph.add_edge("filter_links", "generate_purchase_response")
    graph.add_edge("generate_purchase_response", END)

    # Terminal nodes
    graph.add_edge("refuse", END)
    graph.add_edge("clarify", END)

    return graph.compile(checkpointer=checkpointer)


# Pre-built compiled graph for import convenience.
chat_graph = build_chat_graph()
