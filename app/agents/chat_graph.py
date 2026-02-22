"""LangGraph chat agent.

Builds a compiled :class:`StateGraph` that routes user messages through
intent classification, retrieval planning, execution, relevance filtering,
and response generation.

Graph structure::

    START -> route_intent -> +-- HEALTH_GENERAL  --+
                             +-- HEALTH_PERSONAL --+--> plan_retrieval --+--> execute_retrieval --+--> filter_relevance --> generate_response --> END
                             +-- HEALTH_COMBINED --+                    |                        |
                             |                                         |                        +--> analyze_and_replan --> execute_targeted_retrieval --> filter_relevance --> generate_response --> END
                             |                                         +--> clarify --> END
                             |
                             +-- PURCHASE --> rephrase_purchase_query --+--> search_purchase --> filter_links --> generate_purchase_response --> END
                             |                                         +--> clarify --> END
                             +-- OUT_OF_SCOPE --> refuse --> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.agents.nodes import (
    ChatState,
    analyze_and_replan_node,
    clarify_node,
    execute_retrieval_node,
    execute_targeted_retrieval_node,
    filter_links_node,
    filter_relevance_node,
    generate_purchase_response_node,
    generate_response_node,
    plan_retrieval_node,
    refuse_node,
    rephrase_purchase_query_node,
    route_intent_node,
    search_purchase_node,
)
from app.models.schemas import IntentType

# -- Routing functions --------------------------------------------------------


def _route_by_intent(state: ChatState) -> str:
    """After intent classification: health intents -> plan_retrieval, purchase -> rephrase_purchase."""
    intent = state.get("intent")
    if intent is IntentType.PURCHASE:
        return "rephrase_purchase_query"
    if intent is IntentType.OUT_OF_SCOPE:
        return "refuse"
    # HEALTH_GENERAL, HEALTH_PERSONAL, HEALTH_COMBINED all go to plan_retrieval.
    return "plan_retrieval"


def _route_after_plan(state: ChatState) -> str:
    """After planning retrieval: clarify or execute the plan."""
    if state.get("needs_clarification"):
        return "clarify"
    return "execute_retrieval"


def _route_after_execute(state: ChatState) -> str:
    """After first retrieval: HEALTH_COMBINED gets two-phase analysis, others go to filter."""
    intent = state.get("intent")
    if intent is IntentType.HEALTH_COMBINED:
        return "analyze_and_replan"
    return "filter_relevance"


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
    graph.add_node("plan_retrieval", plan_retrieval_node)
    graph.add_node("execute_retrieval", execute_retrieval_node)
    graph.add_node("analyze_and_replan", analyze_and_replan_node)
    graph.add_node("execute_targeted_retrieval", execute_targeted_retrieval_node)
    graph.add_node("rephrase_purchase_query", rephrase_purchase_query_node)
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
            "plan_retrieval": "plan_retrieval",
            "rephrase_purchase_query": "rephrase_purchase_query",
            "refuse": "refuse",
        },
    )

    graph.add_conditional_edges(
        "plan_retrieval",
        _route_after_plan,
        {
            "clarify": "clarify",
            "execute_retrieval": "execute_retrieval",
        },
    )

    # After execute_retrieval: HEALTH_COMBINED -> analyze_and_replan, others -> filter
    graph.add_conditional_edges(
        "execute_retrieval",
        _route_after_execute,
        {
            "analyze_and_replan": "analyze_and_replan",
            "filter_relevance": "filter_relevance",
        },
    )

    # Two-phase HEALTH_COMBINED path
    graph.add_edge("analyze_and_replan", "execute_targeted_retrieval")
    graph.add_edge("execute_targeted_retrieval", "filter_relevance")

    # Common tail: filter -> generate
    graph.add_edge("filter_relevance", "generate_response")
    graph.add_edge("generate_response", END)

    graph.add_conditional_edges(
        "rephrase_purchase_query",
        _route_after_purchase_rephrase,
        {
            "clarify": "clarify",
            "search_purchase": "search_purchase",
        },
    )

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
