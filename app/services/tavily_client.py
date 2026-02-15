"""Tavily search API wrapper.

IMPORTANT: This service is strictly for *purchase / product* searches.
It must NEVER be used to retrieve medical evidence, health claims, or
clinical data.  All health-related retrieval must go through the
vector-store knowledge base.
"""

from __future__ import annotations

import logging

from tavily import TavilyClient

from app.config import settings
from app.models.schemas import PurchaseLink

logger = logging.getLogger(__name__)

# Keywords that signal a result is purchase-relevant.
_SHOPPING_SIGNALS = ("buy", "shop", "price", "cart", "product", "store", "order")


class TavilyService:
    """Thin wrapper around the Tavily search API for product lookups.

    WARNING: This service is for purchase-link retrieval ONLY.
    Never use it for medical or health-evidence queries.
    """

    def __init__(self) -> None:
        self._client = TavilyClient(api_key=settings.tavily_api_key)
        self._timeout = settings.tavily_timeout

    def search_purchase_links(
        self,
        query: str,
        max_results: int = 5,
        location: str | None = None,
    ) -> list[PurchaseLink]:
        """Search for product purchase links.

        Automatically appends "buy" to the query when no purchase-related
        keyword is already present, then filters results to only include
        shopping-relevant pages.

        Args:
            query: User-facing product search string.
            max_results: Cap on returned results.
            location: Optional city/country to append for local results.

        Returns:
            List of :class:`PurchaseLink` instances (empty on failure).
        """
        # Ensure the query is purchase-oriented.
        if not any(kw in query.lower() for kw in ("buy", "purchase")):
            query = f"buy {query}"

        # Append location for more relevant local results.
        if location and location.lower() not in query.lower():
            query = f"{query} in {location}"

        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                timeout=self._timeout,
            )
        except Exception:
            logger.exception("Tavily search failed for query: %s", query)
            return []

        results: list[PurchaseLink] = []
        for item in response.get("results", []):
            url: str = item.get("url", "")
            title: str = item.get("title", "")
            content: str = item.get("content", "")
            combined = f"{url} {title} {content}".lower()

            if any(signal in combined for signal in _SHOPPING_SIGNALS):
                results.append(PurchaseLink(title=title, url=url))

        return results[:max_results]
