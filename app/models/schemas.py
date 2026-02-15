"""Request / response models and domain enumerations."""

from enum import Enum

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────────

class IntentType(str, Enum):
    """Classified intent of an incoming user message."""

    HEALTH_GENERAL = "health_general"
    HEALTH_PERSONAL = "health_personal"
    HEALTH_COMBINED = "health_combined"
    PURCHASE = "purchase"
    OUT_OF_SCOPE = "out_of_scope"


class DocClassification(str, Enum):
    """How an uploaded document is classified for routing."""

    GENERAL = "general"
    PERSONAL = "personal"
    AMBIGUOUS = "ambiguous"


# ── Shared sub-models ────────────────────────────────────────────────

class Citation(BaseModel):
    """A reference back to a specific knowledge-base chunk."""

    doc_id: str
    chunk_id: str
    title: str
    source: str
    snippet: str
    full_text: str = ""


class PurchaseLink(BaseModel):
    """A product link returned by the purchase-research agent."""

    title: str
    url: str
    price: str | None = None


# ── Chat ─────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    """Incoming chat message from a user."""

    user_id: str
    message: str
    conversation_history: list[dict] = Field(default_factory=list)
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    """Advisor reply sent back to the user."""

    reply: str
    citations: list[Citation]
    purchase_links: list[PurchaseLink] = Field(default_factory=list)
    conversation_id: str | None = None


# ── Conversations ─────────────────────────────────────────────────

class ConversationInfo(BaseModel):
    """Metadata for a chat conversation."""

    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str


class ConversationMessage(BaseModel):
    """A single message in a conversation."""

    role: str
    content: str


# ── Document upload ──────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Result returned after a document is ingested."""

    doc_id: str
    filename: str
    classification: str
    chunks_created: int
    chunks: list[dict] = Field(default_factory=list, exclude=False)


class ConfirmUploadRequest(BaseModel):
    """Request to complete ingestion after an AMBIGUOUS classification."""

    user_id: str
    doc_id: str
    filename: str
    classification: str  # "general" or "personal"
    chunks: list[dict]


class DocumentInfo(BaseModel):
    """Metadata for a previously uploaded document."""

    doc_id: str
    filename: str
    classification: str
    created_at: str
