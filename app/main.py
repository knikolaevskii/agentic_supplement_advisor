"""FastAPI application entry point."""

from __future__ import annotations

import logging
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage

from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    ConfirmUploadRequest,
    ConversationInfo,
    ConversationMessage,
    DocumentInfo,
    UploadResponse,
)
from app.services.conversation import ConversationService
from app.services.vectorstore import VectorStoreService

logger = logging.getLogger(__name__)

# ── Service singletons ───────────────────────────────────────────────

vectorstore: VectorStoreService | None = None
conversation_service: ConversationService | None = None
_chat_graph = None  # compiled graph with checkpointer


def _get_vectorstore() -> VectorStoreService:
    if vectorstore is None:
        raise RuntimeError("VectorStoreService not initialised")
    return vectorstore


def _get_conversation_service() -> ConversationService:
    if conversation_service is None:
        raise RuntimeError("ConversationService not initialised")
    return conversation_service


def _get_chat_graph():
    if _chat_graph is None:
        raise RuntimeError("Chat graph not initialised")
    return _chat_graph


# ── Lifespan ─────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorstore, conversation_service, _chat_graph

    vectorstore = VectorStoreService()

    os.makedirs("data", exist_ok=True)
    conversation_service = ConversationService("data/chats.db")

    from langgraph.checkpoint.sqlite import SqliteSaver

    conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()

    from app.agents.chat_graph import build_chat_graph

    _chat_graph = build_chat_graph(checkpointer=checkpointer)

    logger.info("Services initialised")

    from app.config import settings

    if settings.langsmith_api_key and settings.langsmith_tracing:
        logger.info(
            "LangSmith tracing enabled for project: %s",
            settings.langsmith_project,
        )
    else:
        logger.info("LangSmith tracing disabled (no API key set)")

    yield
    logger.info("Shutting down")


# ── App ──────────────────────────────────────────────────────────────

app = FastAPI(title="Supplement Advisor", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Error handling ───────────────────────────────────────────────────


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Helpers ──────────────────────────────────────────────────────────


MAX_MESSAGE_LENGTH = 2000
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {"pdf", "txt"}


def _validate_user_id(user_id: str) -> None:
    if not user_id or not user_id.strip():
        raise HTTPException(status_code=422, detail="user_id must be a non-empty string")


def _validate_message(message: str) -> None:
    if len(message) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Message exceeds maximum length of {MAX_MESSAGE_LENGTH} characters",
        )


def _validate_file(file_bytes: bytes, filename: str) -> None:
    ext = filename.rsplit(".", maxsplit=1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '.{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=422,
            detail=f"File exceeds maximum size of {MAX_FILE_SIZE // (1024 * 1024)} MB",
        )


# ── Endpoints ────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    _validate_user_id(req.user_id)
    _validate_message(req.message)

    logger.info("Chat request from user=%s len=%d", req.user_id, len(req.message))

    graph = _get_chat_graph()
    convs = _get_conversation_service()
    store = _get_vectorstore()
    user_docs = store.list_user_documents(req.user_id)
    has_personal = len(user_docs) > 0
    print(f"[CHAT] user_id={req.user_id}, has_personal_data={has_personal}, personal_docs={user_docs}")

    # Resolve or create conversation.
    conversation_id = req.conversation_id
    if conversation_id:
        conv = convs.get(conversation_id)
        if not conv or conv["user_id"] != req.user_id:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        title = req.message[:50]
        conv = convs.create(req.user_id, title)
        conversation_id = conv["id"]

    config = {"configurable": {"thread_id": conversation_id}}
    result = graph.invoke(
        {
            "messages": [HumanMessage(content=req.message)],
            "user_id": req.user_id,
            "has_personal_data": has_personal,
        },
        config=config,
    )

    convs.touch(conversation_id)

    logger.info("Chat response generated for user=%s conv=%s", req.user_id, conversation_id)
    return ChatResponse(
        reply=result.get("response", ""),
        citations=result.get("citations", []),
        purchase_links=result.get("purchase_links", []),
        conversation_id=conversation_id,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(user_id: str = Form(...), file: UploadFile = Form(...)):
    _validate_user_id(user_id)

    file_bytes = await file.read()
    filename = file.filename or "document"
    _validate_file(file_bytes, filename)
    doc_id = uuid.uuid4().hex[:12]

    logger.info("Upload request: user=%s file=%s size=%d", user_id, filename, len(file_bytes))

    from app.agents.upload_graph import upload_graph

    result = upload_graph.invoke({
        "user_id": user_id,
        "file_bytes": file_bytes,
        "filename": filename,
        "doc_id": doc_id,
    })

    classification = result.get("classification", "ambiguous")
    if hasattr(classification, "value"):
        classification = classification.value

    collection_name = f"personal_{user_id}" if classification == "personal" else "general_kb"
    print(f"[UPLOAD] user_id={user_id}, doc_id={doc_id}, classification={classification}, collection={collection_name}")

    # When AMBIGUOUS, pass chunks back so the UI can send them to /upload/confirm.
    chunks = result.get("chunks", []) if classification == "ambiguous" else []

    logger.info("Upload complete: doc_id=%s classification=%s", doc_id, classification)
    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        classification=classification,
        chunks_created=result.get("chunks_created", 0),
        chunks=chunks,
    )


@app.post("/upload/confirm", response_model=UploadResponse)
async def upload_confirm(req: ConfirmUploadRequest):
    _validate_user_id(req.user_id)

    if req.classification not in ("general", "personal"):
        raise HTTPException(
            status_code=422,
            detail="classification must be 'general' or 'personal'",
        )

    store = _get_vectorstore()
    collection_name = f"personal_{req.user_id}" if req.classification == "personal" else "general_kb"
    print(f"[UPLOAD_CONFIRM] user_id={req.user_id}, doc_id={req.doc_id}, classification={req.classification}, collection={collection_name}")

    store.add_documents(
        req.chunks,
        collection_type=req.classification,
        user_id=req.user_id if req.classification == "personal" else None,
    )

    return UploadResponse(
        doc_id=req.doc_id,
        filename=req.filename,
        classification=req.classification,
        chunks_created=len(req.chunks),
    )


# ── Conversation endpoints ───────────────────────────────────────────


@app.get("/conversations/{user_id}", response_model=list[ConversationInfo])
async def list_conversations(user_id: str):
    _validate_user_id(user_id)
    convs = _get_conversation_service()
    rows = convs.list_for_user(user_id)
    return [ConversationInfo(**r) for r in rows]


@app.post("/conversations/{user_id}", response_model=ConversationInfo)
async def create_conversation(user_id: str):
    _validate_user_id(user_id)
    convs = _get_conversation_service()
    row = convs.create(user_id, "New Chat")
    return ConversationInfo(**row)


@app.delete("/conversations/{user_id}/{conversation_id}")
async def delete_conversation(user_id: str, conversation_id: str):
    _validate_user_id(user_id)
    convs = _get_conversation_service()
    conv = convs.get(conversation_id)
    if not conv or conv["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    convs.delete(conversation_id)
    return {"deleted": True}


@app.get(
    "/conversations/{user_id}/{conversation_id}/messages",
    response_model=list[ConversationMessage],
)
async def get_conversation_messages(user_id: str, conversation_id: str):
    _validate_user_id(user_id)

    convs = _get_conversation_service()
    conv = convs.get(conversation_id)
    if not conv or conv["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Conversation not found")

    graph = _get_chat_graph()
    config = {"configurable": {"thread_id": conversation_id}}
    snapshot = graph.get_state(config)

    msgs: list[ConversationMessage] = []
    if snapshot and snapshot.values and snapshot.values.get("messages"):
        for m in snapshot.values["messages"]:
            role = "assistant" if isinstance(m, AIMessage) else "user"
            msgs.append(ConversationMessage(role=role, content=m.content))
    return msgs


# ── Document endpoints ───────────────────────────────────────────────


@app.get("/documents/preview/{doc_id}")
async def get_document_preview(doc_id: str, collection_type: str, user_id: str | None = None):
    if collection_type not in ("general", "personal"):
        raise HTTPException(status_code=422, detail="collection_type must be 'general' or 'personal'")
    if collection_type == "personal" and not user_id:
        raise HTTPException(status_code=422, detail="user_id required for personal documents")

    store = _get_vectorstore()
    preview = store.get_document_preview(doc_id, collection_type, user_id=user_id)
    return {"doc_id": doc_id, "preview": preview}


@app.get("/documents/general", response_model=list[DocumentInfo])
async def list_general_documents():
    store = _get_vectorstore()
    docs = store.list_general_documents()

    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d.get("title", ""),
            classification="general",
            created_at="",
        )
        for d in docs
    ]


@app.get("/documents/{user_id}", response_model=list[DocumentInfo])
async def list_documents(user_id: str):
    _validate_user_id(user_id)

    store = _get_vectorstore()
    docs = store.list_user_documents(user_id)

    return [
        DocumentInfo(
            doc_id=d["doc_id"],
            filename=d.get("title", ""),
            classification="personal",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        for d in docs
    ]


@app.delete("/documents/{user_id}/{doc_id}")
async def delete_document(user_id: str, doc_id: str):
    _validate_user_id(user_id)

    store = _get_vectorstore()
    store.delete_document(doc_id, collection_type="personal", user_id=user_id)

    return {"deleted": True}
