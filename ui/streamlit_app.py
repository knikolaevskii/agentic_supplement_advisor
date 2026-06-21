"""Streamlit UI for Supplement Advisor."""

from __future__ import annotations

import base64
import re

import requests
import streamlit as st


def _linkify_citations(text: str) -> str:
    """Replace [N] markers in the response text with clickable anchor links."""
    return re.sub(
        r"\[(\d+)\]",
        r'<a href="#citation-\1" style="text-decoration:none;color:#0068C9;font-weight:600;">[\1]</a>',
        text,
    )


def _render_citations(citations: list[dict]) -> None:
    """Render a flat source list with anchor targets for each citation."""
    if not citations:
        return
    st.markdown(
        '<p style="margin:0.75rem 0 0.25rem;font-weight:600;color:#555;font-size:0.85rem;">Sources</p>',
        unsafe_allow_html=True,
    )
    for i, c in enumerate(citations, 1):
        page = c.get("page", 1)
        page_info = f" · p. {page}" if page and page > 1 else ""
        full = c.get("full_text") or c.get("snippet", "")
        st.markdown(
            f'<div id="citation-{i}" style="scroll-margin-top:4rem"></div>',
            unsafe_allow_html=True,
        )
        with st.expander(f"[{i}] {c['title']}{page_info}"):
            st.markdown(full)

def _render_document(api_base: str, doc_id: str, filename: str, **preview_params) -> None:
    """Embed the original file if available, else fall back to text preview."""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    try:
        file_resp = requests.get(
            f"{api_base}/documents/file/{doc_id}",
            params={"filename": filename},
            timeout=15,
        )
        if file_resp.status_code == 200:
            if ext == "pdf":
                b64 = base64.b64encode(file_resp.content).decode()
                st.markdown(
                    f'<iframe src="data:application/pdf;base64,{b64}" '
                    f'width="100%" height="600" style="border:none;border-radius:6px;"></iframe>',
                    unsafe_allow_html=True,
                )
            else:
                with st.container(height=300):
                    st.text(file_resp.text)
            return
    except Exception:
        pass

    # Fallback: text preview from vector store
    try:
        prev_resp = requests.get(
            f"{api_base}/documents/preview/{doc_id}",
            params=preview_params,
            timeout=10,
        )
        prev_resp.raise_for_status()
        preview = prev_resp.json().get("preview", "")
        if preview:
            with st.container(height=300):
                st.text(preview)
        else:
            st.caption("No content available.")
    except Exception:
        st.caption("Could not load document.")


st.set_page_config(page_title="Supplement Advisor", page_icon="💊")

# ── Custom CSS ────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Vertically centre sidebar conversation rows. */
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        align-items: center;
    }
    /* Make chat messages container fill available height. */
    div[data-testid="stTabs"] div[data-testid="stVerticalBlockBorderWrapper"] {
        height: calc(100vh - 18rem) !important;
        max-height: calc(100vh - 18rem) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────

st.sidebar.title("Settings")
user_id = st.sidebar.text_input("User ID", value="default_user")
api_base = st.sidebar.text_input("API Base URL", value="http://localhost:8000")

if not user_id.strip():
    st.warning("Please enter a User ID in the sidebar.")
    st.stop()

# ── Session state defaults ───────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": ..., "content": ...}

if "pending_upload" not in st.session_state:
    st.session_state.pending_upload = None  # stash AMBIGUOUS upload data

if "active_conversation_id" not in st.session_state:
    st.session_state.active_conversation_id = None

if "conversations" not in st.session_state:
    st.session_state.conversations = []

if "prev_user_id" not in st.session_state:
    st.session_state.prev_user_id = user_id

# Reset on user_id change.
if st.session_state.prev_user_id != user_id:
    st.session_state.active_conversation_id = None
    st.session_state.chat_history = []
    st.session_state.conversations = []
    st.session_state.prev_user_id = user_id

# ── Conversation sidebar ─────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.subheader("Conversations")

if st.sidebar.button("+ New Chat", use_container_width=True):
    st.session_state.active_conversation_id = None
    st.session_state.chat_history = []
    st.rerun()

# Fetch conversations from API.
try:
    resp = requests.get(f"{api_base}/conversations/{user_id}", timeout=5)
    resp.raise_for_status()
    st.session_state.conversations = resp.json()
except Exception:
    pass  # silently keep cached list

for conv in st.session_state.conversations:
    col1, col2 = st.sidebar.columns([5, 1])
    is_active = conv["id"] == st.session_state.active_conversation_id
    label = f"**{conv['title']}**" if is_active else conv["title"]
    if col1.button(label, key=f"conv_{conv['id']}", use_container_width=True):
        st.session_state.active_conversation_id = conv["id"]
        # Load messages from API.
        try:
            msg_resp = requests.get(
                f"{api_base}/conversations/{user_id}/{conv['id']}/messages",
                timeout=10,
            )
            msg_resp.raise_for_status()
            st.session_state.chat_history = msg_resp.json()
        except Exception:
            st.session_state.chat_history = []
        st.rerun()
    if col2.button("🗑", key=f"del_conv_{conv['id']}"):
        try:
            requests.delete(
                f"{api_base}/conversations/{user_id}/{conv['id']}",
                timeout=10,
            )
        except Exception:
            pass
        if st.session_state.active_conversation_id == conv["id"]:
            st.session_state.active_conversation_id = None
            st.session_state.chat_history = []
        st.rerun()

# ── Tabs ─────────────────────────────────────────────────────────────

tab_chat, tab_upload, tab_personal, tab_general = st.tabs(
    ["Chat", "Upload Documents", "Personal Documents", "General Documents"],
)

# =====================================================================
# Tab 1: Chat
# =====================================================================

with tab_chat:
    # Scrollable messages area.
    chat_container = st.container(height=900)
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div style="text-align:center; padding:4rem 1rem 2rem;">
                    <h2>💊 Welcome to Supplement Advisor</h2>
                    <p style="color:#888; margin-bottom:2rem;">
                        Ask me anything about vitamins and supplements.
                    </p>
                    <hr style="margin:2rem auto; width:60%; border:none; border-top:1px solid #333;">
                    <p style="color:#666; font-size:0.8rem;">
                        Upload your lab results in the <b>Upload</b> tab for personalized recommendations.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        for msg in st.session_state.chat_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            with st.chat_message(role):
                if role == "assistant":
                    st.markdown(_linkify_citations(content), unsafe_allow_html=True)
                    _render_citations(msg.get("citations", []))
                else:
                    st.markdown(content)

    # Input form pinned below the messages area.
    with st.form(key="chat_form", clear_on_submit=True):
        cols = st.columns([6, 1])
        with cols[0]:
            prompt = st.text_input(
                "Message",
                placeholder="Ask about vitamins & supplements...",
                label_visibility="collapsed",
            )
        with cols[1]:
            submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        payload = {
                            "user_id": user_id,
                            "message": prompt,
                            "conversation_history": [],
                        }
                        if st.session_state.active_conversation_id:
                            payload["conversation_id"] = st.session_state.active_conversation_id

                        resp = requests.post(
                            f"{api_base}/chat",
                            json=payload,
                            timeout=60,
                        )
                        resp.raise_for_status()
                        data = resp.json()

                        reply = data.get("reply", "")
                        citations = data.get("citations", [])
                        links = data.get("purchase_links", [])

                        # Track conversation_id from response.
                        if data.get("conversation_id"):
                            st.session_state.active_conversation_id = data["conversation_id"]

                        st.markdown(_linkify_citations(reply), unsafe_allow_html=True)
                        _render_citations(citations)

                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": reply,
                            "citations": citations,
                            "purchase_links": links,
                        })

                    except requests.exceptions.ConnectionError:
                        st.error("Cannot connect to the API. Is the server running?")
                    except requests.exceptions.HTTPError as e:
                        st.error(f"API error: {e.response.status_code} - {e.response.text}")

# =====================================================================
# Tab 2: Upload Documents
# =====================================================================

with tab_upload:
    st.header("Upload a Document")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt"],
        key="file_uploader",
    )

    if uploaded_file is not None and st.button("Upload"):
        with st.spinner("Processing document..."):
            try:
                resp = requests.post(
                    f"{api_base}/upload",
                    data={"user_id": user_id},
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                classification = data["classification"]

                if classification == "ambiguous":
                    st.warning("The system couldn't determine the document type.")
                    st.session_state.pending_upload = data
                else:
                    st.success(
                        f"Document uploaded as **{classification}**.  \n"
                        f"Created **{data['chunks_created']}** chunks."
                    )

            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the API. Is the server running?")
            except requests.exceptions.HTTPError as e:
                st.error(f"Upload failed: {e.response.status_code} - {e.response.text}")

    # Handle pending AMBIGUOUS classification.
    if st.session_state.pending_upload is not None:
        pending = st.session_state.pending_upload
        st.info(
            f"**{pending['filename']}** needs manual classification "
            f"(doc_id: `{pending['doc_id']}`)."
        )

        choice = st.radio(
            "Classify this document as:",
            ["General Knowledge", "Personal Document"],
            key="ambiguous_choice",
        )
        classification = "general" if choice == "General Knowledge" else "personal"

        if st.button("Confirm Classification"):
            with st.spinner("Ingesting..."):
                try:
                    resp = requests.post(
                        f"{api_base}/upload/confirm",
                        json={
                            "user_id": user_id,
                            "doc_id": pending["doc_id"],
                            "filename": pending["filename"],
                            "classification": classification,
                            "chunks": pending.get("chunks", []),
                        },
                        timeout=120,
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    st.success(
                        f"Document saved as **{classification}**.  \n"
                        f"Created **{result['chunks_created']}** chunks."
                    )
                    st.session_state.pending_upload = None
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to the API.")
                except requests.exceptions.HTTPError as e:
                    st.error(f"Confirm failed: {e.response.status_code}")

# =====================================================================
# Tab 3: Personal Documents
# =====================================================================

with tab_personal:
    st.header("Personal Documents")

    try:
        resp = requests.get(
            f"{api_base}/documents/{user_id}",
            timeout=10,
        )
        resp.raise_for_status()
        docs = resp.json()

        if not docs:
            st.info("No personal documents yet. Upload one in the Upload tab.")
        else:
            for doc in docs:
                with st.expander(doc["filename"]):
                    _render_document(
                        api_base, doc["doc_id"], doc["filename"],
                        collection_type="personal", user_id=user_id,
                    )

                    if st.button("Delete", key=f"del_{doc['doc_id']}"):
                        requests.delete(
                            f"{api_base}/documents/{user_id}/{doc['doc_id']}",
                            timeout=10,
                        )
                        st.rerun()

    except requests.exceptions.ConnectionError:
        st.warning("Cannot connect to the API. Is the server running?")
    except requests.exceptions.HTTPError as e:
        st.error(f"Failed to load documents: {e.response.status_code}")

# =====================================================================
# Tab 4: General Documents
# =====================================================================

with tab_general:
    st.header("General Documents")

    try:
        resp = requests.get(
            f"{api_base}/documents/general",
            timeout=10,
        )
        resp.raise_for_status()
        docs = resp.json()

        if not docs:
            st.info("No general knowledge documents yet.")
        else:
            for doc in docs:
                with st.expander(doc["filename"]):
                    _render_document(
                        api_base, doc["doc_id"], doc["filename"],
                        collection_type="general",
                    )

    except requests.exceptions.ConnectionError:
        st.warning("Cannot connect to the API. Is the server running?")
    except requests.exceptions.HTTPError as e:
        st.error(f"Failed to load documents: {e.response.status_code}")
