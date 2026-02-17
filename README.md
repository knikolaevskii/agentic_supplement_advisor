# Agentic Supplement Advisor
![Demo](./demo/demo.gif)

An AI-powered vitamin and supplement advisor that combines RAG over a general knowledge base with personal health data analysis. Users can ask health questions, upload personal documents (lab results, prescriptions), and search for purchase links — all routed through an intent-classifying agent pipeline.

Built with FastAPI, LangGraph, Streamlit, and ChromaDB.

## Features

- **General supplement knowledge Q&A** with inline citations
- **Personal lab results analysis** and personalized recommendations
- **Combined general + personal** health insights in a single answer
- **Purchase link search** with location awareness (Tavily)
- **Knowledge gap detection** — tells you what information is missing and suggests uploading documents
- **Multi-conversation chat** with persistent memory across turns
- **Document upload** with automatic classification (general vs personal)
- **Cross-user data isolation** — personal documents are stored in per-user collections
- **Strict relevance filtering** — only topic-specific chunks are used in answers

## Tech Stack

- **Backend:** Python, FastAPI
- **Agent orchestration:** LangGraph (StateGraph with conditional routing)
- **UI:** Streamlit
- **Vector store:** ChromaDB (separate general and per-user personal collections)
- **LLM:** OpenAI (configurable model via `LLM_MODEL` env var)
- **Web search:** Tavily (purchase links only — never used for medical evidence)
- **Tracing:** LangSmith (optional)

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd agentic_supplement_advisor
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### 3. Seed the general knowledge base

```bash
python -m scripts.seed_kb --general-only
```

### 4. Start the application

```bash
# Terminal 1 — API server
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run ui/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Project Structure

```
app/
├── main.py                  # FastAPI endpoints (/chat, /upload, /documents, /conversations)
├── config.py                # Pydantic Settings (.env)
├── agents/
│   ├── router.py            # Intent & document classification (LLM)
│   ├── nodes.py             # LangGraph node functions (rephrase, retrieve, filter, generate)
│   ├── chat_graph.py        # Chat agent StateGraph wiring
│   └── upload_graph.py      # Upload/ingest StateGraph
├── models/
│   └── schemas.py           # Pydantic request/response models
├── services/
│   ├── document.py          # PDF/TXT extraction & chunking
│   ├── vectorstore.py       # ChromaDB wrapper (general + personal KB)
│   └── tavily_client.py     # Purchase link search (Tavily)
└── utils/
    └── citations.py         # Citation formatting

ui/
└── streamlit_app.py         # Streamlit frontend

data/
└── seed_documents/
    ├── general/             # .txt/.pdf files for shared knowledge base
    └── personal/            # .txt/.pdf files for personal KB seeding

scripts/
├── seed_kb.py               # Seed knowledge base from data/seed_documents/
└── clear_kb.py              # Clear vector store collections

tests/
├── test_documents.py        # Document extraction & chunking
├── test_routing.py          # Intent & document classification (mocked LLM)
├── test_vectorstore.py      # ChromaDB operations & data isolation
└── test_e2e.py              # End-to-end graph flow tests
```

## Configuration

All configuration is via environment variables (or `.env` file):

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM calls |
| `TAVILY_API_KEY` | Yes | Tavily API key for purchase link search |
| `LLM_MODEL` | No | OpenAI model name (default: `gpt-4o-mini`) |
| `CHROMA_PERSIST_DIR` | No | ChromaDB storage path (default: `./data/chroma`) |
| `LANGSMITH_TRACING` | No | Enable LangSmith tracing (`true`/`false`) |
| `LANGSMITH_API_KEY` | No | LangSmith API key |
| `LANGSMITH_PROJECT` | No | LangSmith project name (default: `supplement-advisor`) |

## Scripts

**Seed the knowledge base:**

```bash
python -m scripts.seed_kb                                 # ingest both folders
python -m scripts.seed_kb --clear                         # wipe + re-ingest all
python -m scripts.seed_kb --force                         # re-ingest without wiping
python -m scripts.seed_kb --general-only                  # general folder only
python -m scripts.seed_kb --personal-only --user-id alice # personal folder for user "alice"
```

**Clear collections:**

```bash
python -m scripts.clear_kb --general                          # clear general KB
python -m scripts.clear_kb --personal --user-id default_user  # clear a user's personal KB
python -m scripts.clear_kb --all                              # clear everything
```

## Testing

```bash
python -m pytest tests/ -v
```

All 47 tests run without API keys — LLM calls are fully mocked.

## Architecture

### Intent Classification

Every user message is classified into one of five intents:

- **health_general** — general vitamin/supplement questions (e.g., "What are the benefits of Vitamin D?")
- **health_personal** — questions about the user's own health data (e.g., "What do my lab results show?")
- **health_combined** — questions needing both personal data and general knowledge (e.g., "Given my low Vitamin D, what supplements should I take?")
- **purchase** — buying/pricing queries (e.g., "Where can I buy Vitamin C?")
- **out_of_scope** — anything unrelated to health/supplements

The intent always reflects what the user asked for, not what data is available. Data availability is handled downstream by knowledge gap detection.

### RAG Pipeline (Health Intents)

Each health intent follows the same four-stage pipeline:

```
rephrase → retrieve → filter → generate
```

1. **Rephrase** — rewrites the user's message into a search-optimized query. Understands the classified intent to avoid unnecessary clarification (e.g., won't ask "which lab results?" when intent is health_personal).

2. **Retrieve** — searches the appropriate ChromaDB collection(s):
   - health_general: general KB only
   - health_personal: user's personal KB only
   - health_combined: both collections, results merged

3. **Filter** — strict LLM-based relevance filtering. Only keeps chunks that directly address the specific topic (a chunk about Iron is dropped if the question is about Selenium). After filtering, runs knowledge gap detection to determine if the surviving chunks are sufficient.

4. **Generate** — produces a cited answer based on three cases:
   - Sufficient info: full answer with inline citations
   - Partial info: answer what's possible, note the gap, suggest uploading more documents
   - No info: skip LLM, return a message directing the user to upload relevant documents

Citations are renumbered sequentially so the `[N]` references in the response text always match the Sources list exactly.

### Purchase Flow

```
rephrase → search (Tavily) → filter links → generate
```

Includes location awareness — asks for city and country to avoid ambiguous results, persists location across conversation turns.

### Upload Flow

```
extract → chunk → classify → ingest
```

Documents are automatically classified as general or personal. Ambiguous documents are returned to the UI for the user to confirm.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/chat` | Chat with the advisor |
| `POST` | `/upload` | Upload a document (PDF/TXT) |
| `POST` | `/upload/confirm` | Confirm ambiguous document classification |
| `GET` | `/documents/preview/{doc_id}` | Preview document text |
| `GET` | `/documents/general` | List general knowledge documents |
| `GET` | `/documents/{user_id}` | List user's personal documents |
| `DELETE` | `/documents/{user_id}/{doc_id}` | Delete a personal document |
| `GET` | `/conversations/{user_id}` | List user's conversations |
| `DELETE` | `/conversations/{user_id}/{id}` | Delete a conversation |
| `GET` | `/conversations/{user_id}/{id}/messages` | Get conversation messages |

## Key Design Decisions

- **Dual knowledge base:** General KB is shared across all users; personal KB uses per-user ChromaDB collections with strict isolation
- **Tavily is purchase-only:** Never used for medical evidence — all health answers come from the vector store
- **Intent reflects user intent:** The classifier reports what the user asked for, not what data exists. Knowledge gap detection handles missing data gracefully
- **Strict relevance filtering:** Chunks must directly address the specific topic asked about. Being about supplements in general is not enough
- **Graceful fallbacks:** LLM failures default to safe classifications; Tavily failures return empty results
- **No medical diagnoses:** System prompts explicitly forbid diagnostic claims and recommend consulting healthcare providers
