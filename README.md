# Intelligent Document Q&A System (RAG + Memory + Gemini)

A lightweight, beginner-friendly RAG application: upload PDFs, ask questions, get cited answers, and the system remembers your conversation.

## Architecture

```
┌─────────────────┐      HTTP       ┌──────────────────────┐
│  Streamlit UI   │ ───────────────▶│   FastAPI backend    │
│  frontend/app.py│                 │  /upload /ask /...   │
└─────────────────┘                 └──────────┬───────────┘
                                               │
              ┌────────────────────────────────┼──────────────────────────────┐
              ▼                                ▼                              ▼
   ┌────────────────────┐         ┌─────────────────────┐         ┌────────────────────┐
   │   RAG pipeline     │         │  Memory (SQLite)    │         │  Google Gemini     │
   │ (rag/*)            │         │  - chat_messages    │         │  - gemini-1.5-flash│
   │  PDF → chunks →    │         │  - qa_memory        │         │  - text-embedding  │
   │  embeddings →      │         │  - documents        │         │    -004            │
   │  ChromaDB          │         └─────────────────────┘         └────────────────────┘
   └────────┬───────────┘
            ▼
   ┌────────────────────┐
   │   ChromaDB (disk)  │
   │   chroma_db/       │
   └────────────────────┘
```

### Memory model
- **Short-term** — last 5 messages from `chat_messages`, passed to Gemini for follow-ups.
- **Long-term** — `qa_memory` table: question, answer, feedback score, correction, confidence, timestamp.
- **Episodic** — full ordered history per `session_id`, viewable from the sidebar.

## Project structure

```
Intelligent_doc/
├── backend/
│   ├── main.py              # FastAPI app entry
│   ├── routes.py            # /upload /ask /feedback /history /documents
│   ├── gemini_service.py    # Gemini gen + embeddings + summarize
│   ├── memory.py            # short/long/episodic memory helpers
│   ├── database.py          # SQLAlchemy engine + session
│   └── models.py            # ORM models
│
├── rag/
│   ├── ingest.py            # PDF → text → chunks → Chroma
│   ├── chunking.py          # 800/150 char splitter
│   ├── embeddings.py        # Chroma persistent client
│   └── retrieval.py         # top-k vector search
│
├── frontend/
│   └── app.py               # Streamlit UI
│
├── uploads/                 # saved PDFs
├── chroma_db/               # vector store (auto-created)
├── memory.db                # SQLite (auto-created)
├── requirements.txt
├── .env                     # GEMINI_API_KEY
└── README.md
```

## Setup

### 1. Prerequisites
- Python 3.11+
- A Gemini API key — get one at https://aistudio.google.com/apikey

### 2. Install
```bash
# from the project root
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure the API key
Edit `.env` and set:
```
GEMINI_API_KEY=your_key_here
```

### 4. Run the backend
```bash
uvicorn backend.main:app --reload
```
The API will be available at http://localhost:8000 (docs at http://localhost:8000/docs).

### 5. Run the frontend (in a second terminal)
```bash
streamlit run frontend/app.py
```
Open http://localhost:8501.

## How to use

1. **Upload** a PDF in the sidebar and click "Ingest document". You'll see a chunk count and an auto-generated summary.
2. **Ask** a question in the chat box. The answer includes a confidence score, sources (filename + chunk preview), and suggested follow-ups.
3. **Follow up** — ask "Explain that more simply" and the system uses the last 5 turns for context.
4. **Feedback** — click 👍 or 👎 on any answer. 👎 lets you supply a correction; both go to SQLite.
5. **New session** starts a fresh `session_id` (the memory namespace).

## API reference

| Method | Path                     | Body                                                     |
|--------|--------------------------|----------------------------------------------------------|
| POST   | `/upload`                | multipart `file=<pdf>`                                   |
| POST   | `/ask`                   | `{ "session_id": "abc", "question": "..." }`             |
| POST   | `/feedback`              | `{ "question": "...", "answer": "...", "rating": 1 \| -1, "correction": "..." }` |
| GET    | `/history/{session_id}`  | —                                                        |
| GET    | `/documents`             | —                                                        |

## UI overview (text mock)

```
┌────────────────────────────────┬──────────────────────────────────────────────┐
│  📄 Document Q&A               │  Ask your documents                          │
│  RAG + Memory + Gemini         │  ─────────────────────────────────────────── │
│                                │                                              │
│  Upload a PDF                  │   USER: What is RAG?                         │
│  [drop zone]                   │                                              │
│  [Ingest document]             │   ASSISTANT: Retrieval-Augmented Generation  │
│                                │   combines a retriever with a generator...   │
│  Session                       │   Confidence: 87/100                         │
│  abc123ef456                   │   ▸ Sources (3)                              │
│  [New session]                 │   👍 👎    Suggested:                        │
│                                │     • How does the retriever work?           │
│  Uploaded this session         │     • Compare RAG to fine-tuning             │
│  • paper.pdf (42 chunks)       │     • What is a vector embedding?            │
│                                │                                              │
│  [View full history]           │   [type your question here ...........]     │
└────────────────────────────────┴──────────────────────────────────────────────┘
```

## Troubleshooting

- **"GEMINI_API_KEY is missing"** — set the key in `.env` and restart uvicorn.
- **"No extractable text found"** — the PDF is image-only; OCR is not included in this MVP.
- **"Backend unreachable"** — make sure uvicorn is running on port 8000.
- **Wiped state** — delete `memory.db` and `chroma_db/` to fully reset.

## Bonus features included
- PDF summarization on upload
- Confidence score per answer
- Suggested follow-up questions
