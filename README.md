# AI Writing Assistant API ✍️

A production-ready AI-powered writing assistant built with Django REST Framework and Google Gemini. Features 6 AI writing tools plus a production-grade RAG (Retrieval Augmented Generation) system with pgvector, hybrid search, citations, and confidence scoring.

**Live API:** `https://aiwritingassistant-production.up.railway.app`

---

## Features

### Writing Tools
- **Text Improvement** — Rewrite text in professional, casual, or friendly tone
- **Email Generator** — Convert bullet points into polished emails
- **Summarizer** — Condense long text into short, medium, or long summaries
- **Blog Post Generator** — Generate full 500+ word blog posts from a topic

### Document Intelligence (RAG)
- **Multi-format Support** — PDF, DOCX, TXT, MD, CSV, Excel files
- **Smart Chunking** — Content-aware splitting (prose, markdown, code, structured data)
- **pgvector Search** — Vector embeddings stored in PostgreSQL
- **Hybrid Search** — Combines semantic vector search with keyword search
- **Re-ranking** — LLM re-ranks retrieved chunks for better accuracy
- **Citations** — Every answer shows exactly which part of the document it came from
- **Confidence Scores** — 0-100% score indicating answer reliability
- **Hallucination Detection** — Flags answers not grounded in source material
- **Query Rewriting** — Improves vague questions before searching
- **Follow-up Questions** — AI suggests 3 related questions after each answer
- **Cross-document Search** — Search across all your documents at once
- **Image Descriptions** — Gemini Vision describes images found in PDFs

### Chat
- **Persistent Conversations** — Full history saved to PostgreSQL
- **Smart Memory** — Summarizes old messages, keeps recent ones in full
- **Conversation Stats** — See token usage and memory strategy per conversation

### Production Features
- Token-based authentication
- Per-user daily quotas (50 requests/day)
- Response caching (MD5-based, 1 hour TTL)
- Embedding caching (24 hour TTL)
- Retry logic with exponential backoff
- Rate limiting (100 requests/hour)
- Token budget management

---

## Tech Stack

- Python 3.12 / Django 6.x
- Django REST Framework
- Google Gemini 2.5 Flash (AI generation)
- Google Gemini Embedding 001 (3072-dim embeddings)
- LangChain + LangChain Google GenAI
- pgvector (vector search inside PostgreSQL)
- PostgreSQL (production) / SQLite (development)
- Railway (deployment)

---

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-writing-assistant.git
cd ai-writing-assistant
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create .env file

Create `.env` in the same folder as `manage.py`:

```
SECRET_KEY=your-django-secret-key-here
DEBUG=True
DATABASE_URL=sqlite:///db.sqlite3
GEMINI_API_KEY=your-gemini-api-key-here
```

Get a free Gemini API key at: https://aistudio.google.com

### 5. Run migrations and start server

```bash
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

API available at: `http://localhost:8000/api/`

> **Note:** pgvector features require PostgreSQL with the vector extension enabled. For local development without PostgreSQL, document uploads will fall back gracefully.

---

## Project Structure

```
ai-writing-assistant/
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── writing_assistant/
│   ├── models.py          # AIRequest, Document, DocumentChunk, Conversation, Message
│   ├── serializers.py     # DRF serializers
│   ├── services.py        # Writing AI features (improve, email, blog, etc.)
│   ├── rag_service.py     # RAG pipeline (pgvector, hybrid search, citations)
│   ├── chunker.py         # Smart document chunking for all file types
│   ├── table_processor.py # Excel and CSV processing
│   ├── image_processor.py # Gemini Vision image descriptions
│   ├── memory.py          # Conversation history management
│   ├── views.py           # API endpoints
│   └── urls.py            # URL routing
├── manage.py
├── requirements.txt
├── Procfile
├── runtime.txt
└── .env                   # Never commit this
```

---

## API Reference

### Authentication

All endpoints (except register/login) require:
```
Authorization: Token your_token_here
```

---

### Auth Endpoints

#### Register
```
POST /api/auth/register/
```
```json
{
    "username": "john",
    "password": "securepass123",
    "email": "john@example.com"
}
```
Response `201 Created`:
```json
{
    "token": "9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b",
    "username": "john",
    "message": "Account created successfully"
}
```

#### Login
```
POST /api/auth/login/
```
```json
{ "username": "john", "password": "securepass123" }
```

#### Logout
```
POST /api/auth/logout/
```

---

### Writing Features

#### Text Improvement
```
POST /api/improve/
```
```json
{
    "text": "the meeting was bad and things went wrong",
    "tone": "professional"
}
```
Tone options: `professional` | `casual` | `friendly`

Response:
```json
{
    "original": "the meeting was bad and things went wrong",
    "improved": "The meeting was unproductive, and several issues arose.",
    "tone": "professional",
    "requests_today": 1,
    "daily_limit": 50
}
```

#### Email Generator
```
POST /api/email/
```
```json
{
    "bullets": "- project delayed\n- need 3 more days\n- will update daily",
    "recipient_context": "writing to my manager",
    "tone": "professional"
}
```
Tone options: `professional` | `formal` | `friendly`

#### Summarizer
```
POST /api/summarize/
```
```json
{
    "text": "paste long article or document here...",
    "length": "short"
}
```
Length options: `short` | `medium` | `long`

Response includes `original_length` and `summary_length` for comparison.

#### Blog Post Generator
```
POST /api/blog/
```
```json
{
    "topic": "Why Django is great for beginners",
    "keywords": "Python, web framework, REST API",
    "tone": "informative"
}
```
Tone options: `informative` | `conversational` | `persuasive`

Response includes `word_count`.

---

### Document Q&A (RAG)

#### Upload Document
```
POST /api/docs/
Content-Type: multipart/form-data
```
Form fields:
- `file` — PDF, DOCX, TXT, MD, CSV, or XLSX file
- `title` — document title (optional, defaults to filename)

Response `201 Created`:
```json
{
    "id": 1,
    "title": "My Document",
    "status": "ready",
    "chunk_count": 42,
    "uploaded_at": "2026-03-21T09:00:00Z"
}
```

#### List Documents
```
GET /api/docs/
```

#### Ask Question (Basic)
```
POST /api/docs/ask/
```
```json
{
    "document_id": 1,
    "question": "What is the main topic?"
}
```
Response:
```json
{
    "question": "What is the main topic?",
    "answer": "The document discusses...",
    "sources": [
        { "text": "...chunk preview...", "page": 2, "distance": 0.18 }
    ],
    "document": "My Document"
}
```

#### Ask Question V2 (with Citations)
```
POST /api/docs/ask/v2/
```
```json
{
    "document_id": 1,
    "question": "What is the notice period?"
}
```
Response:
```json
{
    "question": "What is the notice period?",
    "rewritten_question": null,
    "answer": "The notice period is 30 days [1].",
    "confidence": 88.5,
    "confidence_label": "High",
    "citations": [
        {
            "citation_number": 1,
            "document": "Employment Contract",
            "page": 4,
            "relevance_score": 88.5,
            "snippet": "Employee shall provide 30 days written notice...",
            "chunk_type": "text"
        }
    ],
    "follow_up_questions": [
        "What happens if the notice period is not served?",
        "Is there a different notice period for probation?",
        "Can the notice period be waived?"
    ],
    "hallucination_check": {
        "hallucination_risk": "Low",
        "grounded": true,
        "note": "Answer appears grounded in source material"
    },
    "document": "Employment Contract"
}
```

#### Cross-Document Search
```
POST /api/docs/search/
```
```json
{ "question": "What are the payment terms?" }
```
Searches across ALL your uploaded documents at once.

#### Rewrite Question (Preview)
```
POST /api/docs/rewrite/
```
```json
{ "question": "tell me more about it" }
```
Response:
```json
{
    "original": "tell me more about it",
    "rewritten": "What are the key details of the main topic?",
    "was_rewritten": true
}
```

#### Delete Document
```
DELETE /api/docs/1/
```
Returns `204 No Content`. Also removes all vector embeddings.

---

### Chat with Memory

#### Send Message
```
POST /api/chat/
```
```json
{
    "message": "Hello, my name is Palash",
    "conversation_id": null
}
```
Response:
```json
{
    "response": "Hello Palash! How can I help you today?",
    "conversation_id": 1,
    "conversation_title": "Hello, my name is Palash",
    "requests_today": 5,
    "daily_limit": 50
}
```

Memory strategy:
- ≤10 messages → full history sent
- 11-20 messages → sliding window (last 10 kept)
- >20 messages → old messages summarized, recent kept in full

#### List Conversations
```
GET /api/conversations/
```

#### Get Conversation + Full History
```
GET /api/conversations/1/
```

#### Clear Conversation
```
DELETE /api/conversations/1/
```

#### Conversation Memory Stats
```
GET /api/conversations/1/stats/
```
Response:
```json
{
    "total_messages": 25,
    "total_chars": 8420,
    "approximate_tokens": 2105,
    "would_summarize": true,
    "strategy": "summarize_and_recent"
}
```

---

### Usage Stats
```
GET /api/usage/
```
Response:
```json
{
    "total_requests": 47,
    "by_feature": {
        "improve": 12,
        "email": 8,
        "summarize": 15,
        "blog": 4,
        "doc_qa": 6,
        "chat": 2
    },
    "recent": [...]
}
```

---

## Rate Limits

| Limit | Value |
|-------|-------|
| Requests per hour | 100 (DRF throttling) |
| AI requests per day | 50 per user (resets midnight UTC) |
| Max text for improvement | 5,000 characters |
| Max text for summarization | 25,000 characters |
| RAG context budget | ~2,000 tokens per query |

---

## Confidence Score Guide

| Score | Label | Meaning |
|-------|-------|---------|
| 85-100% | High | Answer directly supported by document |
| 60-84% | Medium | Answer likely correct, good context found |
| 35-59% | Low | Partial match, verify manually |
| 0-34% | Very Low | Answer may not be in document |

---

## Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created successfully |
| 204 | Deleted successfully |
| 400 | Validation error |
| 401 | Missing or invalid token |
| 429 | Daily quota or hourly rate limit exceeded |
| 500 | Server error |

---

## Supported File Types

| Extension | Type | Notes |
|-----------|------|-------|
| `.pdf` | PDF | Text + image descriptions via Gemini Vision |
| `.docx` | Word Document | Paragraphs and tables extracted |
| `.txt` | Plain Text | Full text |
| `.md` | Markdown | Header-aware chunking |
| `.csv` | CSV | Row-by-row structured chunking |
| `.xlsx` / `.xls` | Excel | Multi-sheet support, row-by-row chunking |
| `.py` | Python | Code-aware chunking (function/class boundaries) |
| `.js` / `.ts` | JavaScript/TypeScript | Code-aware chunking |

---

## Deployment (Railway)

### Prerequisites
- Railway account
- GitHub repository
- Google Gemini API key

### Steps

1. Push to GitHub
2. Railway → New Project → Deploy from GitHub repo
3. Add PostgreSQL: **+ New → Database → PostgreSQL**
4. Enable pgvector in Railway PostgreSQL Query tab:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
5. Set environment variables in Railway → your Django service → Variables:
   ```
   SECRET_KEY     = (generate new — see below)
   DEBUG          = False
   GEMINI_API_KEY = your-key
   DB_URL         = ${{Postgres.DATABASE_URL}}
   ```
6. Settings → Pre-deploy Command:
   ```
   python manage.py migrate
   ```
7. Settings → Start Command:
   ```
   gunicorn config.wsgi
   ```

Generate a new secret key:
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SECRET_KEY` | Django secret key | Yes |
| `DEBUG` | `True` for dev, `False` for prod | Yes |
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `DB_URL` | PostgreSQL connection URL | Production only |

---

## License

MIT
