# AI Writing Assistant API ✍️

A production-ready AI-powered writing assistant built with Django REST Framework and Google Gemini. Features 6 AI tools including text improvement, email generation, summarization, blog post generation, document Q&A (RAG), and chat with memory.

**Live API:** `https://aiwritingassistant-production.up.railway.app`

---

## Features

- **Text Improvement** — Rewrite text in professional, casual, or friendly tone
- **Email Generator** — Convert bullet points into polished emails
- **Summarizer** — Condense long text into short, medium, or long summaries
- **Blog Post Generator** — Generate full 500+ word blog posts from a topic
- **Document Q&A (RAG)** — Upload PDF/TXT and ask questions about the content
- **Chat with Memory** — Ongoing conversations with full history persistence

---

## Tech Stack

- Python 3.12 / Django 6.x
- Django REST Framework
- Google Gemini 2.5 Flash (AI)
- LangChain + LangChain Google GenAI
- ChromaDB (vector database for RAG)
- PostgreSQL (production) / SQLite (development)
- Token Authentication
- Deployed on Railway

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

Create a `.env` file in the same folder as `manage.py`:

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

---

## Project Structure

```
ai-writing-assistant/
├── config/
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── writing_assistant/
│   ├── models.py        # AIRequest, Document, Conversation, Message
│   ├── serializers.py   # DRF serializers for all features
│   ├── services.py      # AI logic (Gemini, LangChain, ChromaDB)
│   ├── views.py         # API endpoints
│   └── urls.py          # URL routing
├── manage.py
├── requirements.txt
├── Procfile             # Railway deployment
├── runtime.txt          # Python version
└── .env                 # Never commit this
```

---

## API Reference

### Authentication

All endpoints (except register/login) require a token header:

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
{
    "username": "john",
    "password": "securepass123"
}
```

#### Logout
```
POST /api/auth/logout/
```

---

### Feature 1 — Text Improvement
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

---

### Feature 2 — Email Generator
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

Response:
```json
{
    "email": "Subject: Project Update\n\nDear [Name],\n\n...",
    "tone": "professional",
    "requests_today": 2,
    "daily_limit": 50
}
```

---

### Feature 3 — Summarizer
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

Response:
```json
{
    "summary": "...",
    "length": "short",
    "original_length": 5420,
    "summary_length": 180,
    "requests_today": 3,
    "daily_limit": 50
}
```

---

### Feature 4 — Blog Post Generator
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

Response:
```json
{
    "blog_post": "# Title\n\n## Introduction\n\n...",
    "topic": "Why Django is great for beginners",
    "word_count": 835,
    "requests_today": 4,
    "daily_limit": 50
}
```

---

### Feature 5 — Document Q&A (RAG)

#### Upload Document
```
POST /api/docs/
Content-Type: multipart/form-data
```
Form fields:
- `file` — PDF or TXT file
- `title` — document title (optional)

Response `201 Created`:
```json
{
    "id": 1,
    "title": "My Document",
    "status": "ready",
    "chunk_count": 42,
    "uploaded_at": "2026-03-20T09:00:00Z"
}
```

#### List Documents
```
GET /api/docs/
```

#### Ask Question
```
POST /api/docs/ask/
```
```json
{
    "document_id": 1,
    "question": "What is the main topic of this document?"
}
```
Response:
```json
{
    "question": "What is the main topic?",
    "answer": "The document discusses...",
    "sources": [
        {
            "text": "...relevant chunk preview...",
            "page": 2,
            "score": 0.8821
        }
    ],
    "document": "My Document"
}
```

#### Delete Document
```
DELETE /api/docs/1/
```
Returns `204 No Content`

---

### Feature 6 — Chat with Memory

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

Send follow-up (AI remembers context):
```json
{
    "message": "What is my name?",
    "conversation_id": 1
}
```

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

---

### Usage Stats
```
GET /api/usage/
```
Response:
```json
{
    "total_requests": 6,
    "by_feature": {
        "improve": 1,
        "email": 1,
        "summarize": 1,
        "blog": 1,
        "chat": 2
    },
    "recent": [...]
}
```

---

## Rate Limits

- 100 requests per hour (DRF throttling)
- 50 AI requests per day per user (quota system — resets at midnight UTC)

---

## Status Codes

| Code | Meaning |
|------|---------|
| 200  | Success |
| 201  | Created successfully |
| 204  | Deleted successfully |
| 400  | Validation error |
| 401  | Missing or invalid token |
| 429  | Daily quota exceeded |
| 500  | Server error |

---

## Deployment (Railway)

1. Push to GitHub
2. Connect repo to Railway
3. Add PostgreSQL database
4. Set environment variables:
   ```
   SECRET_KEY     = (generate new key)
   DEBUG          = False
   GEMINI_API_KEY = your-key
   DB_URL         = (auto-set by Railway PostgreSQL)
   ```
5. Set Pre-deploy Command: `python manage.py migrate`
6. Set Start Command: `gunicorn config.wsgi`

Generate a new secret key:
```bash
python -c "from django.core.management.utils import get_random_secret_key; print(get_random_secret_key())"
```

---

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SECRET_KEY` | Django secret key | Yes |
| `DEBUG` | True for dev, False for prod | Yes |
| `GEMINI_API_KEY` | Google Gemini API key | Yes |
| `DB_URL` | PostgreSQL connection URL | Production only |

---

## License

MIT
