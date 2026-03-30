import os
import time
import logging
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from decouple import config
from django.conf import settings
import hashlib
from django.core.cache import cache
from .memory import build_history_for_llm

logger = logging.getLogger(__name__)

# ── Initialise AI components once ─────────────────────────────

def get_llm(temperature=0.7):
    return ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        google_api_key=config('GEMINI_API_KEY'),
        temperature=temperature,
        max_retries=3,          # LangChain retries automatically
        timeout=30,             # fail after 30 seconds, don't hang forever
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model='models/gemini-embedding-001',
        google_api_key=config('GEMINI_API_KEY')
    )

parser = StrOutputParser()


def _run(prompt_template, variables, temperature=0.7, max_attempts=3):
    """
    Run a chain with exponential backoff retry.
    If Gemini returns a rate limit or server error, wait and try again.

    Exponential backoff means:
    Attempt 1 fails → wait 2 seconds
    Attempt 2 fails → wait 4 seconds
    Attempt 3 fails → raise the error
    """
    last_error = None

    for attempt in range(max_attempts):
        try:
            chain = prompt_template | get_llm(temperature) | parser
            return chain.invoke(variables)

        except Exception as e:
            last_error = e
            error_str = str(e).lower()

            # Rate limit — wait longer
            if '429' in error_str or 'quota' in error_str or 'rate' in error_str:
                wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                logger.warning(f"Rate limit hit. Waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)

            # Server error — wait briefly
            elif '500' in error_str or '503' in error_str:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                logger.warning(f"Server error. Waiting {wait_time}s before retry {attempt + 1}")
                time.sleep(wait_time)

            # Other errors — don't retry
            else:
                logger.error(f"Non-retryable error: {e}")
                raise

    logger.error(f"All {max_attempts} attempts failed. Last error: {last_error}")
    raise last_error


# ── Feature 1: Text Improvement ───────────────────────────────

IMPROVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert writing editor.
Improve the given text to make it {tone}.

Rules:
- Keep the original meaning intact
- Fix grammar, spelling, and punctuation
- Improve clarity and flow
- Match the requested tone exactly
- Return ONLY the improved text, no explanations"""),
    ("user", "{text}")
])


MAX_CHARS = 25000  # roughly 6000 tokens — safe limit for a single request

def truncate_to_limit(text, max_chars=MAX_CHARS):
    """
    Truncate text to stay within token limits.
    Adds a note if truncation happened.
    """
    if len(text) <= max_chars:
        return text

    truncated = text[:max_chars]
    logger.warning(f"Text truncated from {len(text)} to {max_chars} chars")
    return truncated + "\n\n[Note: Text was truncated to fit within processing limits]"


# Update improve_text to use it
def improve_text(text, tone='professional'):
    safe_text = truncate_to_limit(text, max_chars=5000)  # keep improvements short
    return _run(IMPROVE_PROMPT, {"text": safe_text, "tone": tone}, temperature=0.4)



# ── Feature 2: Email Generator ────────────────────────────────

EMAIL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a professional email writer.
Convert bullet points into a complete, {tone} email.

Always include:
- Subject line (format: Subject: ...)
- Appropriate greeting
- Clear body paragraphs
- Professional closing

Example input:
- meeting postponed
- need to reschedule to next week
- apologise for inconvenience

Example output:
Subject: Meeting Reschedule Request

Dear [Name],

I hope this message finds you well. I am writing to inform you that 
our upcoming meeting will need to be postponed.

I would like to reschedule to sometime next week at your convenience.
I sincerely apologise for any inconvenience this may cause.

Please let me know your availability and I will arrange accordingly.

Best regards,
[Your Name]

{recipient_context}"""),
    ("user", "{bullets}")
])


def generate_email(bullets, recipient_context='', tone='professional'):
    context_str = f"\nRecipient context: {recipient_context}" if recipient_context else ""
    return _run(
        EMAIL_PROMPT,
        {"bullets": bullets, "tone": tone, "recipient_context": context_str},
        temperature=0.5
    )


# ── Feature 3: Summarizer ─────────────────────────────────────

LENGTH_INSTRUCTIONS = {
    'short': '2-3 sentences maximum',
    'medium': '1 paragraph (5-7 sentences)',
    'long': '3-4 paragraphs covering all key points',
}

SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a precise text summarizer.
Summarize the provided text in {length_instruction}.

Rules:
- Capture the most important points
- Use clear, simple language
- Do not add information not in the original
- Return ONLY the summary"""),
    ("user", "{text}")
])


def summarize_text(text, length='medium'):
    safe_text = truncate_to_limit(text)  # allow longer text for summaries
    text_hash = hashlib.md5(f"{safe_text}{length}".encode()).hexdigest()
    cache_key = f"summary_{text_hash}"

    cached = cache.get(cache_key)
    if cached:
        return cached

    result = _run(
        SUMMARIZE_PROMPT,
        {"text": safe_text, "length_instruction": LENGTH_INSTRUCTIONS[length]},
        temperature=0.3
    )
    cache.set(cache_key, result, timeout=3600)
    return result


# ── Feature 4: Blog Post Generator ───────────────────────────

BLOG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert blog writer.
Write a complete, {tone} blog post about the given topic.

Structure the post with:
# Title

## Introduction
(engaging opening paragraph)

## [Main Section 1]
(detailed content)

## [Main Section 2]
(detailed content)

## [Main Section 3]
(detailed content)

## Conclusion
(summary and call to action)

Use the provided keywords naturally throughout the post.
Write at least 500 words."""),
    ("user", "Topic: {topic}\nKeywords to include: {keywords}")
])


def generate_blog(topic, keywords='', tone='informative'):
    return _run(
        BLOG_PROMPT,
        {"topic": topic, "keywords": keywords, "tone": tone},
        temperature=0.8   # higher temp for creative writing
    )


# ── Feature 5: Document RAG ───────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document assistant.
Answer questions based ONLY on the provided document context.

Rules:
- Only use information from the context
- If the answer is not in the context, say "I couldn't find that in the document"
- Be concise and accurate
- Cite which part of the document your answer comes from"""),
    ("user", """Context:
{context}

Question: {question}""")
])


def process_document(document):
    """Extract, chunk, embed, and store a document in ChromaDB"""
    file_path = document.file.path
    ext = os.path.splitext(file_path)[1].lower()

    # Extract text
    if ext == '.pdf':
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({'text': text, 'page': i + 1})
    elif ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            pages = [{'text': f.read(), 'page': 1}]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not pages:
        raise ValueError("No text could be extracted from this document")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks, metadatas = [], []

    for page in pages:
        for i, chunk in enumerate(splitter.split_text(page['text'])):
            chunks.append(chunk)
            metadatas.append({
                'document_id': str(document.id),
                'title': document.title,
                'page': page['page'],
                'chunk_index': i,
            })

    # Store in ChromaDB
    vectorstore = Chroma(
        collection_name=f"doc_{document.id}",
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_DIR
    )
    vectorstore.add_texts(texts=chunks, metadatas=metadatas)

    return len(chunks)


def answer_document_question(document, question):
    """RAG: find relevant chunks → ask AI → return answer + sources"""
    vectorstore = Chroma(
        collection_name=f"doc_{document.id}",
        embedding_function=get_embeddings(),
        persist_directory=settings.CHROMA_DIR
    )

    results = vectorstore.similarity_search_with_score(question, k=4)

    if not results:
        return {'answer': "No relevant content found.", 'sources': []}

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    sources = [
        {
            'text': doc.page_content[:150] + "...",
            'page': doc.metadata.get('page'),
            'score': round(float(score), 4),
        }
        for doc, score in results
    ]

    chain = RAG_PROMPT | get_llm() | parser
    answer = chain.invoke({"context": context, "question": question})

    return {'answer': answer, 'sources': sources}


def delete_document_vectors(document_id):
    try:
        Chroma(
            collection_name=f"doc_{document_id}",
            embedding_function=get_embeddings(),
            persist_directory=settings.CHROMA_DIR
        ).delete_collection()
    except Exception:
        pass


# ── Feature 6: Chat with Memory ───────────────────────────────

CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, friendly AI assistant. You remember the full conversation."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{message}")
])


def chat_response(conversation, user_message):
    """
    Send a message with smart history management.
    Uses summarization for long conversations.
    """
    # Build history using smart memory management
    # instead of loading ALL messages every time
    history = build_history_for_llm(conversation)

    chain = CHAT_PROMPT | get_llm() | parser
    response = chain.invoke({
        "history": history,
        "message": user_message
    })

    # Save both messages to database
    from .models import Message
    Message.objects.create(
        conversation=conversation,
        role='user',
        content=user_message
    )
    Message.objects.create(
        conversation=conversation,
        role='assistant',
        content=response
    )

    if conversation.title == 'New Conversation':
        conversation.title = user_message[:60]
        conversation.save()

    return response