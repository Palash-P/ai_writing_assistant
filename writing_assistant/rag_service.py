# writing_assistant/rag_service.py
import os
import logging
from pypdf import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector.django import CosineDistance
from decouple import config
from django.conf import settings
from .chunker import process_file
from django.db.models import Q
import hashlib
from django.core.cache import cache

logger = logging.getLogger(__name__)


# ── AI Components ─────────────────────────────────────────────

def get_llm(temperature=0.1):
    return ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        google_api_key=config('GEMINI_API_KEY'),
        temperature=temperature,
        max_retries=3,
        timeout=30,
    )

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model='models/gemini-embedding-001',
        google_api_key=config('GEMINI_API_KEY')
    )

def get_cached_embedding(text, embeddings_model):
    """
    Cache embeddings to avoid redundant API calls.

    When is this useful?
    - Same question asked multiple times
    - Similar questions with same keywords
    - Document re-processing after minor edits

    Cache key = MD5 hash of the text (same text = same hash = same cache)
    Cache TTL = 24 hours (embeddings don't change)
    """
    cache_key = f"emb_{hashlib.md5(text.encode()).hexdigest()}"

    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(f"Embedding cache hit for: {text[:50]}...")
        return cached

    # Not cached — call API
    vector = embeddings_model.embed_query(text)
    cache.set(cache_key, vector, timeout=86400)  # cache for 24 hours
    return vector


parser = StrOutputParser()

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document assistant.
Answer questions based ONLY on the provided document context.
If the answer is not in the context, say "I couldn't find that in the document."
Always be concise and accurate."""),
    ("user", "Context:\n{context}\n\nQuestion: {question}")
])


# ── Text Extraction ───────────────────────────────────────────

def extract_text_from_file(file_path):
    """Extract text from PDF or TXT file, returns list of {text, page} dicts"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.pdf':
        reader = PdfReader(file_path)
        pages = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({'text': text, 'page': i + 1})
        return pages

    elif ext in ['.txt', '.md']:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [{'text': f.read(), 'page': 1}]

    else:
        raise ValueError(f"Unsupported file type: {ext}")


# ── Document Processing ───────────────────────────────────────

def process_document_pgvector(document):
    """
    Full pipeline using smart chunking:
    1. Extract text (supports PDF, DOCX, TXT, MD)
    2. Smart chunk based on content type
    3. Create embeddings in batch
    4. Store in PostgreSQL with rich metadata
    """
    from .models import DocumentChunk

    # Step 1 & 2: Extract and smart chunk
    # process_file handles both extraction and chunking
    chunks = process_file(document.file.path)

    if not chunks:
        raise ValueError("No content could be extracted from this document")

    logger.info(f"Document {document.id}: {len(chunks)} smart chunks created")

    # Step 3: Batch embed all chunks at once
    embeddings_model = get_embeddings()
    texts = [chunk['text'] for chunk in chunks]
    logger.info(f"Creating {len(texts)} embeddings...")
    vectors = embeddings_model.embed_documents(texts)
    logger.info("Embeddings complete")

    # Step 4: Delete old chunks and save new ones
    DocumentChunk.objects.filter(document=document).delete()

    chunk_objects = [
        DocumentChunk(
            document=document,
            content=chunk['text'],
            embedding=vector,
            chunk_index=i,
            page_number=chunk['metadata']['page'],
            metadata=chunk['metadata'],   # now includes word_count, content_type etc
        )
        for i, (chunk, vector) in enumerate(zip(chunks, vectors))
    ]

    DocumentChunk.objects.bulk_create(chunk_objects)
    logger.info(f"Saved {len(chunk_objects)} chunks to PostgreSQL")

    return len(chunk_objects)


# ── Similarity Search ─────────────────────────────────────────

def search_similar_chunks(document, question, top_k=4):
    """
    Find the most relevant chunks for a question using vector similarity.

    How it works:
    1. Embed the question into a vector
    2. Compare that vector against all chunk vectors using CosineDistance
    3. Return the top_k closest chunks

    This is the core of RAG — finding relevant context before answering
    """
    from .models import DocumentChunk

    # Embed the question
    embeddings_model = get_embeddings()
    query_vector = get_cached_embedding(question, embeddings_model)

    # CosineDistance annotates each chunk with its distance to the query
    # order_by('distance') puts closest (most relevant) first
    # [:top_k] takes only the top results
    similar_chunks = (
        DocumentChunk.objects
        .filter(document=document)
        .annotate(distance=CosineDistance('embedding', query_vector))
        .order_by('distance')
        [:top_k]
    )

    return similar_chunks

# ── Hybrid Search ─────────────────────────────────────────

def hybrid_search(document, question, top_k=4):
    """
    Combine vector similarity search with keyword search.

    Pipeline:
    1. Vector search → top 10 semantically similar chunks
    2. Keyword search → chunks containing question words
    3. Merge and deduplicate
    4. Re-rank combined results
    5. Return top_k

    Why this works better:
    - Vector finds conceptually related chunks
    - Keyword finds exactly matching chunks
    - Together they cover more ground
    """
    from .models import DocumentChunk

    embeddings_model = get_embeddings()
    query_vector = embeddings_model.embed_query(question)

    # --- Vector Search ---
    vector_results = list(
        DocumentChunk.objects
        .filter(document=document)
        .annotate(distance=CosineDistance('embedding', query_vector))
        .order_by('distance')
        [:10]
    )

    # --- Keyword Search ---
    # Extract meaningful words (skip short words)
    keywords = [
        word.strip('.,?!;:')
        for word in question.split()
        if len(word) > 3
    ]

    keyword_results = []
    if keywords:
        # Build OR query: chunk contains keyword1 OR keyword2 OR keyword3
        keyword_filter = Q()
        for keyword in keywords:
            keyword_filter |= Q(content__icontains=keyword)

        keyword_results = list(
            DocumentChunk.objects
            .filter(document=document)
            .filter(keyword_filter)
            .annotate(distance=CosineDistance('embedding', query_vector))
            [:10]
        )

    # --- Merge and Deduplicate ---
    seen_ids = set()
    combined = []

    # Add vector results first (already sorted by relevance)
    for chunk in vector_results:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            combined.append(chunk)

    # Add keyword results that weren't already found by vector search
    for chunk in keyword_results:
        if chunk.id not in seen_ids:
            seen_ids.add(chunk.id)
            combined.append(chunk)

    logger.info(
        f"Hybrid search: {len(vector_results)} vector + "
        f"{len(keyword_results)} keyword = {len(combined)} unique chunks"
    )

    # Re-rank combined results and return top_k
    if len(combined) > top_k:
        return rerank_chunks(combined, question, top_k=top_k)

    return combined[:top_k]


def answer_question_hybrid(document, question):
    """
    Answer using hybrid search — better than pure vector search.
    Use this as the default for document Q&A.
    """
    chunks = hybrid_search(document, question, top_k=4)

    if not chunks:
        return {'answer': "No relevant content found.", 'sources': []}

    final_chunks = select_chunks_within_budget(chunks, token_budget=2000)
    context = "\n\n---\n\n".join([chunk.content for chunk in final_chunks])

    sources = [
        {
            'text': chunk.content[:150] + "...",
            'page': chunk.page_number,
            'distance': round(float(chunk.distance), 4),
        }
        for chunk in final_chunks
    ]

    chain = RAG_PROMPT | get_llm() | parser
    answer = chain.invoke({"context": context, "question": question})

    return {'answer': answer, 'sources': sources}


# ── RAG Answer ────────────────────────────────────────────────

def answer_question_pgvector(document, question, use_reranking=True):
    """
    Full RAG pipeline with optional re-ranking:
    1. Retrieve top 10 candidates (vector search)
    2. Re-rank to find best 4 (LLM scoring) — optional
    3. Filter by token budget
    4. Generate answer
    """
    from .models import DocumentChunk

    embeddings_model = get_embeddings()
    query_vector = embeddings_model.embed_query(question)

    # Step 1: Get more candidates than we'll use (10 instead of 4)
    # We retrieve more so re-ranking has options to choose from
    candidates = list(
        DocumentChunk.objects
        .filter(document=document)
        .annotate(distance=CosineDistance('embedding', query_vector))
        .order_by('distance')
        [:10]  # ← retrieve 10, not 4
    )

    if not candidates:
        return {'answer': "No relevant content found.", 'sources': []}

    # Step 2: Re-rank (optional — costs extra tokens but improves quality)
    if use_reranking and len(candidates) > 4:
        logger.info("Re-ranking chunks...")
        best_chunks = rerank_chunks(candidates, question, top_k=4)
    else:
        best_chunks = candidates[:4]

    # Step 3: Stay within token budget
    final_chunks = select_chunks_within_budget(best_chunks, token_budget=2000)

    # Step 4: Build context and get answer
    context = "\n\n---\n\n".join([chunk.content for chunk in final_chunks])

    sources = [
        {
            'text': chunk.content[:150] + "...",
            'page': chunk.page_number,
            'distance': round(float(chunk.distance), 4),
        }
        for chunk in final_chunks
    ]

    chain = RAG_PROMPT | get_llm() | parser
    answer = chain.invoke({"context": context, "question": question})

    return {'answer': answer, 'sources': sources}


def delete_document_chunks(document_id):
    """Delete all chunks for a document — call when deleting a document"""
    from .models import DocumentChunk
    deleted_count, _ = DocumentChunk.objects.filter(
        document_id=document_id
    ).delete()
    logger.info(f"Deleted {deleted_count} chunks for document {document_id}")

# ── Cross-Document Search ────────────────────────────────────────────────

def search_all_documents(user, question, top_k=5):
    """
    Search across ALL of a user's documents at once.
    Returns the most relevant chunks regardless of which document they're from.

    This is the power of pgvector + Django ORM:
    One SQL query searches millions of vectors across all documents.
    ChromaDB would require one query per collection (per document).
    """
    from .models import DocumentChunk

    embeddings_model = get_embeddings()
    query_vector = embeddings_model.embed_query(question)

    # Filter by user's documents, search all at once
    similar_chunks = (
        DocumentChunk.objects
        .filter(document__user=user, document__status='ready')
        .annotate(distance=CosineDistance('embedding', query_vector))
        .order_by('distance')
        .select_related('document')   # fetch document info in same query
        [:top_k]
    )

    return similar_chunks


def answer_across_documents(user, question):
    """
    Answer a question by searching across all user's documents.
    Returns answer + which document(s) it came from.
    """
    similar_chunks = search_all_documents(user, question, top_k=5)

    if not similar_chunks:
        return {
            'answer': "No relevant content found across your documents.",
            'sources': []
        }

    context = "\n\n---\n\n".join([chunk.content for chunk in similar_chunks])

    sources = [
        {
            'document_title': chunk.document.title,
            'document_id': chunk.document.id,
            'text': chunk.content[:150] + "...",
            'page': chunk.page_number,
            'distance': round(float(chunk.distance), 4),
        }
        for chunk in similar_chunks
    ]

    chain = RAG_PROMPT | get_llm() | parser
    answer = chain.invoke({"context": context, "question": question})

    return {'answer': answer, 'sources': sources}

def rerank_chunks(chunks, question, top_k=4):
    """
    Re-rank retrieved chunks by actual relevance to the question.

    Why re-rank if we already sorted by cosine distance?
    Cosine distance measures vector similarity — good but not perfect.
    Re-ranking uses the LLM itself to judge relevance — more accurate
    but costs tokens. So we:
    1. Retrieve 10 candidates cheaply (vector search)
    2. Re-rank top 10 expensively (LLM scoring)
    3. Send only best 4 (saves tokens on final answer)

    This is called "retrieve then re-rank" — industry standard pattern.
    """
    if len(chunks) <= top_k:
        return chunks  # no need to re-rank if we have few chunks

    llm = get_llm(temperature=0.0)  # temperature 0 for consistent scoring

    rerank_prompt = ChatPromptTemplate.from_messages([
        ("system", """Score how relevant this text chunk is to the question.
            Return ONLY a number from 0-10. Nothing else.
            10 = directly answers the question
            5  = somewhat related
            0  = completely irrelevant"""),
        ("user", "Question: {question}\n\nChunk: {chunk}\n\nScore:")
    ])

    chain = rerank_prompt | llm | StrOutputParser()

    scored = []
    for chunk in chunks:
        try:
            score_str = chain.invoke({
                "question": question,
                "chunk": chunk.content[:500]  # only first 500 chars for speed
            })
            score = float(score_str.strip())
        except (ValueError, Exception):
            score = 5.0  # default if scoring fails

        scored.append((score, chunk))
        logger.info(f"Chunk scored {score:.1f}: {chunk.content[:60]}...")

    # Sort by score descending, return top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:top_k]]


def count_tokens_approximate(text):
    """
    Approximate token count — 1 token ≈ 4 characters.
    Good enough for budget management without calling a tokenizer API.
    """
    return len(text) // 4


def select_chunks_within_budget(chunks, token_budget=2000):
    """
    Select as many chunks as possible without exceeding token budget.
    Prioritizes higher-ranked chunks (assumes already sorted by relevance).
    """
    selected = []
    tokens_used = 0

    for chunk in chunks:
        chunk_tokens = count_tokens_approximate(chunk.content)

        if tokens_used + chunk_tokens <= token_budget:
            selected.append(chunk)
            tokens_used += chunk_tokens
        else:
            # Try to fit a truncated version
            remaining_tokens = token_budget - tokens_used
            if remaining_tokens > 100:  # only bother if meaningful space left
                truncated_chars = remaining_tokens * 4
                # We can't modify the chunk object, so we note it
                logger.info(f"Token budget reached at {tokens_used} tokens")
            break

    logger.info(f"Selected {len(selected)} chunks using ~{tokens_used} tokens")
    return selected

# ── Enhanced Citation System ────────────────────────────────────────────────

def calculate_confidence(chunks):
    """
    Calculate confidence score based on chunk distances.

    How it works:
    - CosineDistance 0.0 = perfect match (identical meaning)
    - CosineDistance 1.0 = completely unrelated
    - We convert distance to a 0-100% confidence score

    Multiple close matches = higher confidence
    All chunks far away  = lower confidence

    This is approximate — not a probability, but a useful signal.
    """
    if not chunks:
        return 0.0

    # Convert distances to similarity scores (1 - distance)
    # Distance 0.3 → similarity 0.7 → 70%
    similarities = [
        max(0, 1 - float(chunk.distance))
        for chunk in chunks
    ]

    # Weight: best match counts most, others contribute less
    if len(similarities) == 1:
        confidence = similarities[0]
    else:
        # Best chunk = 60% weight, remaining = 40% split evenly
        best = similarities[0]
        rest = sum(similarities[1:]) / len(similarities[1:])
        confidence = (best * 0.6) + (rest * 0.4)

    return round(confidence * 100, 1)  # return as percentage


def extract_relevant_snippet(chunk_content, question, snippet_length=200):
    """
    Find the most relevant sentence within a chunk for citation display.

    Instead of showing the first 200 chars of a chunk (which might not
    be the relevant part), find the sentence that best answers the question.
    """
    sentences = chunk_content.replace('\n', ' ').split('. ')
    question_words = set(question.lower().split())

    best_sentence = sentences[0]
    best_score = 0

    for sentence in sentences:
        sentence_words = set(sentence.lower().split())
        # Score = overlap between question words and sentence words
        overlap = len(question_words & sentence_words)
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence

    # Return snippet, truncated to snippet_length
    snippet = best_sentence.strip()
    if len(snippet) > snippet_length:
        snippet = snippet[:snippet_length] + "..."

    return snippet


def build_citations(chunks, question):
    """
    Build rich citation objects from retrieved chunks.
    Each citation tells the user exactly where the answer came from.
    """
    citations = []

    for i, chunk in enumerate(chunks):
        snippet = extract_relevant_snippet(chunk.content, question)

        citation = {
            'citation_number': i + 1,
            'document': chunk.document.title if hasattr(chunk, 'document') else 'Unknown',
            'page': chunk.page_number,
            'relevance_score': round((1 - float(chunk.distance)) * 100, 1),
            'snippet': snippet,
            'full_text': chunk.content,
            'chunk_type': chunk.metadata.get('content_type', 'text'),
        }
        citations.append(citation)

    return citations


def answer_with_citations(document, question):
    """
    Full RAG pipeline with citations and confidence scores.
    This is the production-grade version of answer_question_hybrid.
    """
    # Step 1: Hybrid search (vector + keyword)
    chunks = hybrid_search(document, question, top_k=5)

    if not chunks:
        return {
            'answer': "I couldn't find relevant information in this document.",
            'citations': [],
            'confidence': 0.0,
            'follow_up_questions': [],
        }

    # Step 2: Token budget management
    final_chunks = select_chunks_within_budget(chunks, token_budget=2000)

    # Step 3: Build context with citation markers
    context_parts = []
    for i, chunk in enumerate(final_chunks):
        # Add [1], [2], [3] markers so AI can reference them
        context_parts.append(f"[{i+1}] {chunk.content}")

    context = "\n\n".join(context_parts)

    # Step 4: Generate answer with citation instructions
    citation_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a precise document assistant.
Answer the question using ONLY the provided context.
When you use information, reference it with [1], [2], etc.

Rules:
- Only use information from the numbered context sections
- Add [N] after each fact you use
- If the answer is not in the context, say exactly:
  "I couldn't find information about this in the document."
- Be concise and direct"""),
        ("user", """Context:
{context}

Question: {question}

Answer (with citation numbers):""")
    ])

    chain = citation_prompt | get_llm(temperature=0.1) | parser
    answer = chain.invoke({"context": context, "question": question})

    # Step 5: Calculate confidence
    confidence = calculate_confidence(final_chunks)

    # Step 6: Build citations
    citations = build_citations(final_chunks, question)

    # Step 7: Generate follow-up questions
    follow_ups = generate_follow_up_questions(question, answer, document.title)

    return {
        'answer': answer,
        'citations': citations,
        'confidence': confidence,
        'confidence_label': get_confidence_label(confidence),
        'follow_up_questions': follow_ups,
    }


def get_confidence_label(confidence):
    """Convert confidence percentage to human-readable label"""
    if confidence >= 85:
        return 'High'
    elif confidence >= 60:
        return 'Medium'
    elif confidence >= 35:
        return 'Low'
    else:
        return 'Very Low — answer may not be in document'


def generate_follow_up_questions(question, answer, document_title):
    """
    Generate 3 related questions the user might want to ask next.
    Helps users explore the document without knowing what's in it.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate exactly 3 short follow-up questions based on
the original question and answer. Questions should help explore related topics.
Return ONLY the 3 questions, one per line, no numbering, no explanation."""),
        ("user", f"""Document: {document_title}
Original question: {question}
Answer received: {answer[:300]}

3 follow-up questions:""")
    ])

    chain = prompt | get_llm(temperature=0.7) | parser

    try:
        result = chain.invoke({})
        questions = [q.strip() for q in result.strip().split('\n') if q.strip()]
        return questions[:3]  # ensure max 3
    except Exception:
        return []  # follow-ups are optional, don't fail if this errors
    
# ── Hallucination Detection ────────────────────────────────────────────────
def detect_hallucination(answer, chunks):
    """
    Basic hallucination detection — check if the answer
    is grounded in the retrieved chunks.

    How it works:
    - Extract key claims from the answer
    - Check if those claims appear in any chunk
    - If answer contains facts not in chunks → possible hallucination

    This is a simple heuristic, not perfect, but catches obvious cases.
    """
    if not chunks or not answer:
        return {'hallucination_risk': 'unknown', 'grounded': False}

    # Combine all chunk content
    all_chunk_text = ' '.join(chunk.content.lower() for chunk in chunks)

    # Check if AI said "I couldn't find" — that's a good sign it's honest
    honest_indicators = [
        "couldn't find",
        "not in the document",
        "not mentioned",
        "no information",
        "i don't know",
    ]

    if any(indicator in answer.lower() for indicator in honest_indicators):
        return {
            'hallucination_risk': 'Low',
            'grounded': True,
            'note': 'AI correctly acknowledged missing information'
        }

    # Extract numbers and specific claims from the answer
    # If answer has specific numbers/dates not in context — red flag
    import re
    answer_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', answer))
    chunk_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', all_chunk_text))

    ungrounded_numbers = answer_numbers - chunk_numbers

    if len(ungrounded_numbers) > 2:
        return {
            'hallucination_risk': 'Medium',
            'grounded': False,
            'note': f'Answer contains numbers not found in source: {ungrounded_numbers}'
        }

    return {
        'hallucination_risk': 'Low',
        'grounded': True,
        'note': 'Answer appears grounded in source material'
    }

def rewrite_query(question, conversation_history=None):
    """
    Rewrite vague or ambiguous questions for better retrieval.

    Examples:
    "Tell me more about it"  →  "What are the key details of the topic discussed?"
    "What about the cost?"   →  "What is the cost or pricing information?"
    "Explain that"           →  "Explain the main concept mentioned in the document"

    Also handles follow-up questions that reference previous context:
    "And what about the second option?" → "What is the second option mentioned?"
    """
    # Skip rewriting for clear, specific questions
    vague_indicators = [
        'tell me more', 'explain that', 'what about', 'and the',
        'more details', 'elaborate', 'what do you mean', 'it ', 'that ',
        'this ', 'they ', 'them '
    ]

    is_vague = (
        len(question.split()) < 5 or
        any(indicator in question.lower() for indicator in vague_indicators)
    )

    if not is_vague:
        return question  # clear question, no rewriting needed

    context = ""
    if conversation_history:
        # Include last 2 exchanges for context
        recent = conversation_history[-4:]
        context = "\n".join([
            f"{'User' if i % 2 == 0 else 'AI'}: {msg}"
            for i, msg in enumerate(recent)
        ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Rewrite the user's question to be more specific and searchable.
Make it standalone (no pronouns like 'it', 'that', 'they').
Return ONLY the rewritten question, nothing else."""),
        ("user", f"""Previous context:
{context}

Original question: {question}

Rewritten question:""")
    ])

    chain = prompt | get_llm(temperature=0.0) | parser

    try:
        rewritten = chain.invoke({}).strip()
        logger.info(f"Query rewritten: '{question}' → '{rewritten}'")
        return rewritten
    except Exception:
        return question  # fallback to original if rewriting fails