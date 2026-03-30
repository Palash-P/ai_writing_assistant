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
    query_vector = embeddings_model.embed_query(question)

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


# ── RAG Answer ────────────────────────────────────────────────

def answer_question_pgvector(document, question):
    """
    Full RAG pipeline using pgvector:
    1. Find relevant chunks via vector similarity search
    2. Build context from those chunks
    3. Send context + question to Gemini
    4. Return answer with source citations
    """
    # Step 1: Find relevant chunks
    similar_chunks = search_similar_chunks(document, question, top_k=4)

    if not similar_chunks:
        return {
            'answer': "No relevant content found in this document.",
            'sources': []
        }

    # Step 2: Build context string
    context_parts = [chunk.content for chunk in similar_chunks]
    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Build source citations for the response
    sources = [
        {
            'text': chunk.content[:150] + "...",
            'page': chunk.page_number,
            'distance': round(float(chunk.distance), 4),
            # distance: 0 = identical, closer to 0 = more relevant
        }
        for chunk in similar_chunks
    ]

    # Step 4: Get AI answer
    chain = RAG_PROMPT | get_llm() | parser
    answer = chain.invoke({
        "context": context,
        "question": question
    })

    return {
        'answer': answer,
        'sources': sources
    }


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