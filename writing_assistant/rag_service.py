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
    Full pipeline:
    1. Extract text from file
    2. Split into chunks
    3. Create embedding for each chunk
    4. Store chunks + embeddings in PostgreSQL via DocumentChunk model

    This replaces process_document() which used ChromaDB
    """
    from .models import DocumentChunk

    # Step 1: Extract text
    pages = extract_text_from_file(document.file.path)

    if not pages:
        raise ValueError("No text could be extracted from this document")

    # Step 2: Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    all_chunks = []   # text strings
    all_metadata = [] # page numbers etc

    for page in pages:
        chunks = splitter.split_text(page['text'])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadata.append({'page': page['page'], 'index': i})

    logger.info(f"Document {document.id}: created {len(all_chunks)} chunks")

    # Step 3: Create embeddings
    # embed_documents() takes a list of strings and returns a list of vectors
    # We do this in one batch call — much faster than one call per chunk
    embeddings_model = get_embeddings()
    logger.info(f"Creating embeddings for {len(all_chunks)} chunks...")
    vectors = embeddings_model.embed_documents(all_chunks)
    logger.info(f"Embeddings created successfully")

    # Step 4: Save to PostgreSQL
    # Delete any existing chunks for this document first (re-processing)
    DocumentChunk.objects.filter(document=document).delete()

    # bulk_create saves all chunks in ONE database query — very efficient
    chunk_objects = [
        DocumentChunk(
            document=document,
            content=chunk,
            embedding=vector,      # pgvector stores this as a vector type
            chunk_index=i,
            page_number=meta['page'],
            metadata=meta,
        )
        for i, (chunk, vector, meta) in enumerate(
            zip(all_chunks, vectors, all_metadata)
        )
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