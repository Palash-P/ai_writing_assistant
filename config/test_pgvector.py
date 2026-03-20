# test_pgvector.py — run this to see the difference
import os
import django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
django.setup()

from writing_assistant.rag_service import search_similar_chunks
from writing_assistant.models import Document, DocumentChunk
import time

# This test assumes you have at least one processed document
# If not, process one first via the API

docs = Document.objects.filter(status='ready')
if not docs.exists():
    print("No ready documents found. Upload one via the API first.")
else:
    doc = docs.first()
    chunk_count = DocumentChunk.objects.filter(document=doc).count()
    print(f"Testing with: '{doc.title}' ({chunk_count} chunks)")

    questions = [
        "What is the main topic?",
        "What are the key skills?",
        "What experience is mentioned?",
    ]

    for question in questions:
        start = time.time()
        chunks = search_similar_chunks(doc, question, top_k=3)
        elapsed = (time.time() - start) * 1000

        print(f"\nQ: {question}")
        print(f"Search time: {elapsed:.1f}ms")
        print(f"Top result (distance: {chunks[0].distance:.4f}):")
        print(f"  {chunks[0].content[:100]}...")