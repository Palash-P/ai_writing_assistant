# writing_assistant/views.py
import datetime
from rest_framework.decorators import api_view, permission_classes, parser_classes
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from rest_framework.authtoken.models import Token
from django.contrib.auth import authenticate
from django.shortcuts import get_object_or_404
from django.core.cache import cache
from .models import AIRequest, Document, Conversation, Message
from .serializers import (
    RegisterSerializer, ImproveSerializer, EmailSerializer,
    SummarizeSerializer, BlogSerializer, AskDocumentSerializer,
    DocumentSerializer, ChatSerializer, ConversationSerializer,
    AIRequestSerializer
)
from . import services
from . import rag_service
from .memory import get_conversation_stats



DAILY_REQUEST_LIMIT = 50


def check_quota(user):
    cache_key = f"quota_{user.id}_{datetime.date.today()}"  # ← clean import
    count = cache.get(cache_key, 0)
    if count >= DAILY_REQUEST_LIMIT:
        return False, count
    cache.set(cache_key, count + 1, timeout=86400)
    return True, count + 1


def save_request(user, feature, prompt, response_text):
    AIRequest.objects.create(
        user=user,
        feature=feature,
        prompt=prompt[:500],
        response=response_text[:2000],
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
    )


# ── Auth ──────────────────────────────────────────────────────

@api_view(['POST'])
@permission_classes([AllowAny])
def register(request):
    serializer = RegisterSerializer(data=request.data)
    if serializer.is_valid():
        user = serializer.save()
        token = Token.objects.create(user=user)
        return Response({
            'token': token.key,
            'username': user.username,
            'message': 'Account created successfully'
        }, status=status.HTTP_201_CREATED)
    return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
@permission_classes([AllowAny])
def login(request):
    username = request.data.get('username')
    password = request.data.get('password')
    user = authenticate(username=username, password=password)
    if not user:
        return Response({'error': 'Invalid credentials'}, status=status.HTTP_401_UNAUTHORIZED)
    token, _ = Token.objects.get_or_create(user=user)
    return Response({'token': token.key, 'username': user.username})


@api_view(['POST'])
def logout(request):
    request.user.auth_token.delete()
    return Response({'message': 'Logged out successfully'})


# ── Feature 1: Text Improvement ───────────────────────────────

@api_view(['POST'])
def improve_text(request):
    allowed, count = check_quota(request.user)
    if not allowed:
        return Response(
            {'error': f'Daily limit of {DAILY_REQUEST_LIMIT} requests reached. Resets at midnight.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )

    serializer = ImproveSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    result = services.improve_text(data['text'], data['tone'])
    save_request(request.user, 'improve', data['text'], result)

    return Response({
        'original': data['text'],
        'improved': result,
        'tone': data['tone'],
        'requests_today': count,
        'daily_limit': DAILY_REQUEST_LIMIT,
    })


# ── Feature 2: Email Generator ────────────────────────────────

@api_view(['POST'])
def generate_email(request):
    allowed, count = check_quota(request.user)  # ← added
    if not allowed:
        return Response(
            {'error': f'Daily limit of {DAILY_REQUEST_LIMIT} requests reached.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )

    serializer = EmailSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    result = services.generate_email(
        data['bullets'],
        data['recipient_context'],
        data['tone']
    )
    save_request(request.user, 'email', data['bullets'], result)

    return Response({
        'email': result,
        'tone': data['tone'],
        'requests_today': count,
        'daily_limit': DAILY_REQUEST_LIMIT,
    })


# ── Feature 3: Summarizer ─────────────────────────────────────

@api_view(['POST'])
def summarize(request):
    allowed, count = check_quota(request.user)  # ← added
    if not allowed:
        return Response(
            {'error': f'Daily limit of {DAILY_REQUEST_LIMIT} requests reached.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )

    serializer = SummarizeSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    result = services.summarize_text(data['text'], data['length'])
    save_request(request.user, 'summarize', data['text'], result)

    return Response({
        'summary': result,
        'length': data['length'],
        'original_length': len(data['text']),
        'summary_length': len(result),
        'requests_today': count,
        'daily_limit': DAILY_REQUEST_LIMIT,
    })


# ── Feature 4: Blog Generator ─────────────────────────────────

@api_view(['POST'])
def generate_blog(request):
    allowed, count = check_quota(request.user)  # ← added
    if not allowed:
        return Response(
            {'error': f'Daily limit of {DAILY_REQUEST_LIMIT} requests reached.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )

    serializer = BlogSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    result = services.generate_blog(data['topic'], data['keywords'], data['tone'])
    save_request(request.user, 'blog', data['topic'], result)

    return Response({
        'blog_post': result,
        'topic': data['topic'],
        'word_count': len(result.split()),
        'requests_today': count,
        'daily_limit': DAILY_REQUEST_LIMIT,
    })


# ── Feature 5: Document RAG ───────────────────────────────────

@api_view(['GET', 'POST'])
@parser_classes([MultiPartParser, FormParser, JSONParser])
def documents(request):
    if request.method == 'GET':
        docs = Document.objects.filter(user=request.user)
        return Response(DocumentSerializer(docs, many=True).data)

    allowed, count = check_quota(request.user)
    if not allowed:
        return Response(
            {'error': f'Daily limit of {DAILY_REQUEST_LIMIT} requests reached.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )

    file = request.FILES.get('file')
    title = request.data.get('title', '').strip()

    if not file:
        return Response({'error': 'No file provided'}, status=status.HTTP_400_BAD_REQUEST)
    if not title:
        title = file.name

    document = Document.objects.create(
        user=request.user,
        title=title,
        file=file,
        status='processing'
    )

    try:
        chunk_count = rag_service.process_document_pgvector(document)  # ← changed
        document.status = 'ready'
        document.chunk_count = chunk_count
        document.save()
    except Exception as e:
        document.status = 'error'
        document.error_message = str(e)
        document.save()
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    return Response(DocumentSerializer(document).data, status=status.HTTP_201_CREATED)


@api_view(['DELETE'])
def delete_document(request, pk):
    document = get_object_or_404(Document, pk=pk, user=request.user)
    rag_service.delete_document_chunks(document.id)  # ← changed
    document.delete()
    return Response(status=status.HTTP_204_NO_CONTENT)


@api_view(['POST'])
def ask_document(request):
    allowed, count = check_quota(request.user)
    if not allowed:
        return Response(
            {'error': f'Daily limit of {DAILY_REQUEST_LIMIT} requests reached.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )

    serializer = AskDocumentSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    document = get_object_or_404(
        Document,
        pk=data['document_id'],
        user=request.user,
        status='ready'
    )

    # ← use hybrid search instead of pure vector search
    result = rag_service.answer_question_hybrid(document, data['question'])
    save_request(request.user, 'doc_qa', data['question'], result['answer'])

    return Response({
        'question': data['question'],
        'answer': result['answer'],
        'sources': result['sources'],
        'document': document.title,
        'requests_today': count,
        'daily_limit': DAILY_REQUEST_LIMIT,
    })

# ── Feature 6: Chat with Memory ───────────────────────────────

@api_view(['GET', 'POST'])
def conversations(request):
    if request.method == 'GET':
        convs = Conversation.objects.filter(user=request.user)
        return Response(ConversationSerializer(convs, many=True).data)

    conv = Conversation.objects.create(user=request.user)
    return Response(ConversationSerializer(conv).data, status=status.HTTP_201_CREATED)


@api_view(['GET', 'DELETE'])
def conversation_detail(request, pk):
    conversation = get_object_or_404(Conversation, pk=pk, user=request.user)

    if request.method == 'GET':
        return Response(ConversationSerializer(conversation).data)

    conversation.messages.all().delete()
    conversation.title = 'New Conversation'
    conversation.save()
    return Response({'message': 'Conversation cleared'})

@api_view(['GET'])
def conversation_stats(request, pk):
    """See memory usage for a conversation"""
    conversation = get_object_or_404(Conversation, pk=pk, user=request.user)
    stats = get_conversation_stats(conversation)
    return Response(stats)


@api_view(['POST'])
def chat(request):
    allowed, count = check_quota(request.user)  # ← added
    if not allowed:
        return Response(
            {'error': f'Daily limit of {DAILY_REQUEST_LIMIT} requests reached.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )

    serializer = ChatSerializer(data=request.data)
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    data = serializer.validated_data
    conv_id = data.get('conversation_id')

    if conv_id:
        conversation = get_object_or_404(Conversation, pk=conv_id, user=request.user)
    else:
        conversation = Conversation.objects.create(user=request.user)

    response_text = services.chat_response(conversation, data['message'])
    save_request(request.user, 'chat', data['message'], response_text)

    return Response({
        'response': response_text,
        'conversation_id': conversation.id,
        'conversation_title': conversation.title,
        'requests_today': count,
        'daily_limit': DAILY_REQUEST_LIMIT,
    })


# ── Usage Stats ───────────────────────────────────────────────

@api_view(['GET'])
def usage_stats(request):
    requests = AIRequest.objects.filter(user=request.user)
    by_feature = {}
    for req in requests:
        by_feature[req.feature] = by_feature.get(req.feature, 0) + 1

    return Response({
        'total_requests': requests.count(),
        'by_feature': by_feature,
        'recent': AIRequestSerializer(requests[:10], many=True).data
    })


# ── Cross-Document Search ───────────────────────────────────────────────

@api_view(['POST'])
def search_documents(request):
    """Search across ALL user documents at once"""
    allowed, count = check_quota(request.user)
    if not allowed:
        return Response(
            {'error': f'Daily limit of {DAILY_REQUEST_LIMIT} requests reached.'},
            status=status.HTTP_429_TOO_MANY_REQUESTS
        )

    question = request.data.get('question', '').strip()
    if not question:
        return Response(
            {'error': 'Question is required'},
            status=status.HTTP_400_BAD_REQUEST
        )

    result = rag_service.answer_across_documents(request.user, question)
    save_request(request.user, 'doc_qa', question, result['answer'])

    return Response({
        'question': question,
        'answer': result['answer'],
        'sources': result['sources'],
        'searched_across': 'all documents',
        'requests_today': count,
        'daily_limit': DAILY_REQUEST_LIMIT,
    })