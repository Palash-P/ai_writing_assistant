from django.db import models
from django.contrib.auth.models import User


class AIRequest(models.Model):
    """Tracks every AI API call — tokens, cost, feature used"""

    FEATURE_CHOICES = [
        ('improve', 'Text Improvement'),
        ('email', 'Email Generator'),
        ('summarize', 'Summarizer'),
        ('blog', 'Blog Post Generator'),
        ('doc_qa', 'Document Q&A'),
        ('chat', 'Chat'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='ai_requests')
    feature = models.CharField(max_length=20, choices=FEATURE_CHOICES)
    prompt = models.TextField()
    response = models.TextField()
    model = models.CharField(max_length=50, default='gemini-2.5-flash')
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    total_tokens = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.feature} - {self.created_at}"


class Document(models.Model):
    """Uploaded documents for RAG"""

    STATUS_CHOICES = [
        ('processing', 'Processing'),
        ('ready', 'Ready'),
        ('error', 'Error'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/')
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='processing')
    chunk_count = models.IntegerField(default=0)
    error_message = models.TextField(blank=True, default='')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.user.username} - {self.title}"


class Conversation(models.Model):
    """A chat session — one user can have many"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='conversations')
    title = models.CharField(max_length=200, default='New Conversation')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-updated_at']

    def __str__(self):
        return f"{self.user.username} - {self.title}"


class Message(models.Model):
    """Individual message in a conversation"""

    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages'
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"