from rest_framework import serializers
from django.contrib.auth.models import User
from .models import AIRequest, Document, Conversation, Message

 
# ── Auth ──────────────────────────────────────────

class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ['username', 'email', 'password']

    def create(self, validated_data):
        return User.objects.create_user(**validated_data)


# ── Writing Features ───────────────────────────────

class ImproveSerializer(serializers.Serializer):
    text = serializers.CharField(min_length=10)
    tone = serializers.ChoiceField(
        choices=['professional', 'casual', 'friendly'],
        default='professional'
    )


class EmailSerializer(serializers.Serializer):
    bullets = serializers.CharField(min_length=10)
    recipient_context = serializers.CharField(required=False, default='')
    tone = serializers.ChoiceField(
        choices=['professional', 'formal', 'friendly'],
        default='professional'
    )


class SummarizeSerializer(serializers.Serializer):
    text = serializers.CharField(min_length=50)
    length = serializers.ChoiceField(
        choices=['short', 'medium', 'long'],
        default='medium'
    )


class BlogSerializer(serializers.Serializer):
    topic = serializers.CharField(min_length=5)
    keywords = serializers.CharField(required=False, default='')
    tone = serializers.ChoiceField(
        choices=['informative', 'conversational', 'persuasive'],
        default='informative'
    )


# ── Document ───────────────────────────────────────

class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ['id', 'title', 'status', 'chunk_count', 'uploaded_at']
        read_only_fields = ['id', 'status', 'chunk_count', 'uploaded_at']


class AskDocumentSerializer(serializers.Serializer):
    document_id = serializers.IntegerField()
    question = serializers.CharField(min_length=5)


# ── Chat ───────────────────────────────────────────

class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ['id', 'role', 'content', 'created_at']
        read_only_fields = ['id', 'created_at']


class ConversationSerializer(serializers.ModelSerializer):
    messages = MessageSerializer(many=True, read_only=True)
    message_count = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = ['id', 'title', 'message_count', 'messages', 'created_at', 'updated_at']
        read_only_fields = ['id', 'title', 'created_at', 'updated_at']

    def get_message_count(self, obj):
        return obj.messages.count()


class ChatSerializer(serializers.Serializer):
    message = serializers.CharField(min_length=1)
    conversation_id = serializers.IntegerField(required=False, allow_null=True)


# ── Usage ──────────────────────────────────────────

class AIRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = AIRequest
        fields = ['id', 'feature', 'total_tokens', 'created_at']
        read_only_fields = fields