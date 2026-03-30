# writing_assistant/memory.py
"""
Conversation history management.
Keeps recent messages in full, summarizes older ones.
"""
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from decouple import config

logger = logging.getLogger(__name__)

# How many recent messages to keep in full
RECENT_MESSAGE_LIMIT = 10

# Summarize when history exceeds this many messages
SUMMARIZE_THRESHOLD = 20


def get_llm():
    return ChatGoogleGenerativeAI(
        model='gemini-2.5-flash',
        google_api_key=config('GEMINI_API_KEY'),
        temperature=0.3,
    )


def summarize_messages(messages):
    """
    Compress a list of messages into a concise summary.
    Called when conversation history gets too long.
    """
    if not messages:
        return ""

    # Build a text representation of the conversation
    conversation_text = "\n".join([
        f"{'User' if msg.role == 'user' else 'AI'}: {msg.content}"
        for msg in messages
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """Summarize this conversation history concisely.
            Focus on: key facts established, topics discussed, decisions made.
            Keep the summary under 200 words.
            Write in third person: 'The user asked about... The AI explained...'"""),
        ("user", "{conversation}")
    ])

    chain = prompt | get_llm() | StrOutputParser()

    try:
        summary = chain.invoke({"conversation": conversation_text})
        logger.info(f"Summarized {len(messages)} messages into {len(summary)} chars")
        return summary
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        # Fallback: just return first/last message
        return f"Earlier in this conversation: {messages[0].content[:100]}..."


def build_history_for_llm(conversation, max_messages=None):
    """
    Build the message history to send to the LLM.

    Strategy:
    - If conversation is short (≤ RECENT_MESSAGE_LIMIT): send all messages
    - If conversation is medium: send recent messages only (sliding window)
    - If conversation is long (> SUMMARIZE_THRESHOLD): summarize old + keep recent

    Returns a list of LangChain message objects.
    """
    all_messages = list(conversation.messages.all())
    total = len(all_messages)

    if max_messages:
        # Simple override — just take last N messages
        messages_to_use = all_messages[-max_messages:]
        return _messages_to_langchain(messages_to_use)

    if total <= RECENT_MESSAGE_LIMIT:
        # Short conversation — send everything
        logger.info(f"Short history: sending all {total} messages")
        return _messages_to_langchain(all_messages)

    elif total <= SUMMARIZE_THRESHOLD:
        # Medium conversation — sliding window, keep recent messages only
        recent = all_messages[-RECENT_MESSAGE_LIMIT:]
        logger.info(f"Medium history: sliding window, keeping last {RECENT_MESSAGE_LIMIT}")
        return _messages_to_langchain(recent)

    else:
        # Long conversation — summarize old messages, keep recent in full
        old_messages = all_messages[:-RECENT_MESSAGE_LIMIT]
        recent_messages = all_messages[-RECENT_MESSAGE_LIMIT:]

        logger.info(f"Long history: summarizing {len(old_messages)} old messages")
        summary = summarize_messages(old_messages)

        # Build history: summary as system context + recent messages
        history = []
        if summary:
            # Add summary as a system-like message at the start
            history.append(SystemMessage(
                content=f"Summary of earlier conversation:\n{summary}"
            ))
        history.extend(_messages_to_langchain(recent_messages))

        logger.info(f"Final history: 1 summary + {len(recent_messages)} recent messages")
        return history


def _messages_to_langchain(messages):
    """Convert Django Message objects to LangChain message objects"""
    result = []
    for msg in messages:
        if msg.role == 'user':
            result.append(HumanMessage(content=msg.content))
        else:
            result.append(AIMessage(content=msg.content))
    return result


def get_conversation_stats(conversation):
    """Return stats about a conversation's memory usage"""
    messages = list(conversation.messages.all())
    total_chars = sum(len(m.content) for m in messages)
    approximate_tokens = total_chars // 4

    return {
        'total_messages': len(messages),
        'total_chars': total_chars,
        'approximate_tokens': approximate_tokens,
        'would_summarize': len(messages) > SUMMARIZE_THRESHOLD,
        'strategy': (
            'full' if len(messages) <= RECENT_MESSAGE_LIMIT
            else 'sliding_window' if len(messages) <= SUMMARIZE_THRESHOLD
            else 'summarize_and_recent'
        )
    }