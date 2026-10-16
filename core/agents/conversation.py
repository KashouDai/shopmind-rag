# -*- coding: utf-8 -*-
"""
对话管理模块 — PostgreSQL 持久化 + 内存缓存
"""

import json
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from core.data.redis_cache import get_redis_cache
from core.data.conversation_db import (
    ConversationDB,
    ConversationModel,
    MessageModel,
)

logger = logging.getLogger(__name__)

class Message:
    """消息类"""

    def __init__(self, role: str, content: str, timestamp: str = None,
                 confidence: float = None, rag_trace: dict = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.confidence = confidence
        self.rag_trace = rag_trace

    def to_dict(self) -> dict:
        result = {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp
        }
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.rag_trace is not None:
            result["rag_trace"] = self.rag_trace
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp"),
            confidence=data.get("confidence"),
            rag_trace=data.get("rag_trace")
        )

    @classmethod
    def from_model(cls, model: MessageModel) -> "Message":
        return cls(
            role=model.role,
            content=model.content,
            timestamp=model.created_at.strftime("%Y-%m-%d %H:%M:%S") if model.created_at else None,
            confidence=float(model.confidence) if model.confidence else None,
            rag_trace=json.loads(model.rag_trace) if model.rag_trace else None
        )

    def to_model(self, conversation_id: str = None) -> MessageModel:
        return MessageModel(
            conversation_id=conversation_id,
            role=self.role,
            content=self.content,
            confidence=str(self.confidence) if self.confidence is not None else None,
            rag_trace=json.dumps(self.rag_trace, ensure_ascii=False) if self.rag_trace else None,
            created_at=datetime.utcnow()
        )


class Conversation:
    """对话类"""

    STATUS_NORMAL = "normal"
    STATUS_PENDING_HUMAN = "pending_human"
    STATUS_HUMAN_HANDLING = "human_handling"
    STATUS_HUMAN_CLOSED = "human_closed"

    def __init__(self, conv_id: str = None, title: str = "新对话",
                 messages: List[Message] = None, status: str = None,
                 human_agent_id: str = None):
        self.id = conv_id or str(uuid.uuid4())
        self.title = title
        self.messages = messages or []
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.updated_at = self.created_at
        self.status = status or self.STATUS_NORMAL
        self.human_agent_id = human_agent_id

    def add_message(self, role: str, content: str,
                    confidence: float = None, rag_trace: dict = None) -> Message:
        message = Message(role, content, confidence=confidence, rag_trace=rag_trace)
        self.messages.append(message)
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if role == "user" and self.title == "新对话":
            self.title = content[:20] + ("..." if len(content) > 20 else "")
        return message

    def transfer_to_human(self):
        self.status = self.STATUS_PENDING_HUMAN
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def accept_by_human(self, agent_id: str = "human_agent"):
        self.status = self.STATUS_HUMAN_HANDLING
        self.human_agent_id = agent_id
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def close_human_service(self):
        self.status = self.STATUS_HUMAN_CLOSED
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def back_to_ai(self):
        self.status = self.STATUS_NORMAL
        self.human_agent_id = None
        self.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "human_agent_id": self.human_agent_id
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Conversation":
        if isinstance(data, cls):
            return data
        conv = cls(
            conv_id=data["id"],
            title=data.get("title", "新对话"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])]
        )
        conv.created_at = data.get("created_at", conv.created_at)
        conv.updated_at = data.get("updated_at", conv.updated_at)
        conv.status = data.get("status", conv.STATUS_NORMAL)
        conv.human_agent_id = data.get("human_agent_id")
        return conv

    @classmethod
    def from_model(cls, model: ConversationModel) -> "Conversation":
        conv = cls(conv_id=model.id)
        conv.title = model.title or "新对话"
        conv.created_at = model.created_at.strftime("%Y-%m-%d %H:%M:%S") if model.created_at else ""
        conv.updated_at = model.updated_at.strftime("%Y-%m-%d %H:%M:%S") if model.updated_at else ""
        conv.status = model.status or cls.STATUS_NORMAL
        conv.human_agent_id = model.human_agent_id
        return conv


class ConversationManager:
    """对话管理器 - PostgreSQL 持久化 + 内存缓存"""

    _instance = None
    DEFAULT_PAGE_SIZE = 20

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        from core.data.conversation_db import ConversationDB
        from core.data.redis_cache import RedisCache
        self.db = ConversationDB()
        self.cache = RedisCache()
        self.conversations: Dict[str, Conversation] = {}
        self.current_conversation: Optional[Conversation] = None
        self._conversation_ids: List[str] = []
        self._load_conversations()

    def _load_conversations(self) -> None:
        current_id = self.current_conversation.id if self.current_conversation else None
        self.conversations.clear()
        self._conversation_ids.clear()

        # 先查 Redis 缓存
        cached = self.cache.get_session_list()
        if cached:
            for conv_data in cached:
                if isinstance(conv_data, Conversation):
                    conv = conv_data
                else:
                    conv = Conversation.from_dict(conv_data)
                # 缓存列表中只有元数据，需要补充消息
                full = self.db.get_conversation_with_messages(conv.id)
                if full:
                    conv.messages = [Message.from_dict(m) for m in full.get("messages", [])]
                    conv.status = full.get("status", conv.status)
                    conv.human_agent_id = full.get("human_agent_id")
                    conv.updated_at = full.get("updated_at", conv.updated_at)
                    conv.title = full.get("title", conv.title)
                self.conversations[conv.id] = conv
                self._conversation_ids.append(conv.id)
        else:
            # 缓存未命中，查 PG（带消息）
            conv_dicts = self.db.get_all_conversations_with_messages(limit=self.DEFAULT_PAGE_SIZE * 2)
            for conv_dict in conv_dicts:
                conv = Conversation.from_dict(conv_dict)
                self.conversations[conv.id] = conv
                self._conversation_ids.append(conv.id)

        if current_id and current_id in self.conversations:
            self.current_conversation = self.conversations[current_id]
        elif current_id:
            self.current_conversation = None

    def create_conversation(self, user_id: str = "") -> Conversation:
        conv_dict = self.db.create_conversation(user_id=user_id)
        conv = Conversation.from_dict(conv_dict)
        self.conversations[conv.id] = conv
        self._conversation_ids.insert(0, conv.id)
        self.current_conversation = conv
        self.cache.invalidate_session_list()
        return conv

    def get_conversation(self, conv_id: str) -> Optional[Conversation]:
        if conv_id in self.conversations:
            # 确保内存中的会话有消息（可能缓存没有）
            conv = self.conversations[conv_id]
            if not conv.messages:
                full = self.db.get_conversation_with_messages(conv_id)
                if full:
                    conv_dict = full
                    conv.messages = [Message.from_dict(m) for m in conv_dict.get("messages", [])]
                    conv.updated_at = conv_dict.get("updated_at", conv.updated_at)
                    conv.status = conv_dict.get("status", conv.status)
                    conv.human_agent_id = conv_dict.get("human_agent_id")
                    self.conversations[conv_id] = conv
            return conv
        conv_dict = self.db.get_conversation_with_messages(conv_id)
        if conv_dict:
            conv = Conversation.from_dict(conv_dict)
            self.conversations[conv_id] = conv
            # 写入缓存（不含消息，节省空间）
            self.cache.set_session_meta(conv_id, {k: v for k, v in conv_dict.items() if k != "messages"})
            self.cache.set_session_messages(conv_id, conv_dict.get("messages", []))
            return conv
        return None

    def delete_conversation(self, conv_id: str) -> bool:
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            if conv_id in self._conversation_ids:
                self._conversation_ids.remove(conv_id)
        result = self.db.delete_conversation(conv_id)
        self.cache.invalidate_session(conv_id)
        self.cache.invalidate_session_list()
        if result and self.current_conversation and self.current_conversation.id == conv_id:
            self.current_conversation = None
        return result

    def get_all_conversations(self) -> List[Conversation]:
        return sorted(self.conversations.values(), key=lambda x: x.updated_at, reverse=True)

    def get_conversations_page(self, page: int = 1, page_size: int = None) -> Tuple[List[Conversation], int]:
        page_size = page_size or self.DEFAULT_PAGE_SIZE
        all_convs = self.get_all_conversations()
        total_pages = (len(all_convs) + page_size - 1) // page_size
        start = (page - 1) * page_size
        return all_convs[start:start + page_size], total_pages

    def get_conversation_count(self) -> int:
        return self.db.get_conversation_count()

    def set_current_conversation(self, conv_id: str) -> Optional[Conversation]:
        conv = self.get_conversation(conv_id)
        if conv:
            self.current_conversation = conv
        return conv

    def add_message(self, role: str, content: str, confidence: float = None,
                    rag_trace: dict = None) -> Optional[Message]:
        if not self.current_conversation:
            return None
        conv_id = self.current_conversation.id
        msg = self.db.add_message(conv_id, role, content, confidence=confidence, rag_trace=rag_trace)
        if msg:
            self.current_conversation.add_message(role, content, confidence=confidence, rag_trace=rag_trace)
            self.conversations[conv_id] = self.current_conversation
        return msg


# 全局单例
_conversation_manager: Optional["ConversationManager"] = None


def get_conversation_manager() -> "ConversationManager":
    global _conversation_manager
    if _conversation_manager is None:
        _conversation_manager = ConversationManager()
    return _conversation_manager
