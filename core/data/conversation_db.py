# -*- coding: utf-8 -*-
"""
会话数据访问层 — PostgreSQL + Redis 缓存
替代 conversation.py 中的 JSON 文件读写
"""

import json
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from core.config_utils.config import Config
from core.data.redis_cache import get_redis_cache

logger = logging.getLogger(__name__)

Base = declarative_base()


class ConversationModel(Base):
    """会话表 ORM 模型"""
    __tablename__ = "conversation"

    id = Column(String(36), primary_key=True, comment="UUID")
    user_id = Column(String(50), default="", comment="关联用户")
    title = Column(String(200), default="新对话", comment="会话标题")
    status = Column(String(20), default="normal", comment="会话状态")
    human_agent_id = Column(String(50), default=None, comment="人工客服ID")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")


class MessageModel(Base):
    """消息表 ORM 模型"""
    __tablename__ = "message"

    id = Column(Integer, primary_key=True, autoincrement=True, comment="自增ID")
    conversation_id = Column(String(36), comment="关联会话ID")
    role = Column(String(20), nullable=False, comment="user/assistant/human")
    content = Column(Text, nullable=False, comment="消息内容")
    confidence = Column(String(10), default=None, comment="检索匹配分（兼容列）")
    rag_trace = Column(Text, default=None, comment="检索追溯信息JSON")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")


class ConversationDB:
    """会话数据访问层 — PostgreSQL + Redis 缓存"""

    _instance = None
    _engine = None
    _session_factory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._config = Config()
        self._engine = self._create_engine()
        self._session_factory = sessionmaker(bind=self._engine)
        self._create_tables()

    def _get_db_url(self) -> str:
        config = self._config.get_all()
        host = config.get("postgres_host", "localhost")
        port = config.get("postgres_port", 5432)
        database = config.get("postgres_database", "ecom_db")
        user = config.get("postgres_user", "postgres")
        password = config.get("postgres_password", "123456")
        return f"postgresql+pg8000://{user}:{password}@{host}:{port}/{database}"

    def _create_engine(self):
        db_url = self._get_db_url()
        try:
            engine = create_engine(db_url, echo=False)
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("PostgreSQL 会话连接成功")
            return engine
        except Exception as e:
            logger.error("PostgreSQL 会话连接失败: %s", e)
            raise

    def _create_tables(self):
        Base.metadata.create_all(self._engine)
        logger.info("会话表检查完成")

    def _session(self) -> Session:
        return self._session_factory()

    # ============================================================
    # 会话 CRUD
    # ============================================================

    def create_conversation(self, user_id: str = "") -> Dict:
        session = self._session()
        try:
            conv_id = str(__import__('uuid').uuid4())
            now = datetime.utcnow()
            model = ConversationModel(
                id=conv_id, user_id=user_id,
                created_at=now, updated_at=now
            )
            session.add(model)
            session.commit()
            # 清除缓存
            cache = get_redis_cache()
            cache.invalidate_session_list()
            logger.info("创建会话: %s", conv_id[:8])
            return self._model_to_dict(model)
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("创建会话失败: %s", e)
            raise
        finally:
            session.close()

    def get_conversation(self, conv_id: str) -> Optional[Dict]:
        # 先查缓存
        cache = get_redis_cache()
        cached = cache.get_session_meta(conv_id)
        if cached:
            # 补充消息
            msgs = cache.get_session_messages(conv_id)
            if msgs is not None:
                cached["messages"] = msgs
                return cached

        session = self._session()
        try:
            model = session.query(ConversationModel).filter_by(id=conv_id).first()
            if not model:
                return None
            result = self._model_to_dict(model)

            # 加载消息
            messages = session.query(MessageModel)\
                .filter_by(conversation_id=conv_id)\
                .order_by(MessageModel.created_at.asc()).all()
            result["messages"] = [self._msg_model_to_dict(m) for m in messages]

            # 写入缓存
            meta = dict(result)
            meta.pop("messages", None)
            cache.set_session_meta(conv_id, meta)
            cache.set_session_messages(conv_id, result["messages"])
            return result
        finally:
            session.close()

    def update_conversation_status(self, conv_id: str, **kwargs) -> bool:
        session = self._session()
        try:
            model = session.query(ConversationModel).filter_by(id=conv_id).first()
            if not model:
                return False
            for key, value in kwargs.items():
                if hasattr(model, key):
                    setattr(model, key, value)
            model.updated_at = datetime.utcnow()
            session.commit()

            # 更新缓存
            cache = get_redis_cache()
            cache.invalidate_session(conv_id)
            cache.invalidate_session_list()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("更新会话状态失败: %s", e)
            return False
        finally:
            session.close()

    def delete_conversation(self, conv_id: str) -> bool:
        session = self._session()
        try:
            model = session.query(ConversationModel).filter_by(id=conv_id).first()
            if not model:
                return False
            session.query(MessageModel).filter_by(conversation_id=conv_id).delete()
            session.delete(model)
            session.commit()

            # 清除缓存
            cache = get_redis_cache()
            cache.invalidate_session(conv_id)
            cache.invalidate_session_list()
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("删除会话失败: %s", e)
            return False
        finally:
            session.close()

    def get_all_conversations(self, limit: int = 100) -> List[Dict]:
        # 尝试从缓存获取（不含消息，快速列表）
        cache = get_redis_cache()
        cached = cache.get_session_list()
        if cached is not None:
            return cached

        session = self._session()
        try:
            models = session.query(ConversationModel)\
                .order_by(ConversationModel.updated_at.desc())\
                .limit(limit).all()
            result = [self._model_to_dict(m) for m in models]

            # 写入缓存
            cache.set_session_list(result)
            return result
        finally:
            session.close()

    def get_all_conversations_with_messages(self, limit: int = 100) -> List[Dict]:
        """获取会话列表（带消息）"""
        session = self._session()
        try:
            models = session.query(ConversationModel)\
                .order_by(ConversationModel.updated_at.desc())\
                .limit(limit).all()
            result = []
            for m in models:
                d = self._model_to_dict(m)
                msgs = session.query(MessageModel)\
                    .filter_by(conversation_id=m.id)\
                    .order_by(MessageModel.created_at.asc()).all()
                d["messages"] = [self._msg_model_to_dict(msg) for msg in msgs]
                result.append(d)

            # 写入缓存（只缓存元数据，不缓存消息以节省空间）
            cache = get_redis_cache()
            cache.set_session_list([self._model_to_dict(m) for m in models])
            return result
        finally:
            session.close()

    def get_conversation_with_messages(self, conv_id: str) -> Optional[Dict]:
        """获取会话及其消息（用于工作台/侧边栏显示）"""
        session = self._session()
        try:
            model = session.query(ConversationModel).filter_by(id=conv_id).first()
            if not model:
                return None
            result = self._model_to_dict(model)

            messages = session.query(MessageModel)\
                .filter_by(conversation_id=conv_id)\
                .order_by(MessageModel.created_at.asc()).all()
            result["messages"] = [self._msg_model_to_dict(m) for m in messages]

            # 更新缓存
            cache = get_redis_cache()
            meta = dict(result)
            meta.pop("messages", None)
            cache.set_session_meta(conv_id, meta)
            cache.set_session_messages(conv_id, result["messages"])
            return result
        finally:
            session.close()

    def get_conversation_count(self) -> int:
        session = self._session()
        try:
            return session.query(ConversationModel).count()
        finally:
            session.close()

    # ============================================================
    # 消息 CRUD
    # ============================================================

    def add_message(self, conv_id: str, role: str, content: str,
                    confidence: float = None, rag_trace: dict = None) -> Optional[Dict]:
        session = self._session()
        try:
            # 验证会话存在
            conv = session.query(ConversationModel).filter_by(id=conv_id).first()
            if not conv:
                logger.warning("会话不存在: %s", conv_id)
                return None

            msg_model = MessageModel(
                conversation_id=conv_id,
                role=role,
                content=content,
                confidence=str(confidence) if confidence is not None else None,
                rag_trace=json.dumps(rag_trace, ensure_ascii=False) if rag_trace else None,
                created_at=datetime.utcnow()
            )
            session.add(msg_model)
            # 更新会话时间
            conv.updated_at = datetime.utcnow()
            if role == "user" and conv.title in (None, "", "新对话"):
                title = " ".join(content.split())
                conv.title = title[:50]
            session.commit()

            # 清除消息缓存
            cache = get_redis_cache()
            cache.invalidate_session(conv_id)

            return self._msg_model_to_dict(msg_model)
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("添加消息失败: %s", e)
            return None
        finally:
            session.close()

    def get_messages(self, conv_id: str) -> List[Dict]:
        # 尝试从缓存获取
        cache = get_redis_cache()
        cached = cache.get_session_messages(conv_id)
        if cached is not None:
            return cached

        session = self._session()
        try:
            messages = session.query(MessageModel)\
                .filter_by(conversation_id=conv_id)\
                .order_by(MessageModel.created_at.asc()).all()
            result = [self._msg_model_to_dict(m) for m in messages]

            # 写入缓存
            cache.set_session_messages(conv_id, result)
            return result
        finally:
            session.close()

    # ============================================================
    # 从旧 JSON 迁移
    # ============================================================

    def migrate_from_json(self, data_dir: str) -> int:
        """从旧 JSON 文件迁移数据到 PostgreSQL（幂等）"""
        import os
        migrated = 0
        if not os.path.exists(data_dir):
            return 0

        session = self._session()
        try:
            for filename in os.listdir(data_dir):
                if not filename.endswith(".json"):
                    continue
                conv_id = filename[:-5]
                existing = session.query(ConversationModel).filter_by(id=conv_id).first()
                if existing:
                    continue

                filepath = os.path.join(data_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except (json.JSONDecodeError, IOError):
                    continue

                conv = data
                session.add(ConversationModel(
                    id=conv["id"],
                    title=conv.get("title", "新对话"),
                    status=conv.get("status", "normal"),
                    human_agent_id=conv.get("human_agent_id"),
                    created_at=datetime.strptime(conv.get("created_at", ""), "%Y-%m-%d %H:%M:%S") if conv.get("created_at") else datetime.utcnow(),
                    updated_at=datetime.strptime(conv.get("updated_at", ""), "%Y-%m-%d %H:%M:%S") if conv.get("updated_at") else datetime.utcnow(),
                ))

                for msg in conv.get("messages", []):
                    session.add(MessageModel(
                        conversation_id=conv["id"],
                        role=msg["role"],
                        content=msg["content"],
                        confidence=str(msg["confidence"]) if msg.get("confidence") else None,
                        rag_trace=json.dumps(msg.get("rag_trace"), ensure_ascii=False) if msg.get("rag_trace") else None,
                        created_at=datetime.strptime(msg.get("timestamp", ""), "%Y-%m-%d %H:%M:%S") if msg.get("timestamp") else datetime.utcnow(),
                    ))

                session.commit()
                migrated += 1
                logger.info("迁移会话: %s (%d 条消息)", conv_id, len(conv.get("messages", [])))

            logger.info("JSON 迁移完成，共迁移 %d 个会话", migrated)
            return migrated
        except Exception as e:
            session.rollback()
            logger.error("迁移失败: %s", e)
            return 0
        finally:
            session.close()

    # ============================================================
    # 内部转换方法
    # ============================================================

    def _model_to_dict(self, model: ConversationModel) -> Dict:
        return {
            "id": model.id,
            "user_id": model.user_id or "",
            "title": model.title or "新对话",
            "status": model.status or "normal",
            "human_agent_id": model.human_agent_id,
            "created_at": model.created_at.strftime("%Y-%m-%d %H:%M:%S") if model.created_at else "",
            "updated_at": model.updated_at.strftime("%Y-%m-%d %H:%M:%S") if model.updated_at else "",
        }

    def _msg_model_to_dict(self, model: MessageModel) -> Dict:
        return {
            "role": model.role,
            "content": model.content,
            "timestamp": model.created_at.strftime("%Y-%m-%d %H:%M:%S") if model.created_at else "",
            "confidence": float(model.confidence) if model.confidence else None,
            "rag_trace": json.loads(model.rag_trace) if model.rag_trace else None,
        }

    def close(self):
        if self._engine:
            self._engine.dispose()


# 全局单例
_db: Optional[ConversationDB] = None


def get_conversation_db() -> ConversationDB:
    global _db
    if _db is None:
        _db = ConversationDB()
    return _db
