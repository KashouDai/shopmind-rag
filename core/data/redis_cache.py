# -*- coding: utf-8 -*-
"""
Redis 缓存层 — 会话缓存 + RAG 结果缓存 + Pub/Sub 广播
与 PostgreSQL 分工明确：PG 是主存储，Redis 是缓存，不重复存储
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional, Callable

import redis

from core.config_utils.config import Config

logger = logging.getLogger(__name__)

# TTL 配置（秒）
SESSION_LIST_TTL = 600        # 会话列表缓存 10 分钟
SESSION_META_TTL = 300        # 会话元数据缓存 5 分钟
RAG_RESULT_TTL = 300          # RAG 检索结果缓存 5 分钟
CONVERSATION_TTL = 600        # 完整会话缓存 10 分钟


class RedisCache:
    """Redis 缓存层"""

    _instance = None

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
        self._r: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        self._connected = False
        self._connect()

    def _connect(self) -> bool:
        """连接 Redis"""
        try:
            host = self._config.get("redis_host", "localhost")
            port = self._config.get("redis_port", 6379)
            password = self._config.get("redis_password", "")
            db = self._config.get("redis_db", 0)

            self._r = redis.Redis(
                host=host, port=int(port), db=int(db),
                password=password if password else None,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5
            )
            self._r.ping()
            self._connected = True
            logger.info("Redis 连接成功: %s:%s", host, port)
            return True
        except Exception as e:
            self._connected = False
            self._r = None
            logger.warning("Redis 不可用，将降级为纯 PostgreSQL 模式: %s", e)
            return False

    def _ensure_connected(self) -> bool:
        if self._connected and self._r:
            try:
                self._r.ping()
                return True
            except Exception:
                self._connected = False
                self._r = None
        return self._connect()

    # ============================================================
    # 通用缓存操作
    # ============================================================

    def get(self, key: str) -> Optional[str]:
        if not self._ensure_connected():
            return None
        try:
            return self._r.get(key)
        except Exception as e:
            logger.debug("Redis GET 失败: %s", e)
            return None

    def set(self, key: str, value: str, ttl: int = None) -> bool:
        if not self._ensure_connected():
            return False
        try:
            if ttl:
                return self._r.setex(key, ttl, value)
            return self._r.set(key, value)
        except Exception as e:
            logger.debug("Redis SET 失败: %s", e)
            return False

    def delete(self, key: str) -> bool:
        if not self._ensure_connected():
            return False
        try:
            self._r.delete(key)
            return True
        except Exception as e:
            logger.debug("Redis DELETE 失败: %s", e)
            return False

    def exists(self, key: str) -> bool:
        if not self._ensure_connected():
            return False
        try:
            return bool(self._r.exists(key))
        except Exception:
            return False

    def set_json(self, key: str, data: Any, ttl: int = None) -> bool:
        """存储 JSON 数据"""
        try:
            value = json.dumps(data, ensure_ascii=False, default=str)
            return self.set(key, value, ttl)
        except Exception as e:
            logger.debug("Redis SET_JSON 失败: %s", e)
            return False

    def get_json(self, key: str) -> Optional[Any]:
        """读取 JSON 数据"""
        value = self.get(key)
        if not value:
            return None
        try:
            return json.loads(value)
        except Exception:
            return None

    # ============================================================
    # 会话缓存
    # ============================================================

    def _conv_list_key(self) -> str:
        return "sessions:list"

    def _conv_meta_key(self, conv_id: str) -> str:
        return f"session:meta:{conv_id}"

    def _conv_full_key(self, conv_id: str) -> str:
        return f"session:full:{conv_id}"

    def _msg_list_key(self, conv_id: str) -> str:
        return f"session:msgs:{conv_id}"

    def get_session_list(self) -> Optional[List[Dict]]:
        """获取会话列表（带缓存）"""
        key = self._conv_list_key()
        cached = self.get_json(key)
        if cached is not None:
            logger.debug("会话列表命中缓存")
            return cached
        return None  # None 表示缓存未命中，调用方应查 PG

    def set_session_list(self, sessions: List[Dict]) -> bool:
        """设置会话列表缓存"""
        return self.set_json(self._conv_list_key(), sessions, SESSION_LIST_TTL)

    def invalidate_session_list(self) -> bool:
        """失效会话列表缓存"""
        return self.delete(self._conv_list_key())

    def get_session_meta(self, conv_id: str) -> Optional[Dict]:
        """获取会话元数据（带缓存）"""
        cached = self.get_json(self._conv_meta_key(conv_id))
        if cached is not None:
            logger.debug("会话元数据命中缓存: %s", conv_id[:8])
            return cached
        return None

    def set_session_meta(self, conv_id: str, meta: Dict) -> bool:
        """设置会话元数据缓存"""
        return self.set_json(self._conv_meta_key(conv_id), meta, SESSION_META_TTL)

    def invalidate_session_meta(self, conv_id: str) -> bool:
        """失效会话元数据缓存"""
        return self.delete(self._conv_meta_key(conv_id))

    def get_session_messages(self, conv_id: str) -> Optional[List[Dict]]:
        """获取会话消息列表（带缓存）"""
        return self.get_json(self._msg_list_key(conv_id))

    def set_session_messages(self, conv_id: str, messages: List[Dict]) -> bool:
        """设置会话消息缓存"""
        return self.set_json(self._msg_list_key(conv_id), messages, CONVERSATION_TTL)

    def invalidate_session(self, conv_id: str) -> bool:
        """失效会话所有缓存"""
        ok = True
        ok &= self.delete(self._conv_meta_key(conv_id))
        ok &= self.delete(self._msg_list_key(conv_id))
        ok &= self.delete(self._conv_full_key(conv_id))
        return ok

    # ============================================================
    # RAG 结果缓存
    # ============================================================

    def _rag_cache_key(self, query: str) -> str:
        import hashlib
        qhash = hashlib.md5(query.encode('utf-8')).hexdigest()
        return f"rag:result:{qhash}"

    def get_rag_result(self, query: str) -> Optional[List[Dict]]:
        """获取 RAG 检索结果缓存"""
        return self.get_json(self._rag_cache_key(query))

    def set_rag_result(self, query: str, results: List[Dict]) -> bool:
        """设置 RAG 检索结果缓存"""
        return self.set_json(self._rag_cache_key(query), results, RAG_RESULT_TTL)

    # ============================================================
    # Pub/Sub 广播
    # ============================================================

    CHANNEL_SSE = "sse:broadcast"
    CHANNEL_HANDOVER = "sse:handover"

    def publish(self, channel: str, message: Dict) -> int:
        """发布消息到频道"""
        if not self._ensure_connected():
            logger.warning("Redis 未连接，无法发布 Pub/Sub 消息")
            return 0
        try:
            data = json.dumps(message, ensure_ascii=False)
            return self._r.publish(channel, data)
        except Exception as e:
            logger.warning("Pub/Sub 发布失败: %s", e)
            return 0

    def publish_sse(self, event_type: str, data: Dict) -> int:
        """发布 SSE 事件"""
        return self.publish(self.CHANNEL_SSE, {"type": event_type, "data": data})

    def publish_handover(self, conv_id: str, agent_id: str) -> int:
        """发布人工客服接管通知"""
        return self.publish(self.CHANNEL_HANDOVER, {
            "conversation_id": conv_id,
            "agent_id": agent_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })

    def subscribe(self, channel: str) -> Optional[redis.client.PubSub]:
        """订阅频道"""
        if not self._ensure_connected():
            return None
        try:
            self._pubsub = self._r.pubsub()
            self._pubsub.subscribe(channel)
            logger.info("订阅频道: %s", channel)
            return self._pubsub
        except Exception as e:
            logger.warning("订阅频道失败: %s", e)
            return None

    def listen(self, pubsub: redis.client.PubSub, timeout: float = 1.0):
        """监听 Pub/Sub 消息（非阻塞）"""
        if not pubsub:
            return None
        try:
            message = pubsub.get_message(timeout=timeout)
            if message and message.get("type") == "message":
                try:
                    return json.loads(message["data"])
                except json.JSONDecodeError:
                    return message["data"]
        except Exception as e:
            logger.debug("Pub/Sub 监听异常: %s", e)
        return None

    def close(self):
        """关闭连接"""
        if self._pubsub:
            try:
                self._pubsub.unsubscribe()
                self._pubsub.close()
            except Exception:
                pass
        if self._r:
            try:
                self._r.close()
            except Exception:
                pass
        self._connected = False
        logger.info("Redis 连接已关闭")


# 全局单例
_cache: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    global _cache
    if _cache is None:
        _cache = RedisCache()
    return _cache
