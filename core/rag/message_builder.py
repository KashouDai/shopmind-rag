# -*- coding: utf-8 -*-
"""
消息构建模块
使用 LangChain ChatPromptTemplate + MessagesPlaceholder，替代手写的 build_messages / trim_history / truncate_text
"""

from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from core.rag.prompt import (
    build_rag_system_prompt,
    build_product_reply_prompt
)


class MessageBuilder:
    """消息构建器"""

    def __init__(self, max_history_messages: int = 12, max_history_chars: int = 6000):
        self.max_history_messages = max_history_messages
        self.max_history_chars = max_history_chars

        # 预编译的 prompt 模板
        self._rag_prompt_template = ChatPromptTemplate.from_messages([
            ("system", build_rag_system_prompt([])),
            MessagesPlaceholder("history", optional=True),
            ("human", "{question}")
        ])

        self._product_prompt_template = ChatPromptTemplate.from_messages([
            ("system", build_product_reply_prompt([])),
            ("human", "{question}")
        ])

    def build_rag_messages(
        self,
        question: str,
        context_chunks: List[Dict[str, str]],
        history: Optional[List[Dict[str, str]]] = None
    ) -> List[Any]:
        """构建 RAG 场景的消息列表"""
        # 构建 system prompt（包含知识库内容）
        system_prompt = build_rag_system_prompt(context_chunks)

        # 对历史消息做上下文工程（摘要压缩），将摘要合并到 system prompt
        summary_hint = ""
        if history:
            from core.agents.context_engine import get_context_engine
            engine = get_context_engine()
            processed = engine.process_history(history)
            # 如果首条是系统消息（摘要），提取出来合并到 system prompt
            if processed and processed[0].get("role") == "system":
                summary_hint = processed[0].get("content", "")
                processed = processed[1:]

            # 规范化剩余历史
            normalized = self._normalize_raw(processed)
        else:
            normalized = []

        # 构建消息列表
        messages = [SystemMessage(content=system_prompt + ("\n\n" + summary_hint if summary_hint else ""))]

        # 添加历史消息
        for msg in normalized:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))

        # 添加当前问题
        messages.append(HumanMessage(content=question))

        return messages

    def build_product_messages(
        self,
        question: str,
        product_data: List[Dict[str, Any]]
    ) -> List[Any]:
        """构建商品查询场景的消息列表"""
        system_prompt = build_product_reply_prompt(product_data)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]

        return messages

    def _normalize_history(
        self,
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """标准化历史消息，移除无效消息，并应用上下文工程"""
        normalized = []
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant") or not content:
                continue
            normalized.append({"role": role, "content": str(content)})

        # 应用上下文工程（摘要压缩 + 滑动窗口）
        from core.agents.context_engine import get_context_engine
        engine = get_context_engine()
        processed = engine.process_history(normalized)

        # 过滤掉 system 摘要消息（已在 build_rag_messages 中处理）
        return [m for m in processed if m.get("role") != "system"]

    def _normalize_raw(
        self,
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """标准化原始历史消息（不含摘要）"""
        normalized = []
        for msg in history:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            content = msg.get("content")
            if role not in ("user", "assistant") or not content:
                continue
            normalized.append({"role": role, "content": str(content)})
        return normalized

    def _trim_history(
        self,
        history: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """裁剪历史消息，保留最近的消息"""
        if not history:
            return []

        # 限制消息数量
        if self.max_history_messages > 0:
            history = history[-self.max_history_messages:]

        # 限制总字符数
        if self.max_history_chars > 0:
            trimmed = []
            total_chars = 0
            for msg in reversed(history):
                content = msg.get("content", "")
                if total_chars + len(content) > self.max_history_chars:
                    if not trimmed:
                        # 最后一个消息，截断
                        trimmed.append({
                            "role": msg.get("role", "user"),
                            "content": content[:self.max_history_chars]
                        })
                    break
                trimmed.append(msg)
                total_chars += len(content)
            history = list(reversed(trimmed))

        return history


# 全局单例
_message_builder: Optional[MessageBuilder] = None


def get_message_builder() -> MessageBuilder:
    """获取消息构建器单例"""
    global _message_builder
    if _message_builder is None:
        _message_builder = MessageBuilder()
    return _message_builder
