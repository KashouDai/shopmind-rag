# -*- coding: utf-8 -*-
"""
上下文工程模块
对话摘要压缩 + 指代消解
"""

import re
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ContextEngine:
    """上下文工程：滑动窗口 + 关键信息摘要"""

    # 默认阈值配置
    DEFAULT_THRESHOLD_CHARS = 4000
    DEFAULT_KEEP_LATEST = 10
    DEFAULT_SUMMARY_MAX_CHARS = 1500
    DEFAULT_WINDOW_MESSAGES = 12

    def __init__(self, threshold_chars: int = None, keep_latest: int = None,
                 summary_max_chars: int = None, window_messages: int = None):
        self.threshold_chars = threshold_chars or self.DEFAULT_THRESHOLD_CHARS
        self.keep_latest = keep_latest or self.DEFAULT_KEEP_LATEST
        self.summary_max_chars = summary_max_chars or self.DEFAULT_SUMMARY_MAX_CHARS
        self.window_messages = window_messages or self.DEFAULT_WINDOW_MESSAGES

    def process_history(self, messages: list) -> list:
        """
        处理对话历史：
        - 短对话：滑动窗口截断（保留最近 WINDOW_MESSAGES 条）
        - 长对话：摘要早期内容 + 保留最近 KEEP_LATEST 条原文
        """
        if not messages:
            return []

        total_chars = sum(len(m.get("content", "")) for m in messages)

        # 未达到阈值：滑动窗口截断
        if total_chars < self.threshold_chars:
            return messages[-self.window_messages:]

        # 达到阈值：摘要 + 保留近期
        summary_end = len(messages) - self.keep_latest
        if summary_end <= 0:
            return messages[-self.window_messages:]

        early_messages = messages[:summary_end]
        recent_messages = messages[summary_end:]

        summary = self._summarize(early_messages)
        return [
            {"role": "system", "content": f"[历史对话摘要]\n{summary}"}
        ] + recent_messages

    def _summarize(self, messages: list) -> str:
        """调用 LLM 生成关键信息摘要"""
        try:
            from core.agents.model_router import get_model_router
            router = get_model_router()
            if not router.is_configured():
                return self._simple_summary(messages)

            system_msg = (
                "你是一个电商客服对话摘要助手。\n"
                "请从对话中提取以下关键信息，以结构化格式输出：\n"
                "1. 用户咨询的商品名称和ID（如果有）\n"
                "2. 用户询问的关键问题（价格、库存、政策等）\n"
                "3. 已答复的关键事实（价格、库存数量、政策条款等）\n"
                "4. 已达成的结论或方案\n"
                "只输出摘要信息，不要回复用户。控制在1500字符以内。"
            )
            user_msg = self._format_for_summary(messages)

            from langchain_core.messages import HumanMessage, SystemMessage
            response = router.chat([
                SystemMessage(content=system_msg),
                HumanMessage(content=user_msg)
            ])
            return response.strip()
        except Exception as e:
            logger.warning("对话摘要失败，使用简单摘要: %s", e)
            return self._simple_summary(messages)

    def _simple_summary(self, messages: list) -> str:
        """LLM 不可用时的降级方案：提取关键文本"""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:150]
            if role == "user":
                lines.append(f"用户: {content}")
            elif role == "assistant":
                lines.append(f"客服: {content}")
        return "\n".join(lines[:8])

    def _format_for_summary(self, messages: list) -> str:
        """格式化为摘要 Prompt"""
        lines = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def resolve_pronouns(self, query: str, history: list) -> str:
        """
        指代消解：将查询中的指代词替换为具体实体
        策略：简单级（单指代词替换），复杂级后续迭代
        """
        if not history or not query:
            return query

        # 单指代词替换
        pronoun_map = ["它", "这个", "那款", "上述", "此"]
        for pronoun in pronoun_map:
            if pronoun in query:
                entity = self._get_last_entity(history)
                if entity:
                    return query.replace(pronoun, entity)

        # 省略句补全："那P002呢" → "P002的价格和库存"
        ne_pattern = re.compile(r"那(.+?)呢")
        match = ne_pattern.search(query)
        if match:
            entity = match.group(1)
            last_entity = self._get_last_entity(history)
            if last_entity and entity != last_entity:
                remainder = query.replace(f"那{entity}呢", "").strip()
                if remainder:
                    return f"{entity} {remainder}"
                return f"{entity}的价格和库存"

        return query

    def _get_last_entity(self, history: list) -> str:
        """从历史中提取最后一个提到的实体（简单中文名词提取）"""
        for msg in reversed(history):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                nouns = re.findall(r'[一-鿿]{2,6}', content)
                if nouns:
                    return nouns[-1]
        return ""


# 全局单例
_context_engine = None


def get_context_engine(**kwargs) -> ContextEngine:
    global _context_engine
    if _context_engine is None:
        _context_engine = ContextEngine(**kwargs)
    return _context_engine
