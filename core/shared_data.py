# -*- coding: utf-8 -*-
"""
共享数据管理器 - 知识库使用 Milvus 向量存储 + PostgreSQL 商品存储

改造说明 (v4.0.0):
- 知识库: 从 JSON 手动录入 → Markdown 文档上传自动切片
- 商品数据: 从 JSON → PostgreSQL
- 向量检索: 从 FAISS → Milvus
- 检索链: 使用 LangChain RAGChain + AgentRouter
"""


from typing import List, Optional, Tuple, Dict, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import os
import logging

from core.config_utils.config import Config

logger = logging.getLogger(__name__)


# ============================================================
# 数据类（保持与 UI 层兼容）
# ============================================================

@dataclass
class KnowledgeItem:
    """知识条目（兼容旧接口）"""
    id: str
    question: str
    answer: str
    keywords: List[str] = field(default_factory=list)
    category: str = "通用"
    score: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'KnowledgeItem':
        return cls(
            id=data.get("id", ""),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            keywords=data.get("keywords", []),
            category=data.get("category", "通用"),
            score=data.get("score", 1.0)
        )


@dataclass
class ProductItem:
    """商品信息（兼容旧接口）"""
    id: str
    name: str
    price: float
    category: str
    description: str
    specifications: Dict[str, str] = field(default_factory=dict)
    stock: int = 0
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'ProductItem':
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            price=data.get("price", 0.0),
            category=data.get("category", ""),
            description=data.get("description", ""),
            specifications=data.get("specifications", {}),
            stock=data.get("stock", 0),
            keywords=data.get("keywords", [])
        )


class RAGSearchResult:
    """RAG搜索结果"""
    def __init__(self):
        self.query = ""
        self.rewritten_query = ""
        self.retrieved_items: List[Tuple[KnowledgeItem, float]] = []
        self.context_text = ""
        # confidence 为兼容字段，当前存储检索匹配分，不代表回答可信概率。
        self.confidence = 0.0
        self.score_type = ""
        self.search_method = "vector"
        self.final_prompt = ""

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "rewritten_query": self.rewritten_query,
            "retrieved_items": [
                {
                    "id": item.id,
                    "question": item.question,
                    "score": score,
                    "score_type": self.score_type
                }
                for item, score in self.retrieved_items
            ],
            "context_text": self.context_text,
            "retrieval_score": self.confidence,
            "score_type": self.score_type,
            "confidence": self.confidence,
            "search_method": self.search_method,
            "final_prompt": self.final_prompt
        }


# ============================================================
# 兼容函数（供旧代码调用）
# ============================================================

def truncate_text(text: Optional[str], max_chars: int) -> str:
    """截断文本（兼容旧接口）"""
    if not text:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def trim_history(history: Optional[list], max_messages: int, max_chars: int) -> list:
    """裁剪历史消息（兼容旧接口）"""
    if not history:
        return []
    normalized = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        normalized.append({"role": role, "content": str(content)})
    if not normalized:
        return []
    if max_messages is not None and max_messages > 0:
        normalized = normalized[-max_messages:]
    if max_chars is None or max_chars <= 0:
        return normalized
    picked_reversed = []
    total = 0
    for msg in reversed(normalized):
        content = msg.get("content", "")
        if not content:
            continue
        next_total = total + len(content)
        if next_total > max_chars:
            if not picked_reversed:
                picked_reversed.append({
                    "role": msg.get("role", "user"),
                    "content": truncate_text(content, max_chars),
                })
            break
        picked_reversed.append(msg)
        total = next_total
    return list(reversed(picked_reversed))


def build_system_prompt(context_text: Optional[str]) -> str:
    """构建系统提示（兼容旧接口，实际由 prompt.py 处理）"""
    from core.rag.prompt import BASE_ROLE, RAG_RULES
    if context_text:
        return f"{BASE_ROLE}\n\n{RAG_RULES}\n\n---知识库内容---\n{context_text}\n---知识库内容结束---"
    return BASE_ROLE


def build_messages(system_prompt: str, user_message: str, history: Optional[list] = None) -> list:
    """构建消息列表（兼容旧接口，实际由 message_builder.py 处理）"""
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for msg in history:
            role = msg.get("role")
            content = msg.get("content")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": str(content)})
    messages.append({"role": "user", "content": user_message})
    return messages


def format_prompt_preview(messages: list) -> str:
    """格式化 prompt 预览（兼容旧接口）"""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "system":
            parts.append(f"【System Message】\n{content}")
        elif role == "user":
            parts.append(f"【User Message】\n{content}")
        elif role == "assistant":
            parts.append(f"【Assistant Message】\n{content}")
    return "\n\n".join(parts)


# ============================================================
# KnowledgeStore（重构为基于 Milvus + doc_loader）
# ============================================================

class KnowledgeStore:
    """知识库存储 - 基于 Milvus 向量检索"""

    def __init__(self):
        self.config = Config()
        self._last_search_result: Optional[RAGSearchResult] = None
        self._rag_chain = None
        self._doc_loader = None

    def _get_rag_chain(self):
        if self._rag_chain is None:
            from core.rag.rag_chain import get_rag_chain
            self._rag_chain = get_rag_chain()
        return self._rag_chain

    def _get_doc_loader(self):
        if self._doc_loader is None:
            from core.rag.doc_loader import get_doc_loader
            self._doc_loader = get_doc_loader()
        return self._doc_loader

    def search(self, query: str, threshold: float = None) -> List[Tuple[KnowledgeItem, float]]:
        """搜索知识库"""
        if threshold is None:
            threshold = self.config.get("similarity_threshold", 0.4)

        chain = self._get_rag_chain()
        results = chain.search(query)

        # 转换为 KnowledgeItem 格式（兼容旧接口）
        items = []
        for chunk, score, score_type in results:
            # 从 chunk 构建 KnowledgeItem
            item = KnowledgeItem(
                id=chunk.source_file,
                question=chunk.heading or chunk.content[:50],
                answer=chunk.content,
                keywords=[],
                category=chunk.category or "通用",
                score=score
            )
            items.append((item, score))

        self._last_search_result = RAGSearchResult()
        self._last_search_result.query = query
        self._last_search_result.rewritten_query = chain.rewrite_query(query)
        self._last_search_result.retrieved_items = items
        self._last_search_result.confidence = items[0][1] if items else 0.0
        self._last_search_result.score_type = results[0][2] if results else ""
        self._last_search_result.search_method = self._last_search_result.score_type

        return items

    def get_last_search_result(self) -> Optional[RAGSearchResult]:
        return self._last_search_result

    def reload_vector_index(self, progress_callback: Callable[[str, int, int], None] = None) -> Tuple[bool, str]:
        """重新加载向量索引（实际从 Milvus 加载）"""
        try:
            vs = self._get_doc_loader()._vector_store
            stats = vs.get_stats()
            return True, f"向量索引已加载，共 {stats.get('num_entities', 0)} 条记录"
        except Exception as e:
            logger.error("加载向量索引失败: %s", e)
            return False, str(e)

    def rebuild_vector_index(self, progress_callback: Callable[[str, int, int], None] = None) -> Tuple[bool, str]:
        """重建向量索引（清空 Milvus 后从本地 Markdown 文件重新索引）"""
        try:
            from core.rag.vector_store import get_vector_store
            loader = self._get_doc_loader()
            vs = get_vector_store()

            # 1. 清空 Milvus Collection
            try:
                vs._client.drop_collection(vs._collection_name)
                vs._create_collection()
                logger.info("Milvus Collection 已清空并重建")
            except Exception as e:
                logger.error("清空 Milvus 失败: %s", e)
                return False, f"清空 Milvus 失败: {e}"

            # 2. 重置本地索引（使 hash 检查不再跳过已存在的文档）
            loader._index = {}
            loader._save_index()

            # 3. 扫描 documents 目录，逐个重新索引
            doc_dir = loader._documents_dir
            md_files = sorted(f for f in os.listdir(doc_dir) if f.endswith('.md'))
            total = len(md_files)
            if total == 0:
                return True, "documents 目录下没有 Markdown 文件，无需重建"

            added_total = 0
            for i, filename in enumerate(md_files):
                file_path = os.path.join(doc_dir, filename)
                inferred_cat = self._infer_category_from_filename(filename)
                if progress_callback:
                    progress_callback("重建索引", i + 1, total)
                result = loader.load_from_file(file_path, category=inferred_cat)
                added_total += result.get("added", 0) + result.get("updated", 0)
                logger.info("已重建 [%d/%d]: %s → +%d", i + 1, total, filename,
                            result.get("added", 0) + result.get("updated", 0))

            if progress_callback:
                progress_callback("完成", total, total)

            return True, f"向量索引重建完成，共处理 {total} 个文档，新增/更新 {added_total} 条记录"
        except Exception as e:
            logger.error("重建向量索引失败: %s", e)
            return False, str(e)

    @property
    def last_vector_index_error(self) -> Optional[dict]:
        return None

    # ============================================================
    # 兼容旧 UI 接口（知识条目 CRUD，基于 PostgreSQL）
    # ============================================================

    @property
    def items(self) -> List[KnowledgeItem]:
        """获取所有知识条目（从 PostgreSQL 读取，供 UI 树形面板使用）"""
        return self._get_all_items()

    def _get_all_items(self) -> List[KnowledgeItem]:
        """从 PostgreSQL 的 product 表中读取商品同步的知识条目（模拟旧知识库）"""
        # 新架构下知识来自 Milvus，UI 的"知识列表"改为显示文档列表
        # 这里返回空列表，保持 UI 不报错
        try:
            from core.rag.doc_loader import get_doc_loader
            from core.rag.vector_store import get_vector_store
            loader = get_doc_loader()
            vs = get_vector_store()
            stats = loader.get_document_stats()
            items = []
            for doc_name, doc_info in stats.get('documents', {}).items():
                # 从 Milvus 获取实际 chunk 数据
                chunks = vs.get_all_chunks(source_file=doc_name)
                for chunk_data in chunks:
                    chunk_hash = chunk_data.get('id', '')
                    heading = chunk_data.get('section', '')
                    items.append(KnowledgeItem(
                        id=chunk_hash,
                        question=heading[:50],
                        answer="",
                        keywords=[],
                        category=doc_info.get('category', '通用'),
                        score=1.0
                    ))
            return items
        except Exception:
            return []

    def get_all_items(self) -> List[KnowledgeItem]:
        return self.items

    def add_item(self, question: str, answer: str, keywords: List[str], category: str = "通用") -> KnowledgeItem:
        """添加知识条目（保留接口，实际通过文档上传管理）"""
        logger.warning("add_item() 已废弃，请使用 Markdown 文档上传功能代替")
        # 不实际写入，保持 UI 不报错
        item_id = f"K{len(self.items) + 1:03d}"
        item = KnowledgeItem(id=item_id, question=question, answer=answer, keywords=keywords, category=category)
        self.items.append(item)
        return item

    def delete_item(self, item_id: str) -> bool:
        """删除知识条目（保留接口）"""
        logger.warning("delete_item() 已废弃，请使用文档管理代替")
        return False

    def update_item(self, item_id: str, **kwargs) -> bool:
        """更新知识条目（保留接口）"""
        logger.warning("update_item() 已废弃，请使用文档管理代替")
        return False

    def get_item_by_id(self, item_id: str) -> Optional[KnowledgeItem]:
        """根据 ID 获取知识条目"""
        for item in self.items:
            if item.id == item_id:
                return item
        return None


# ============================================================
# ProductStore（重构为基于 PostgreSQL）
# ============================================================

class ProductStore:
    """商品存储 - 基于 PostgreSQL"""

    def __init__(self):
        self.config = Config()
        self._product_db = None
        self._knowledge_store = None

    def _get_product_db(self):
        if self._product_db is None:
            from core.data.product_db import get_product_db
            self._product_db = get_product_db()
        return self._product_db

    def _get_knowledge_store(self):
        if self._knowledge_store is None:
            self._knowledge_store = KnowledgeStore()
        return self._knowledge_store

    def add_product(self, name: str, price: float, category: str, description: str,
                    specifications: Dict[str, str] = None, stock: int = 0,
                    keywords: List[str] = None) -> ProductItem:
        """添加商品"""
        result = self._get_product_db().add_product(
            name=name, price=price, category=category,
            description=description, specifications=specifications,
            stock=stock, keywords=keywords
        )
        return ProductItem.from_dict(result)

    def delete_product(self, product_id: str) -> bool:
        """删除商品"""
        return self._get_product_db().delete_product(product_id)

    def update_product(self, product_id: str, **kwargs) -> bool:
        """更新商品"""
        return self._get_product_db().update_product(product_id, **kwargs)

    def get_product(self, product_id: str) -> Optional[ProductItem]:
        """根据 ID 获取商品"""
        result = self._get_product_db().get_product(product_id)
        return ProductItem.from_dict(result) if result else None

    def get_all_products(self) -> List[ProductItem]:
        """获取所有商品"""
        results = self._get_product_db().get_all_products()
        return [ProductItem.from_dict(p) for p in results]

    def search_products(self, query: str) -> List[ProductItem]:
        """搜索商品"""
        results = self._get_product_db().search_products(query)
        return [ProductItem.from_dict(p) for p in results]

    def get_categories(self) -> List[str]:
        """获取所有分类"""
        products = self.get_all_products()
        return list(set(p.category for p in products))

    def reload(self):
        """重新加载（从数据库）"""
        self._product_db = None  # 强制重新连接

    # ============================================================
    # 兼容旧 UI 接口（products 属性、add/delete/update/search）
    # ============================================================

    @property
    def products(self) -> List[ProductItem]:
        """获取所有商品（供 UI 使用，兼容旧接口）"""
        return self.get_all_products()

    def add_product_item(self, name: str, price: float, category: str, description: str,
                         specifications: Dict[str, str] = None, stock: int = 0,
                         keywords: List[str] = None) -> ProductItem:
        """添加商品（兼容旧接口名）"""
        return self.add_product(name, price, category, description, specifications, stock, keywords)

    def delete_product_item(self, product_id: str) -> bool:
        """删除商品（兼容旧接口名）"""
        return self.delete_product(product_id)

    def sync_to_knowledge(self) -> Tuple[int, int]:
        """将商品同步到知识库（兼容旧接口）"""
        # 新架构下商品知识通过文档上传管理，此方法返回成功但不执行
        return len(self.products), 0
