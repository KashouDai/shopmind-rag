# -*- coding: utf-8 -*-
"""
向量存储模块 — Milvus Server 版本
通过 settings.json 的 milvus_uri 配置连接地址
"""

import os
import logging
import hashlib
from typing import List, Tuple, Optional, Dict
from datetime import datetime

try:
    from pymilvus import MilvusClient
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logging.getLogger(__name__).warning("pymilvus 未安装，向量存储功能不可用")

from core.config_utils.config import Config

logger = logging.getLogger(__name__)


class VectorStore:
    """Milvus 向量存储"""

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
        self._collection_name = self._config.get("milvus_collection", "ecom_knowledge")
        self._dimension = self._config.get("milvus_embedding_dim", 1024)
        self._client: Optional[MilvusClient] = None

        self._connect_milvus()
        self._ensure_collection()

    def _connect_milvus(self):
        """连接 Milvus Server"""
        import warnings as _warnings
        _warnings.filterwarnings('ignore', category=DeprecationWarning)
        _warnings.filterwarnings('ignore', message='.*deprecated.*')
        _warnings.filterwarnings('ignore', message='.*will be removed.*')

        uri = self._config.get("milvus_uri", "http://localhost:19530")
        try:
            self._client = MilvusClient(uri=uri)
            logger.info("Milvus 连接成功: %s", uri)
        except Exception as e:
            logger.error("Milvus 连接失败: %s", e)
            raise

    def _ensure_collection(self):
        """确保 Collection 存在并加载"""
        if not MILVUS_AVAILABLE or not self._client:
            logger.error("Milvus 不可用")
            return

        try:
            if self._client.has_collection(self._collection_name):
                # 加载到内存
                self._client.load_collection(self._collection_name)
                logger.info("加载现有 Collection: %s", self._collection_name)
            else:
                self._create_collection()
        except Exception as e:
            logger.error("Collection 加载失败，拒绝删除现有数据: %s", e)
            raise RuntimeError(f"Milvus Collection 加载失败: {self._collection_name}") from e

    def _create_collection(self):
        """创建 Collection"""
        if not MILVUS_AVAILABLE or not self._client:
            return

        from pymilvus import DataType

        schema = self._client.create_schema(
            enable_dynamic_field=False,
            description="电商知识库向量存储"
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self._dimension)
        schema.add_field(field_name="section", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=500)
        schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="file_hash", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="chunk_hash", datatype=DataType.VARCHAR, max_length=64)

        index_params = self._client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="IVF_FLAT", metric_type="IP", params={"nlist": 128})
        index_params.add_index(field_name="source_file", index_type="INVERTED")
        index_params.add_index(field_name="category", index_type="INVERTED")

        self._client.create_collection(
            collection_name=self._collection_name,
            schema=schema,
            index_params=index_params
        )
        logger.info("Collection 创建完成: %s, 维度: %d", self._collection_name, self._dimension)

    def add_vectors(self, vectors: List[List[float]], metadata: List[Dict[str, str]]) -> List[str]:
        """批量添加向量"""
        if not MILVUS_AVAILABLE or not self._client:
            return []

        if len(vectors) != len(metadata):
            raise ValueError("向量数量与元数据数量不一致")

        ids = [item.get("id") or self._generate_id() for item in metadata]
        if len(ids) != len(set(ids)):
            raise ValueError("同一批向量中存在重复的 chunk_id")

        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            data = []
            for j, vec in enumerate(batch_vectors):
                data.append({
                    "id": batch_ids[j],
                    "content": batch_metadata[j].get("content", ""),
                    "vector": vec,
                    "section": batch_metadata[j].get("section", ""),
                    "source_file": batch_metadata[j].get("source_file", ""),
                    "category": batch_metadata[j].get("category", ""),
                    "file_hash": batch_metadata[j].get("file_hash", ""),
                    "chunk_hash": batch_metadata[j].get("chunk_hash", ""),
                })
            self._client.upsert(collection_name=self._collection_name, data=data)

        self._client.flush(self._collection_name)
        logger.info("批量添加向量完成，共 %d 条", len(ids))
        return ids

    def search(self, query_vector: List[float], top_k: int = 5,
               filter_expr: str = None) -> List[Tuple[str, float]]:
        """搜索相似向量"""
        if not MILVUS_AVAILABLE or not self._client:
            return []

        try:
            search_params = {"metric_type": "IP", "params": {"nprobe": 16}}
            kwargs = dict(
                collection_name=self._collection_name,
                data=[query_vector],
                limit=top_k,
                search_params=search_params,
                output_fields=["id", "content", "section", "source_file", "category", "file_hash", "chunk_hash"]
            )
            if filter_expr:
                kwargs["filter"] = filter_expr
            results = self._client.search(**kwargs)
            return [(hit["id"], float(hit["distance"])) for hit_list in results for hit in hit_list]
        except Exception as e:
            logger.error("向量搜索失败: %s", e)
            return []

    def delete_by_hash(self, file_hash: str, chunk_hash: str = None) -> int:
        """根据 hash 删除向量"""
        if not MILVUS_AVAILABLE or not self._client:
            return 0
        try:
            expr = f"chunk_hash == '{chunk_hash}'" if chunk_hash else f"file_hash == '{file_hash}'"
            self._client.delete(collection_name=self._collection_name, filter=expr)
            return 1
        except Exception as e:
            logger.error("删除向量失败: %s", e)
            return 0

    def delete_by_ids(self, chunk_ids: List[str]) -> int:
        """根据主键批量删除向量"""
        if not MILVUS_AVAILABLE or not self._client or not chunk_ids:
            return 0

        deleted = 0
        for i in range(0, len(chunk_ids), 100):
            batch = [str(item).replace("'", "''") for item in chunk_ids[i:i + 100]]
            expr = "id in [" + ", ".join(f"'{item}'" for item in batch) + "]"
            try:
                self._client.delete(collection_name=self._collection_name, filter=expr)
                deleted += len(batch)
            except Exception as e:
                logger.error("按 ID 删除向量失败: %s", e)
        return deleted

    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """根据 ID 获取向量数据"""
        if not MILVUS_AVAILABLE or not self._client:
            return None
        try:
            results = self._client.query(
                collection_name=self._collection_name,
                filter=f"id == '{chunk_id}'",
                output_fields=["*"]
            )
            return results[0] if results else None
        except Exception as e:
            logger.error("查询向量失败: %s", e)
            return None

    def get_all_chunks(self, source_file: str = None) -> List[Dict]:
        """获取所有 chunk（用于调试）"""
        if not MILVUS_AVAILABLE or not self._client:
            return []
        try:
            if source_file:
                return self._client.query(
                    collection_name=self._collection_name,
                    filter=f"source_file == '{source_file}'",
                    output_fields=["*"]
                )
            else:
                # Milvus v2 requires a valid expression; use a wildcard
                return self._client.query(
                    collection_name=self._collection_name,
                    filter="id != ''",
                    output_fields=["*"]
                )
        except Exception as e:
            logger.error("查询所有 chunk 失败: %s", e)
            return []

    def get_stats(self) -> Dict:
        """获取 Collection 统计信息"""
        if not MILVUS_AVAILABLE or not self._client:
            return {}
        try:
            info = self._client.get_collection_stats(self._collection_name)
            return {"num_entities": int(info.get("row_count", 0))}
        except Exception as e:
            logger.error("获取统计信息失败: %s", e)
            return {}

    def _generate_id(self) -> str:
        return hashlib.md5(f"{datetime.utcnow().timestamp()}{os.urandom(8)}".encode()).hexdigest()[:16]

    def close(self):
        if self._client:
            try:
                self._client.close()
            except Exception:
                pass
        logger.info("Milvus 连接已关闭")


_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
