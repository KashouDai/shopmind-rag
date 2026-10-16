# -*- coding: utf-8 -*-
"""
文档加载器
调用 document_parser 解析后，查本地 doc_index.json 做 file_hash/chunk_hash 比对，增量更新 Milvus
"""

import os
import json
import logging
import hashlib
import tempfile
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.rag.document_parser import DocumentParser, Chunk
from core.rag.vector_store import get_vector_store
from core.rag.embedding import EmbeddingClient
from core.infra.file_lock import FileLock

logger = logging.getLogger(__name__)

INDEX_SCHEMA_VERSION = 2


class DocLoader:
    """文档加载器"""

    def __init__(self, index_file: str = None):
        self._parser = DocumentParser()
        self._vector_store = get_vector_store()
        self._embedding_client = EmbeddingClient()

        # 本地索引文件路径
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self._index_file = index_file or os.path.join(base_dir, "data", "doc_index.json")
        self._documents_dir = os.path.join(base_dir, "data", "documents")
        os.makedirs(self._documents_dir, exist_ok=True)
        self._index = self._load_index()

    def _load_index(self) -> Dict[str, Any]:
        """加载本地索引"""
        if os.path.exists(self._index_file):
            try:
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error("加载索引文件失败: %s", e)
                return {}
        return {}

    def _save_index(self):
        """使用文件锁和临时文件原子保存本地索引"""
        temp_path = None
        try:
            os.makedirs(os.path.dirname(self._index_file), exist_ok=True)
            with FileLock(self._index_file):
                fd, temp_path = tempfile.mkstemp(
                    dir=os.path.dirname(self._index_file),
                    prefix=".doc_index_",
                    suffix=".tmp"
                )
                with os.fdopen(fd, 'w', encoding='utf-8') as f:
                    json.dump(self._index, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temp_path, self._index_file)
                temp_path = None
        except Exception as e:
            logger.error("保存索引文件失败: %s", e)
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _compute_file_hash(self, content: str) -> str:
        """计算文件 hash"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _compute_chunk_hash(self, content: str, heading: str = "") -> str:
        """计算 chunk hash"""
        payload = f"{heading}\0{content}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    @staticmethod
    def _make_chunk_id(source_file: str, chunk_hash: str, occurrence: int) -> str:
        """生成跨进程稳定的 chunk ID。"""
        payload = f"{source_file}\0{chunk_hash}\0{occurrence}"
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:32]

    def _prepare_chunk_records(
        self,
        chunks: List[Chunk],
        source_file: str,
        file_hash: str,
        category: str
    ) -> List[Dict[str, Any]]:
        """将解析结果转换为带稳定 ID 的 Milvus 元数据。"""
        records = []
        occurrences: Dict[str, int] = {}

        for chunk in chunks:
            chunk_hash = self._compute_chunk_hash(chunk.content, chunk.heading)
            occurrence = occurrences.get(chunk_hash, 0)
            occurrences[chunk_hash] = occurrence + 1
            chunk_id = self._make_chunk_id(source_file, chunk_hash, occurrence)
            records.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "chunk_hash": chunk_hash,
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "section": chunk.heading,
                "source_file": source_file,
                "category": category,
                "file_hash": file_hash
            })

        return records

    def _invalidate_rag_cache(self):
        """通知 RAG 链重建 BM25 统计量。"""
        try:
            from core.rag.rag_chain import get_rag_chain
            get_rag_chain().invalidate_bm25_stats()
        except Exception as e:
            logger.warning("刷新 RAG 检索缓存失败: %s", e)

    def load_from_file(self, file_path: str, category: str = "") -> Dict[str, int]:
        """
        从文件加载并增量更新 Milvus

        Args:
            file_path: Markdown 文件路径
            category: 分类标签

        Returns:
            {
                "added": 新增数量,
                "updated": 更新数量,
                "skipped": 跳过数量,
                "deleted": 删除数量
            }
        """
        # 读取文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
        except Exception as e:
            logger.error("读取文件失败: %s", e)
            return {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}

        # 复制文件到 documents 目录
        doc_filename = os.path.basename(file_path)
        dest_path = os.path.join(self._documents_dir, doc_filename)
        try:
            import shutil
            shutil.copy2(file_path, dest_path)
            logger.info("文档已保存到: %s", dest_path)
        except Exception as e:
            logger.error("保存文档失败: %s", e)
            # 继续处理，不阻断上传流程

        # 计算文件 hash
        file_hash = self._compute_file_hash(markdown_text)
        source_file = doc_filename  # 使用文件名作为 source_file

        logger.info("开始处理文档: %s, file_hash: %s", source_file, file_hash[:8])

        old_doc = self._index.get(source_file, {})
        current_schema = (
            isinstance(old_doc, dict)
            and old_doc.get("_schema_version", 0) >= INDEX_SCHEMA_VERSION
        )

        # 旧索引版本需要重建，避免只按内容 hash 导致重复内容互相覆盖。
        if (
            current_schema
            and old_doc.get("file_hash") == file_hash
            and old_doc.get("category", "") == category
        ):
            logger.info("文档未变更，跳过: %s", source_file)
            return {"added": 0, "updated": 0, "skipped": len(old_doc.get("chunks", {})), "deleted": 0}

        # 解析文档
        chunks = self._parser.parse(
            markdown_text=markdown_text,
            source_file=source_file,
            category=category
        )

        if not chunks:
            logger.warning("文档解析后没有生成 chunk: %s", source_file)
            return {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}

        current_records = self._prepare_chunk_records(
            chunks, source_file, file_hash, category
        )
        current_ids = {record["id"] for record in current_records}

        old_chunks = old_doc.get("chunks", {}) if current_schema else {}
        old_ids = set(old_chunks.keys())
        records_to_add = []
        records_to_update = []
        skipped_count = 0
        for record in current_records:
            old_record = old_chunks.get(record["id"])
            if old_record is None:
                records_to_add.append(record)
            elif (
                old_record.get("chunk_hash", "") == record["chunk_hash"]
                and old_record.get("chunk_index") == record["chunk_index"]
                and old_record.get("section", "") == record["section"]
                and old_record.get("source_file", "") == record["source_file"]
                and old_record.get("category", "") == record["category"]
                and old_record.get("file_hash", "") == record["file_hash"]
            ):
                skipped_count += 1
            else:
                records_to_update.append(record)

        ids_to_delete = list(old_ids - current_ids) if current_schema else []

        records_to_write = records_to_add + records_to_update
        vectors_to_write = []
        if records_to_write:
            vectors = self._embedding_client.embed_texts(
                [record["content"] for record in records_to_write]
            )
            if (
                not vectors
                or len(vectors) != len(records_to_write)
                or any(vector is None for vector in vectors)
            ):
                logger.error("文档向量化不完整，取消本次索引更新: %s", source_file)
                return {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}
            vectors_to_write = vectors

        # 先写入新向量，再删除旧向量，避免中途失败造成知识缺口。
        added_count = len(records_to_add)
        updated_count = len(records_to_update)
        if records_to_write:
            written_ids = self._vector_store.add_vectors(vectors_to_write, records_to_write)
            if len(written_ids) != len(records_to_write):
                logger.error("向量写入不完整，保留旧索引等待重试: %s", source_file)
                return {"added": 0, "updated": 0, "skipped": 0, "deleted": 0}

        deleted_count = 0
        if not current_schema and old_doc.get("chunks"):
            # 旧索引只保存 chunk_hash，旧向量主键是随机值；只能按旧 hash
            # 精确删除，不能按 file_hash 删除刚写入的新向量。
            for old_chunk_hash in old_doc.get("chunks", {}):
                deleted_count += self._vector_store.delete_by_hash(
                    file_hash="", chunk_hash=old_chunk_hash
                )
        elif ids_to_delete:
            deleted_count = self._vector_store.delete_by_ids(ids_to_delete)

        # 更新本地索引
        old_version = old_doc.get('version', '1.0')
        new_version = str(float(old_version) + 0.1) if old_version else '1.0'

        self._index[source_file] = {
            "_schema_version": INDEX_SCHEMA_VERSION,
            "file_hash": file_hash,
            "category": category,
            "updated_at": datetime.utcnow().isoformat(),
            "version": new_version,
            "chunks": {}
        }
        for record in current_records:
            self._index[source_file]["chunks"][record["id"]] = {
                "chunk_id": record["id"],
                "chunk_hash": record["chunk_hash"],
                "chunk_index": record["chunk_index"],
                "section": record["section"],
                "source_file": record["source_file"],
                "category": record["category"],
                "file_hash": record["file_hash"]
            }

        self._save_index()
        self._invalidate_rag_cache()

        logger.info(
            "文档处理完成: %s, 新增: %d, 更新: %d, 删除: %d",
            source_file, added_count, updated_count, deleted_count
        )

        return {
            "added": added_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "deleted": deleted_count
        }

    def remove_document(self, source_file: str) -> bool:
        """删除文档及其所有 chunk"""
        if source_file not in self._index:
            return False

        # 删除 documents 目录中的文件
        doc_path = os.path.join(self._documents_dir, source_file)
        if os.path.exists(doc_path):
            try:
                os.remove(doc_path)
                logger.info("已删除文档文件: %s", doc_path)
            except Exception as e:
                logger.error("删除文档文件失败: %s", e)

        old_doc = self._index[source_file]
        current_schema = old_doc.get("_schema_version", 0) >= INDEX_SCHEMA_VERSION
        if current_schema:
            self._vector_store.delete_by_ids(list(old_doc.get("chunks", {}).keys()))
        else:
            file_hash = old_doc.get("file_hash", "")
            if file_hash:
                self._vector_store.delete_by_hash(file_hash=file_hash)

        del self._index[source_file]
        self._save_index()
        self._invalidate_rag_cache()
        logger.info("文档删除完成: %s", source_file)
        return True

    def get_index(self) -> Dict[str, Any]:
        """获取本地索引"""
        return self._index

    def get_document_stats(self) -> Dict[str, Any]:
        """获取文档统计信息"""
        stats = {
            "total_documents": len(self._index),
            "total_chunks": 0,
            "documents": {}
        }

        for doc_name, doc_info in self._index.items():
            chunk_count = len(doc_info.get("chunks", {}))
            stats["total_chunks"] += chunk_count
            stats["documents"][doc_name] = {
                "file_hash": doc_info.get("file_hash", "")[:8] + "...",
                "category": doc_info.get("category", ""),
                "chunk_count": chunk_count,
                "updated_at": doc_info.get("updated_at", "")
            }

        return stats


# 全局单例
_doc_loader: Optional[DocLoader] = None


def get_doc_loader() -> DocLoader:
    """获取文档加载器单例"""
    global _doc_loader
    if _doc_loader is None:
        _doc_loader = DocLoader()
    return _doc_loader
