# -*- coding: utf-8 -*-
"""
RAG 检索链模块
双路检索：向量检索 + BM25 关键词检索，RRF 融合排序 + Cross-Encoder 重排
"""

import logging
import re
import json
import math
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from core.config_utils.config import Config
from core.rag.vector_store import get_vector_store
from core.rag.embedding import EmbeddingClient
from core.rag.document_parser import Chunk

logger = logging.getLogger(__name__)


# RRF 融合参数
RRF_K = 6  # 统一的 RRF 常数，两路权重对称


@dataclass
class Bm25Stats:
    """BM25 预计算的统计量，每个 chunk 构建时同步维护"""

    # N: 总文档数
    N: int = 0
    # avgdl: 平均文档长度（字符数）
    avgdl: float = 0.0
    # idf: 每个词项的逆文档频率，key=词项(lower), value=idf值
    idf: Dict[str, float] = field(default_factory=dict)
    # df: 每个词项的文档频率
    df: Dict[str, int] = field(default_factory=dict)
    # doc_lens: 每个 chunk_id 的文档长度（字符数）
    doc_lens: Dict[str, int] = field(default_factory=dict)
    # token_freqs: 每个 chunk_id 的词频索引，key=chunk_id, value={token: freq}
    token_freqs: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @classmethod
    def build(cls, chunks: List[Chunk], chunk_ids: Optional[List[str]] = None) -> "Bm25Stats":
        """从 chunk 列表构建完整 BM25 统计量"""
        if not chunks:
            return cls()
        if chunk_ids is not None and len(chunk_ids) != len(chunks):
            raise ValueError("chunk_ids 数量与 chunks 数量不一致")

        N = len(chunks)
        total_len = 0
        df: Dict[str, int] = {}  # document frequency: 包含该词的文档数
        idf: Dict[str, float] = {}
        doc_lens: Dict[str, int] = {}
        token_freqs: Dict[str, Dict[str, int]] = {}

        for index, chunk in enumerate(chunks):
            cid = (
                chunk_ids[index]
                if chunk_ids is not None
                else (chunk.chunk_id or f"{chunk.source_file}:{chunk.chunk_index}")
            )
            text = chunk.content
            total_len += len(text)
            tokens = _tokenize_for_bm25(text)
            # 提取 token（去重用于 IDF 计算）
            tokens_set = set(tokens)
            for t in tokens_set:
                df[t] = df.get(t, 0) + 1
            # 记录词频（含重复用于 TF 计算）
            freq = _token_freqs(tokens)
            token_freqs[cid] = freq
            doc_lens[cid] = len(text)

        # 计算 IDF
        for t, df_count in df.items():
            idf[t] = math.log((N - df_count + 0.5) / (df_count + 0.5)) + 0.5

        avgdl = total_len / N if N > 0 else 1.0
        stats = cls(N=N, avgdl=avgdl, idf=idf, df=df)
        stats.doc_lens = doc_lens
        stats.token_freqs = token_freqs
        return stats

    @staticmethod
    def _ensure_key(d: dict, key: str):
        if key not in d:
            d[key] = {}

    def update(self, chunk: Chunk, operation: str = "add"):
        """增量更新统计量：add（新增）/ delete（删除）"""
        cid = chunk.chunk_id or f"{chunk.source_file}:{chunk.chunk_index}"
        text = chunk.content
        freq = _token_freqs(_tokenize_for_bm25(text))

        if operation == "add":
            self._remove_document(cid)
            self.N += 1
            self.doc_lens[cid] = len(text)
            self.token_freqs[cid] = freq
            for token in freq:
                self.df[token] = self.df.get(token, 0) + 1

        elif operation == "delete":
            self._remove_document(cid)

        else:
            raise ValueError(f"不支持的 BM25 更新操作: {operation}")

        self.avgdl = (
            sum(self.doc_lens.values()) / self.N
            if self.N
            else 0.0
        )
        self.idf = {
            token: math.log((self.N - df_count + 0.5) / (df_count + 0.5)) + 0.5
            for token, df_count in self.df.items()
        }

    def _remove_document(self, chunk_id: str) -> None:
        """移除一个已有 chunk 对统计量的贡献。"""
        freq = self.token_freqs.pop(chunk_id, None)
        if freq is None:
            return

        for token in freq:
            remaining = self.df.get(token, 0) - 1
            if remaining > 0:
                self.df[token] = remaining
            else:
                self.df.pop(token, None)
        self.doc_lens.pop(chunk_id, None)
        self.N = max(0, self.N - 1)

    def score_query(self, keywords: List[str], chunk_id: str) -> float:
        """对单个 chunk 计算 BM25 分数"""
        if chunk_id not in self.token_freqs:
            return 0.0
        chunk_freqs = self.token_freqs[chunk_id]
        chunk_len = self.doc_lens.get(chunk_id, 1)
        if not self.N or chunk_len <= 0:
            return 0.0

        score = 0.0
        k1 = 1.5
        b = 0.75
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in self.idf:
                continue
            idf_val = self.idf[kw_lower]
            tf = chunk_freqs.get(kw_lower, 0)
            if tf <= 0:
                continue
            # BM25 TF 饱和公式
            tf_sat = tf * (k1 + 1) / (tf + k1 * (1 - b + b * chunk_len / self.avgdl))
            score += idf_val * tf_sat
        return score


def _tokenize_for_bm25(text: str) -> List[str]:
    """为 BM25 提取 token，文档与查询必须使用同一套规则。"""
    tokens = []
    # 英文单词和小写数字
    tokens.extend(re.findall(r'[a-zA-Z]+', text.lower()))
    # 中文连续片段及 2-gram
    for seg in re.findall(r'[一-鿿]{2,}', text):
        tokens.append(seg.lower())
        for i in range(len(seg) - 1):
            tokens.append(seg[i:i+2].lower())
    return tokens


def _token_freqs(tokens: List[str]) -> Dict[str, int]:
    """计算 token 词频"""
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return freq


class SearchResult:
    """搜索结果"""
    chunk: Chunk
    vector_score: float
    bm25_score: float
    rrf_score: float
    confidence: float


class RAGChain:
    """RAG 检索链"""

    def __init__(self):
        self._config = Config()
        self._vector_store = get_vector_store()
        self._embedding_client = EmbeddingClient()
        self._synonyms = self._load_synonyms()

        # BM25 统计量（惰性初始化，首次 _keyword_search 时构建）
        self._bm25_stats: Optional[Bm25Stats] = None
        self._bm25_stats_dirty = True

    def _load_synonyms(self) -> Dict[str, List[str]]:
        """加载同义词词库"""
        import os
        synonyms_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "config", "synonyms.json"
        )
        try:
            with open(synonyms_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error("加载同义词词库失败: %s", e)
            return {}

    def rewrite_query(self, query: str) -> str:
        """查询改写：同义词扩展"""
        if not self._synonyms:
            return query

        rewritten = query
        for original, synonyms in self._synonyms.items():
            if original in query:
                for syn in synonyms:
                    if syn not in rewritten:
                        rewritten += f" {syn}"

        return rewritten.strip()

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float, str]]:
        """
        RAG 检索：向量 + BM25 双路检索，RRF 融合 + Cross-Encoder 重排

        Args:
            query: 用户查询
            top_k: 返回数量

        Returns:
            [(Chunk, score, score_type), ...]
            score_type 为 "rerank" 或 "rrf"，分数保留对应检索阶段的原始量纲。
        """
        # 过滤空查询
        if not query or not query.strip():
            logger.info("RAG 检索: 空查询，返回空结果")
            return []

        # 1. 查询改写
        rewritten_query = self.rewrite_query(query)
        queries = [rewritten_query, query]

        # 2. 向量检索
        vector_results = self._vector_search(queries, top_k * 2)

        # 3. BM25 检索
        bm25_results = self._bm25_search(queries, top_k * 2)

        # 4. RRF 融合（召回 top_k*2）
        fused_results = self._rrf_fusion(vector_results, bm25_results, top_k * 2)

        # 5. Cross-Encoder 重排
        reranked_results, score_type = self._rerank(query, fused_results, top_k)

        # 6. 只对重排分数应用阈值；RRF 分数量纲不同，交回答依据校验处理
        results = []
        similarity_threshold = self._config.get("similarity_threshold", 0.1)
        for chunk, rerank_score in reranked_results:
            if score_type == "rerank" and rerank_score < similarity_threshold:
                continue
            results.append((chunk, rerank_score, score_type))

        logger.info(
            "RAG 检索完成: query=%s, results=%d (score_type=%s, threshold=%.2f)",
            query[:30], len(results), score_type, similarity_threshold
        )
        return results

    def _rerank(
        self,
        query: str,
        candidates: List[Tuple[Chunk, float]],
        top_k: int
    ) -> Tuple[List[Tuple[Chunk, float]], str]:
        """Cross-Encoder 重排（SiliconFlow BAAI/bge-reranker-v2-m3）"""
        api_key = self._config.get("embedding_api_key", "") or self._config.get("api_key", "")
        api_base = self._config.get("embedding_api_base_url", "") or self._config.get("api_base_url", "https://api.siliconflow.cn/v1")
        if not api_key:
            logger.warning("API key 未配置，跳过 Cross-Encoder 重排")
            return candidates[:top_k], "rrf"

        try:
            import requests
            documents = [chunk.content for chunk, _ in candidates]
            resp = requests.post(
                f"{api_base}/rerank",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "BAAI/bge-reranker-v2-m3",
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_k, len(candidates)),
                },
                timeout=15
            )
            if resp.status_code == 200:
                result = resp.json()
                rerank_map = {item["index"]: item["relevance_score"] for item in result.get("results", [])}
                reranked = sorted(
                    [(chunk, rerank_map.get(i, 0.0)) for i, (chunk, _) in enumerate(candidates)],
                    key=lambda x: x[1], reverse=True
                )
                logger.info("Cross-Encoder 重排完成: %d -> %d", len(candidates), len(reranked))
                return reranked[:top_k], "rerank"
            else:
                logger.warning("重排 API 返回 %d，跳过重排直接使用 RRF 结果", resp.status_code)
                return candidates[:top_k], "rrf"
        except Exception as e:
            logger.warning("重排异常: %s，跳过重排", e)
            return candidates[:top_k], "rrf"

    def _vector_search(self, queries: List[str], top_k: int) -> Dict[str, Tuple[Chunk, float]]:
        """向量检索"""
        if not self._embedding_client.is_available():
            logger.warning("Embedding 服务不可用，跳过向量检索")
            return {}

        results = {}
        for query in queries:
            query_vec = self._embedding_client.embed_text(query)
            if not query_vec:
                continue

            search_results = self._vector_store.search(query_vec, top_k=top_k)
            for chunk_id, score in search_results:
                # 从 Milvus 获取 chunk 数据
                chunk_data = self._vector_store.get_by_id(chunk_id)
                if chunk_data:
                    chunk = Chunk(
                        content=chunk_data.get("content", ""),
                        heading=chunk_data.get("section", ""),
                        source_file=chunk_data.get("source_file", ""),
                        chunk_index=0,
                        category=chunk_data.get("category", ""),
                        chunk_id=chunk_id
                    )
                    # 保留最高分
                    if chunk_id not in results or score > results[chunk_id][1]:
                        results[chunk_id] = (chunk, score)

        return results

    def _bm25_search(self, queries: List[str], top_k: int) -> Dict[str, Tuple[Chunk, float]]:
        """BM25 关键词检索"""
        results = {}
        for query in queries:
            query_chunks = self._keyword_search(query, top_k)
            for chunk_id, score in query_chunks.items():
                if chunk_id not in results:
                    # 从向量检索结果中获取 Chunk 对象
                    from core.rag.vector_store import get_vector_store
                    vs = get_vector_store()
                    chunk_data = vs.get_by_id(chunk_id)
                    if chunk_data:
                        from core.rag.document_parser import Chunk
                        chunk = Chunk(
                            content=chunk_data.get("content", ""),
                            heading=chunk_data.get("section", ""),
                            source_file=chunk_data.get("source_file", ""),
                            chunk_index=0,
                            category=chunk_data.get("category", ""),
                            chunk_id=chunk_id
                        )
                        results[chunk_id] = (chunk, score)
                elif score > results[chunk_id][1]:
                    results[chunk_id] = (results[chunk_id][0], score)

        return results

    def _ensure_bm25_stats(self):
        """确保 BM25 统计量已构建，构建失败时降级到简单词频匹配"""
        if self._bm25_stats is not None and not self._bm25_stats_dirty:
            return
        try:
            all_chunks = self._vector_store.get_all_chunks()
            if all_chunks:
                # 从 Milvus 数据重建 Chunk 对象列表
                chunks = []
                for cd in all_chunks:
                    chunks.append(Chunk(
                        content=cd.get("content", ""),
                        heading=cd.get("section", ""),
                        source_file=cd.get("source_file", ""),
                        chunk_index=0,
                        category=cd.get("category", ""),
                        chunk_id=cd.get("id", "")
                    ))
                self._bm25_stats = Bm25Stats.build(chunks)
                self._bm25_stats_dirty = False
                logger.info("BM25 统计量构建完成: %d 个 chunk", len(chunks))
            else:
                self._bm25_stats = Bm25Stats()
                self._bm25_stats_dirty = False
        except Exception as e:
            logger.warning("BM25 统计量构建失败，降级到简单词频: %s", e)
            self._bm25_stats = None

    def invalidate_bm25_stats(self):
        """知识库发生变化后使 BM25 统计量失效。"""
        self._bm25_stats = None
        self._bm25_stats_dirty = True

    def _keyword_search(self, query: str, top_k: int) -> Dict[str, float]:
        """关键词搜索（BM25）"""
        keywords = self._extract_keywords(query)
        if not keywords:
            return {}

        # 确保 BM25 统计量已构建
        self._ensure_bm25_stats()
        if self._bm25_stats is None:
            return {}

        scores = {}
        for chunk_id in self._bm25_stats.token_freqs:
            score = self._bm25_stats.score_query(keywords, chunk_id)
            if score > 0:
                scores[chunk_id] = score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return dict(sorted_results)

    def _rrf_fusion(self,
                   vector_results: Dict[str, Tuple[Chunk, float]],
                   bm25_results: Dict[str, Tuple[Chunk, float]],
                   top_k: int) -> List[Tuple[Chunk, float]]:
        """RRF 融合排序（两路使用统一常数，权重对称）"""
        all_chunks = set(vector_results.keys()) | set(bm25_results.keys())

        rrf_scores = {}
        for chunk_id in all_chunks:
            rrf_score = 0.0

            if chunk_id in vector_results:
                rank = list(vector_results.keys()).index(chunk_id) + 1
                rrf_score += 1.0 / (rank + RRF_K)

            if chunk_id in bm25_results:
                rank = list(bm25_results.keys()).index(chunk_id) + 1
                rrf_score += 1.0 / (rank + RRF_K)  # 统一 RRF_K，两路对称

            if rrf_score > 0:
                rrf_scores[chunk_id] = rrf_score

        # 排序取 top_k
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 返回 Chunk 对象
        results = []
        for chunk_id, rrf_score in sorted_chunks:
            chunk = vector_results.get(chunk_id, (None, 0))[0] or bm25_results.get(chunk_id, (None, 0))[0]
            if chunk:
                results.append((chunk, rrf_score))

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        keywords = _tokenize_for_bm25(text)

        # 去除停用词
        stop_words = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个"}
        keywords = [kw for kw in keywords if kw not in stop_words]

        return list(dict.fromkeys(keywords))
# 全局单例
_rag_chain: Optional[RAGChain] = None


def get_rag_chain() -> RAGChain:
    """获取 RAG 检索链单例"""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain
