# -*- coding: utf-8 -*-
"""
Markdown 文档解析与智能切块
按 ## 标题分割章节，保护表格/代码块/列表结构
"""

import re
import logging
import sys
import io
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# 确保输出编码正确
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """知识块"""
    content: str                      # 文本内容
    heading: str                      # 所属标题路径（如 "## 退换货政策 > ### 退货流程"）
    source_file: str                  # 来源文件路径
    chunk_index: int                  # chunk 序号
    category: str = ""               # 分类标签
    chunk_id: str = ""                # 向量库主键（入库后填充）

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "heading": self.heading,
            "source_file": self.source_file,
            "chunk_index": self.chunk_index,
            "category": self.category,
            "chunk_id": self.chunk_id
        }


class DocumentParser:
    """Markdown 文档解析器"""

    # 需要保护的代码块标记
    CODE_BLOCK_PATTERNS = [
        r"```[\s\S]*?```",      # 三重反引号代码块
        r"`[^`]+`",             # 行内代码
    ]

    # 需要保护的表格
    TABLE_PATTERN = r"\|[^|\n]*\|[^\n]*\n(\|[^|\n]*\|[^\n]*\n)*"

    # 需要保护的列表
    LIST_PATTERN = r"(?m)^[ \t]*[-*+][ \t].*\n(^[ \t]+.*\n)*"

    def __init__(self, max_chunk_size: int = 500, overlap: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self._code_placeholders: List[str] = []
        self._table_placeholders: List[str] = []
        self._list_placeholders: List[str] = []

    def parse(self, markdown_text: str, source_file: str = "",
              category: str = "") -> List[Chunk]:
        """解析 Markdown 文档，返回 chunk 列表"""
        # 1. 保护特殊结构
        protected_text, code_blocks, tables, lists = self._protect_structures(markdown_text)

        # 2. 按标题分割章节
        sections = self._split_by_headings(protected_text)

        # 3. 处理每个章节
        chunks: List[Chunk] = []
        global_chunk_index = 0

        for section_heading, section_content in sections:
            # 恢复被保护的结构
            restored_content = self._restore_structures(section_content, code_blocks, tables, lists)

            # 按句子边界切块
            sentence_chunks = self._split_by_sentences(restored_content)

            # 生成 chunk
            for i, chunk_text in enumerate(sentence_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text or len(chunk_text) < 10:
                    continue

                heading = f"#{section_heading}" if section_heading else ""
                chunk = Chunk(
                    content=chunk_text,
                    heading=heading,
                    source_file=source_file,
                    chunk_index=global_chunk_index,
                    category=category
                )
                chunks.append(chunk)
                global_chunk_index += 1

        logger.info("文档解析完成: %s, 生成 %d 个 chunk", source_file, len(chunks))
        return chunks

    def _protect_structures(self, text: str) -> tuple:
        """保护代码块、表格、列表不被切断"""
        code_blocks = []
        tables = []
        lists = []

        # 保护代码块
        def replace_code(match):
            placeholder = f"[[CODE_BLOCK_{len(code_blocks)}]]"
            code_blocks.append(match.group(0))
            return placeholder

        text = re.sub(r"```[\s\S]*?```", replace_code, text)
        text = re.sub(r"`[^`]+`", lambda m: f"[[INLINE_CODE_{len(code_blocks)}]]", text)
        code_blocks.extend(re.findall(r"`[^`]+`", text))

        # 保护表格
        def replace_table(match):
            placeholder = f"[[TABLE_{len(tables)}]]"
            tables.append(match.group(0))
            return placeholder

        text = re.sub(self.TABLE_PATTERN, replace_table, text, flags=re.MULTILINE)

        # 保护列表
        def replace_list(match):
            placeholder = f"[[LIST_{len(lists)}]]"
            lists.append(match.group(0))
            return placeholder

        text = re.sub(self.LIST_PATTERN, replace_list, text, flags=re.MULTILINE)

        return text, code_blocks, tables, lists

    def _restore_structures(self, text: str, code_blocks: List[str],
                           tables: List[str], lists: List[str]) -> str:
        """恢复被保护的结构"""
        # 恢复代码块
        for i, block in enumerate(code_blocks):
            text = text.replace(f"[[CODE_BLOCK_{i}]]", block)
            text = text.replace(f"[[INLINE_CODE_{i}]]", block)

        # 恢复表格
        for i, table in enumerate(tables):
            text = text.replace(f"[[TABLE_{i}]]", table)

        # 恢复列表
        for i, list_item in enumerate(lists):
            text = text.replace(f"[[LIST_{i}]]", list_item)

        return text

    def _split_by_headings(self, text: str) -> List[tuple]:
        """按 ## 标题分割章节"""
        sections = []
        current_heading = ""
        current_content = []
        heading_pattern = re.compile(r"^(#{1,6}\s+.+)$", re.MULTILINE)

        lines = text.split('\n')
        i = 0
        while i < len(lines):
            line = lines[i]
            match = heading_pattern.match(line.strip())
            if match:
                # 保存上一个章节
                if current_content:
                    sections.append((current_heading, '\n'.join(current_content)))
                # 开始新章节
                current_heading = line.strip()
                current_content = []
            else:
                current_content.append(line)
            i += 1

        # 保存最后一个章节
        if current_content:
            sections.append((current_heading, '\n'.join(current_content)))

        return sections

    def _split_by_sentences(self, text: str) -> List[str]:
        """按句子边界切块"""
        if not text:
            return []

        # 按句号、分号、换行分割
        sentences = re.split(r'(?<=[。；！\n])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # 如果单个句子超过最大长度，按字符切分
            if sentence_length > self.max_chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # 按字符切分长句子
                for i in range(0, len(sentence), self.max_chunk_size - self.overlap):
                    chunk = sentence[i:i + self.max_chunk_size]
                    if chunk.strip():
                        chunks.append(chunk)
                continue

            # 如果加上当前句子超过限制，先输出当前 chunk
            if current_length + sentence_length > self.max_chunk_size and current_chunk:
                chunk = ''.join(current_chunk)
                if chunk.strip():
                    chunks.append(chunk)
                # 保留末尾重叠部分
                if self.overlap > 0:
                    overlap_text = current_chunk[-self.overlap:] if len(current_chunk) >= self.overlap else current_chunk
                    current_chunk = [sentence]
                    current_length = sentence_length
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # 处理剩余的 chunk
        if current_chunk:
            chunk = ''.join(current_chunk)
            if chunk.strip():
                chunks.append(chunk)

        return chunks


# 便捷函数
def parse_markdown(markdown_text: str, source_file: str = "",
                  category: str = "", max_chunk_size: int = 500,
                  overlap: int = 50) -> List[Chunk]:
    """解析 Markdown 文档的便捷函数"""
    parser = DocumentParser(max_chunk_size=max_chunk_size, overlap=overlap)
    return parser.parse(markdown_text, source_file, category)
