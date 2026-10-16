# -*- coding: utf-8 -*-
"""
Prompt 管理模块
统一管理系统中的各种 Prompt 模板和构建函数
"""

import json
from typing import List, Dict, Any, Optional


# ═══════════════════════════════════════════
# 1. 基础角色定义
# ═══════════════════════════════════════════

BASE_ROLE = "你是一个专业的电商客服助手，负责解答用户关于商品、订单、物流、退换货等问题。请用友好、专业的语气回复，回答要简洁有帮助。"


# ═══════════════════════════════════════════
# 2. RAG 相关的 prompt
# ═══════════════════════════════════════════

RAG_RULES = """## 回复规则
1. 优先基于知识库内容回答，引用来源标注 [来源文件 > 章节名]
2. 如果知识库内容无法回答该问题，明确回复："抱歉，该问题超出知识库范围，建议联系人工客服获取更准确的解答"
3. 回复简洁，控制在 200 字以内
4. 涉及具体金额、时限、政策条款等关键信息，必须与知识库原文一致，禁止估算或改写"""


# ═══════════════════════════════════════════
# 3. Agent 路由相关的 prompt
# ═══════════════════════════════════════════

AGENT_ROUTING_PROMPT = """你是一个电商客服 Agent。根据用户问题，选择最合适的工具：

可用工具：
1. search_products — 查询具体商品信息（价格、库存、规格参数、商品描述）
2. search_knowledge — 从知识库检索通用问题的答案（退换货政策、物流说明、支付方式、促销活动规则等）
3. transfer_to_human — 转接人工客服

选择规则：
- 用户提到具体商品名称并询问价格、库存、规格 → search_products
- 用户问政策、流程、规则类问题 → search_knowledge
- 用户明确要求人工服务，或问题超出知识库范围 → transfer_to_human
- 如果同时涉及商品和知识库（如"XX商品的退货流程"），两个工具都调用"""


# ═══════════════════════════════════════════
# 4. 商品工具返回后的回复约束
# ═══════════════════════════════════════════

PRODUCT_REPLY_RULES = """以下是商品查询结果，请按以下规则回复：
1. 价格、库存数量等数值必须与查询结果完全一致，禁止四舍五入或估算
2. 只回复用户关心的字段，不要罗列所有信息
3. 如果用户问的问题在查询结果中找不到对应字段，明确告知"""


# ═══════════════════════════════════════════
# 5. 构建函数
# ═══════════════════════════════════════════


def build_rag_system_prompt(context_chunks: List[Dict[str, str]]) -> str:
    """构建 RAG 场景的完整 system prompt"""
    prompt = f"{BASE_ROLE}\n\n{RAG_RULES}"

    if context_chunks:
        prompt += "\n\n---知识库内容---\n"
        for chunk in context_chunks:
            source = chunk.get("source_file", "未知")
            section = chunk.get("section", "")
            label = f"[{source} > {section}]" if section else f"[{source}]"
            prompt += f"\n{label}\n{chunk.get('content', '')}\n"
        prompt += "\n---知识库内容结束---"

    return prompt


def build_product_reply_prompt(product_data: List[Dict[str, Any]]) -> str:
    """构建商品查询结果的用户回复 prompt"""
    prompt = f"{BASE_ROLE}\n\n{PRODUCT_REPLY_RULES}"
    prompt += f"\n\n查询结果：\n{json.dumps(product_data, ensure_ascii=False, indent=2)}"
    return prompt


def build_agent_routing_prompt() -> str:
    """构建 Agent 路由的 system prompt"""
    return f"{BASE_ROLE}\n\n{AGENT_ROUTING_PROMPT}"


def build_answerability_hint(answerability: str) -> str:
    """根据回答依据等级添加提示语"""
    if answerability == "partial":
        return "\n\n以上信息仅供参考，如需确认请联系人工客服。"
    return ""


def format_chunk_label(chunk: Dict[str, str]) -> str:
    """格式化 chunk 来源标签"""
    source = chunk.get("source_file", "未知")
    section = chunk.get("section", "")
    if section:
        return f"[{source} > {section}]"
    return f"[{source}]"
