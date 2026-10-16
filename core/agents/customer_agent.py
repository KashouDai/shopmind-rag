# -*- coding: utf-8 -*-
"""
CustomerAgent — LangGraph 单 Agent 架构

节点：
  classifier    → 意图分类（规则快速路由 + RAG 兜底）
  product_node  → 商品查询（MySQL）
  order_node    → 订单查询（MySQL）
  knowledge_node → RAG 检索（Milvus）
  answerability → 校验生成回答是否有知识库依据
  human_node    → 转人工客服
  generator     → LLM 生成回复

边（条件路由）：
  classifier → product_node   （商品关键词命中）
  classifier → order_node     （订单关键词命中）
  classifier → human_node     （转人工关键词命中）
  classifier → knowledge_node （规则未命中，走 RAG 兜底）
  knowledge_node → generator  （存在候选资料，生成回复）
  knowledge_node → human_node （没有候选资料，转人工）
  product_node → generator    （商品结果）
  order_node → generator      （订单结果）
  generator → answerability   （校验知识回复依据）
  answerability → human_node  （生成回答无资料依据，转人工）
  human_node → END            （人工介入）
  generator → END             （生成完成）
"""

import logging
import json
import re
import uuid
from typing import Annotated, Literal, TypedDict
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from core.config_utils.config import Config
from core.data.product_db import get_product_db
from core.data.order_db import get_order_db
from core.rag.message_builder import get_message_builder
from core.rag.prompt import build_answerability_hint, build_product_reply_prompt
from core.agents.model_router import get_model_router

logger = logging.getLogger(__name__)


# ============================================================
# Agent State
# ============================================================

class AgentState(TypedDict):
    """Agent 状态"""
    query: str                          # 用户原始查询
    history: list                       # 对话历史 [{"role": "user/assistant", "content": "..."}]
    intent: str                         # 意图分类结果: "product" / "order" / "knowledge" / "human"
    routing_source: str                 # 路由来源: "rule" / "context" / "no_match" / "ambiguous"
    product_results: list               # 商品查询结果
    order_results: list                 # 订单查询结果
    rag_results: list                   # RAG 检索结果 [(Chunk, score, score_type)]
    retrieval_score: float              # 检索匹配分（rerank 或 rrf 原始分）
    score_type: str                     # 检索分来源: "rerank" / "rrf"
    answerability: str                  # 回答依据: "supported" / "partial" / "none"
    answerability_reason: str           # 回答依据判定说明
    need_phone: bool                    # 是否需要用户提供手机号
    response: str                       # 最终回复
    need_human: bool                    # 是否需要人工介入
    trace: dict                         # 调试信息


# ============================================================
# Node 实现
# ============================================================

# 规则快速路由关键词
RULE_ROUTER = {
    "product": [
        "多少钱", "价格", "库存", "有货", "缺货",
        "规格", "尺寸", "颜色",
        "买", "买什么", "推荐", "找", "搜索", "查",
        "有什么", "有没有", "看看", "想了解",
        "哪款好", "哪个好", "推荐一下", "推荐一款",
    ],
    "order": [
        "订单", "快递", "物流", "发货了没", "到哪了",
        "配送", "派送", "快递单号", "包裹", "追踪",
        "查订单", "订单状态", "我的订单", "多久到", "发货",
        "查看订单", "订单查询", "买了什么", "我的购买记录",
        "看订单", "订单在哪", "订单进度",
    ],
    "human": [
        "转人工", "人工客服", "联系客服", "找人工", "人工",
    ]
}

_PHONE_RE = re.compile(r'(?<!\d)(1[3-9]\d{9})(?!\d)')
_ALNUM_IDENTIFIER_RE = re.compile(
    r'(?<![A-Za-z0-9])([A-Za-z]{1,8}\d{8,20})(?![A-Za-z0-9])'
)
_NUMERIC_IDENTIFIER_RE = re.compile(r'(?<!\d)(\d{10,20})(?!\d)')


def _best_match(query: str, keywords: list[str]) -> str:
    """返回查询中命中的最长关键词，作为该意图的特异性依据。"""
    matches = [keyword for keyword in keywords if keyword in query]
    return max(matches, key=len, default="")


def _extract_order_identifiers(query: str) -> list[str]:
    """从自然语言中提取订单号或快递单号。"""
    without_phone = _PHONE_RE.sub(" ", query or "")
    matches = _ALNUM_IDENTIFIER_RE.findall(without_phone)
    matches.extend(_NUMERIC_IDENTIFIER_RE.findall(without_phone))
    return list(dict.fromkeys(matches))


def _mask_phone(phone: str) -> str:
    phone = phone or ""
    if len(phone) < 7:
        return "***"
    return f"{phone[:3]}****{phone[-4:]}"


def _mask_name(name: str) -> str:
    name = name or ""
    if not name:
        return ""
    return name[0] + "*" * max(1, len(name) - 1)


def _mask_address(address: str) -> str:
    address = address or ""
    if len(address) <= 6:
        return "***"
    return address[:6] + "***"


def _sanitize_order_for_customer(order: dict) -> dict:
    """避免把订单中的直接身份信息返回给客户会话。"""
    result = dict(order)
    result["user_phone"] = _mask_phone(result.get("user_phone", ""))
    result["user_name"] = _mask_name(result.get("user_name", ""))
    result["shipping_address"] = _mask_address(result.get("shipping_address", ""))
    result["remark"] = ""
    return result


def _resolve_query(state: AgentState) -> AgentState:
    """指代消解：在路由前改写用户查询"""
    from core.agents.context_engine import get_context_engine
    engine = get_context_engine()
    history = state.get("history", [])
    resolved = engine.resolve_pronouns(state["query"], history)
    if resolved != state["query"]:
        logger.info("指代消解: %s -> %s", state["query"][:30], resolved[:30])
    return {**state, "query": resolved}


def classify_intent(state: AgentState) -> AgentState:
    """
    意图分类节点
    第一层：规则快速路由（human 优先，product / order 同级匹配）
    第二层：上下文感知（如等待提供手机号 → order）
    第三层：RAG 兜底（规则未命中时走知识库检索）
    """
    query = state["query"]
    history = state.get("history", [])

    # 第一层：规则快速路由
    for keyword in RULE_ROUTER["human"]:
        if keyword in query:
            logger.info("规则路由: human -> %s", query[:30])
            return {
                **state,
                "intent": "human",
                "routing_source": "rule"
            }

    product_match = _best_match(query, RULE_ROUTER["product"])
    order_match = _best_match(query, RULE_ROUTER["order"])

    if product_match and not order_match:
        logger.info("规则路由: product -> %s (match=%s)", query[:30], product_match)
        return {
            **state,
            "intent": "product",
            "routing_source": "rule"
        }

    if order_match and not product_match:
        logger.info("规则路由: order -> %s (match=%s)", query[:30], order_match)
        return {
            **state,
            "intent": "order",
            "routing_source": "rule"
        }

    if order_match and product_match:
        if len(order_match) > len(product_match):
            logger.info("规则路由: order -> %s (order_match=%s, product_match=%s)",
                        query[:30], order_match, product_match)
            return {
                **state,
                "intent": "order",
                "routing_source": "rule"
            }

        if len(product_match) > len(order_match):
            logger.info("规则路由: product -> %s (order_match=%s, product_match=%s)",
                        query[:30], order_match, product_match)
            return {
                **state,
                "intent": "product",
                "routing_source": "rule"
            }

        logger.info("规则路由歧义: %s (order_match=%s, product_match=%s)",
                    query[:30], order_match, product_match)
        return {
            **state,
            "intent": "knowledge",
            "routing_source": "ambiguous"
        }

    # 第二层：上下文感知
    # 如果历史显示正在等待手机号，直接路由到 order
    phone_match = _PHONE_RE.search(query)
    if phone_match and any(
        '手机' in m.get('content', '') or '号码' in m.get('content', '')
        for m in history if m.get('role') == 'assistant'
    ):
        logger.info("上下文路由: 手机号回复 -> order")
        return {
            **state,
            "intent": "order",
            "routing_source": "context"
        }

    # 第三层：RAG 兜底
    logger.info("规则未命中，走 RAG 兜底: %s", query[:30])
    return {
        **state,
        "intent": "knowledge",
        "routing_source": "no_match"
    }


def product_node(state: AgentState) -> AgentState:
    """商品查询节点"""
    query = state["query"]
    product_db = get_product_db()
    results = product_db.search_products(query)

    logger.info("商品查询: %s -> %d 结果", query[:30], len(results))

    return {
        **state,
        "product_results": results,
        "trace": {**state.get("trace", {}), "product_query": query, "product_count": len(results)}
    }


def order_node(state: AgentState) -> AgentState:
    """订单查询节点"""
    query = state["query"]
    order_db = get_order_db()

    # 从 query 中提取手机号、订单号和快递单号，避免把整句当成单号查询
    phone_match = _PHONE_RE.search(query)
    phone = phone_match.group(0) if phone_match else None
    identifiers = _extract_order_identifiers(query)

    order = None
    for identifier in identifiers:
        order = order_db.track_order(identifier, exact=True, limit=1)
        if order:
            break

    if order:
        results = [_sanitize_order_for_customer(order)]
        logger.info("订单追踪命中: identifier=%s -> 1 结果", identifiers[0])
    elif phone:
        # 有手机号，按手机号搜索
        results = [
            _sanitize_order_for_customer(order)
            for order in order_db.search_by_phone(phone, exact=True, limit=10)
        ]
        logger.info("订单查询(手机号): %s -> %d 结果", _mask_phone(phone), len(results))
    else:
        # 没有手机号，标记需要引导用户提供
        results = []
        if not identifiers:
            state["need_phone"] = True

    return {
        **state,
        "order_results": results,
        "trace": {
            **state.get("trace", {}),
            "order_count": len(results),
            "phone_provided": bool(phone),
            "identifier_provided": bool(identifiers)
        }
    }


def knowledge_node(state: AgentState) -> AgentState:
    """RAG 知识检索节点"""
    from core.rag.rag_chain import get_rag_chain
    query = state["query"]
    chain = get_rag_chain()
    results = chain.search(query)

    first_result = results[0] if results else None
    retrieval_score = first_result[1] if first_result else 0.0
    score_type = first_result[2] if first_result else ""
    retrieved_items = [
        {
            "id": f"{chunk.source_file}:{chunk.chunk_index}",
            "question": chunk.heading or chunk.content[:50],
            "source_file": chunk.source_file,
            "section": chunk.heading,
            "score": score,
            "score_type": item_score_type
        }
        for chunk, score, item_score_type in results[:5]
    ]
    context_text = "\n\n".join(chunk.content for chunk, _, _ in results[:3])
    logger.info(
        "RAG 检索: %s -> %d 结果, score=%.4f (%s)",
        query[:30], len(results), retrieval_score, score_type or "none"
    )

    return {
        **state,
        "rag_results": results,
        "retrieval_score": retrieval_score,
        "score_type": score_type,
        "trace": {
            **state.get("trace", {}),
            "rag_results_count": len(results),
            "retrieval_score": retrieval_score,
            "score_type": score_type,
            "rewritten_query": chain.rewrite_query(query),
            "retrieved_items": retrieved_items,
            "context_text": context_text
        }
    }


def human_node(state: AgentState) -> AgentState:
    """转人工节点 — 更新会话状态为待人工处理"""
    query = state["query"]
    logger.info("转人工: %s", query[:30])

    # 将当前会话标记为待人工处理
    try:
        from core.agents.conversation import get_conversation_manager
        mgr = get_conversation_manager()
        if mgr.current_conversation:
            mgr.current_conversation.transfer_to_human()
            mgr.db.update_conversation_status(
                mgr.current_conversation.id,
                status=mgr.current_conversation.status,
                human_agent_id=mgr.current_conversation.human_agent_id
            )
    except Exception as e:
        logger.warning("转人工状态更新失败: %s", e)

    return {
        **state,
        "response": "正在为您转接人工客服，请稍候...",
        "need_human": True,
        "trace": {**state.get("trace", {}), "human_requested": True}
    }


def generator_node(state: AgentState) -> AgentState:
    """LLM 生成回复节点"""
    intent = state["intent"]
    query = state["query"]
    history = state.get("history", [])

    if intent == "product":
        return _generate_product_response(state, history)
    elif intent == "order":
        return _generate_order_response(state, history)
    elif intent == "knowledge":
        return _generate_knowledge_response(state, history)
    else:
        return {
            **state,
            "response": "抱歉，我无法处理这个请求。",
            "trace": {**state.get("trace", {}), "error": "unknown intent"}
        }


def _generate_product_response(state: AgentState, history: list) -> AgentState:
    """生成商品回复"""
    products = state.get("product_results", [])
    query = state["query"]

    if not products:
        return {
            **state,
            "response": "抱歉，没有找到相关资料，可以尝试转人工服务。",
            "trace": {**state.get("trace", {}), "product_found": False}
        }

    # 使用 message_builder 构建消息
    builder = get_message_builder()
    messages = builder.build_product_messages(query, products)

    # 调用 LLM
    router = get_model_router()
    if router.is_configured():
        try:
            from langchain_core.messages import SystemMessage, HumanMessage
            # 转换为 LangChain 消息格式
            lc_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    if msg.get("role") == "system":
                        lc_messages.append(SystemMessage(content=msg["content"]))
                    elif msg.get("role") == "user":
                        lc_messages.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        lc_messages.append(AIMessage(content=msg["content"]))
                else:
                    lc_messages.append(msg)

            response = router.chat(lc_messages)
            return {
                **state,
                "response": response,
                "trace": {**state.get("trace", {}), "product_count": len(products), "generated": True}
            }
        except Exception as e:
            logger.error("商品回复生成失败: %s", e)
            return {
                **state,
                "response": f"找到 {len(products)} 个相关商品，请稍后查看。",
                "trace": {**state.get("trace", {}), "error": str(e)}
            }
    else:
        # 未配置 LLM，返回结构化数据
        product_info = []
        for p in products:
            product_info.append({
                "id": p.get("id", ""),
                "name": p.get("name", ""),
                "price": float(p.get("price", 0) or 0),
                "stock": p.get("stock", 0),
                "category": p.get("category", "")
            })
        return {
            **state,
            "response": f"找到 {len(products)} 个商品:\n" + "\n".join(
                f"- {p['name']}: ¥{p['price']} (库存: {p['stock']})"
                for p in product_info[:3]
            ),
            "trace": {**state.get("trace", {}), "product_count": len(products), "no_llm": True}
        }


def _generate_order_response(state: AgentState, history: list) -> AgentState:
    """生成订单回复"""
    orders = state.get("order_results", [])
    query = state["query"]

    # 未提供手机号，引导用户提供
    trace = state.get("trace", {})
    identifier_provided = trace.get("identifier_provided", False)
    if state.get("need_phone") or (
        not orders and not trace.get("phone_provided") and not identifier_provided
    ):
        return {
            **state,
            "response": "查询订单信息需要提供您的手机号码，请输入您下单时使用的手机号。",
            "trace": {**state.get("trace", {}), "order_found": False, "need_phone": True}
        }

    if not orders:
        return {
            **state,
            "response": "抱歉，未找到相关订单信息，请确认订单号或快递单号是否正确。",
            "trace": {**state.get("trace", {}), "order_found": False}
        }

    # 构建订单信息回复
    order_info = []
    for o in orders[:3]:  # 最多显示3条
        status_emoji = {
            "待发货": "⏳", "已发货": "🚚", "运输中": "📦",
            "已签收": "✅", "已退货": "↩️", "已取消": "❌"
        }.get(o.get("status", ""), "📋")

        info = f"{status_emoji} 订单 [{o.get('order_no', '')}]"
        info += f" | 商品: {o.get('product_name', o.get('product_id', ''))}"
        info += f" | 金额: ¥{o.get('total_amount', 0):.2f}"
        info += f" | 状态: {o.get('status', '')}"

        tracking = o.get('tracking_no', '')
        if tracking:
            info += f" | 快递: {o.get('logistics_company', '')} {tracking}"

        order_info.append(info)

    response = "查询到的订单信息：\n\n" + "\n\n".join(order_info)

    if len(orders) > 3:
        response += f"\n\n... 还有 {len(orders) - 3} 条订单，请提供更精确的订单号查询。"

    return {
        **state,
        "response": response,
        "trace": {**state.get("trace", {}), "order_count": len(orders)}
    }


def _generate_knowledge_response(state: AgentState, history: list) -> AgentState:
    """生成知识类回复（RAG 检索结果）"""
    from core.rag.message_builder import get_message_builder
    from core.agents.model_router import get_model_router

    rag_results = state.get("rag_results", [])
    query = state["query"]

    if not rag_results:
        return {
            **state,
            "response": "抱歉，知识库中未找到相关信息，建议联系人工客服获取帮助。",
            "trace": {**state.get("trace", {}), "knowledge_found": False}
        }

    # 构建 context_chunks
    context_chunks = []
    for chunk, score, score_type in rag_results[:3]:
        context_chunks.append({
            "content": chunk.content,
            "source_file": chunk.source_file,
            "section": chunk.heading,
            "category": chunk.category
        })

    # 使用 MessageBuilder 构建消息
    builder = get_message_builder()
    messages = builder.build_rag_messages(
        question=query,
        context_chunks=context_chunks,
        history=history
    )

    # 调用 LLM 生成回复
    router = get_model_router()
    if router.is_configured():
        try:
            from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
            lc_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role == "system":
                        lc_messages.append(SystemMessage(content=content))
                    elif role == "user":
                        lc_messages.append(HumanMessage(content=content))
                    elif role == "assistant":
                        lc_messages.append(AIMessage(content=content))
                else:
                    lc_messages.append(msg)

            response = router.chat(lc_messages)
            return {
                **state,
                "response": response,
                "trace": {
                    **state.get("trace", {}),
                    "retrieval_score": state.get("retrieval_score", 0.0),
                    "score_type": state.get("score_type", ""),
                    "results_count": len(rag_results)
                }
            }
        except Exception as e:
            logger.error("知识回复生成失败: %s", e)
            return {
                **state,
                "response": "服务暂时不可用，请稍后重试。",
                "trace": {**state.get("trace", {}), "error": str(e)}
            }
    else:
        # 未配置 LLM，返回检索结果摘要
        chunks_text = []
        for chunk, score, score_type in rag_results[:3]:
            chunks_text.append(f"[{chunk.source_file} > {chunk.heading}] {chunk.content[:100]}...")
        return {
            **state,
            "response": "检索到以下相关内容：\n" + "\n\n".join(chunks_text),
            "trace": {
                **state.get("trace", {}),
                "retrieval_score": state.get("retrieval_score", 0.0),
                "score_type": state.get("score_type", ""),
                "no_llm": True
            }
        }


def answerability_node(state: AgentState) -> AgentState:
    """校验知识回复是否由检索资料支撑，并决定是否转人工。"""
    if state.get("intent") != "knowledge":
        return state

    rag_results = state.get("rag_results", [])
    response = state.get("response", "")
    if not rag_results or not response:
        return {
            **state,
            "answerability": "none",
            "answerability_reason": "缺少回答或检索资料，无法确认回答依据",
            "trace": {
                **state.get("trace", {}),
                "answerability": "none",
                "answerability_reason": "缺少回答或检索资料，无法确认回答依据"
            }
        }

    router = get_model_router()
    if not router.is_configured():
        reason = "LLM 未配置，无法校验回答依据，仅能展示检索片段"
        return {
            **state,
            "response": response + build_answerability_hint("partial"),
            "answerability": "partial",
            "answerability_reason": reason,
            "trace": {
                **state.get("trace", {}),
                "answerability": "partial",
                "answerability_reason": reason
            }
        }

    sources = []
    for chunk, score, score_type in rag_results[:3]:
        sources.append(
            f"[{chunk.source_file} > {chunk.heading}]\n"
            f"检索分: {score:.4f} ({score_type})\n{chunk.content}"
        )

    system_prompt = """你是客服回答依据校验器。只判断客服回答能否由提供的知识库资料支撑，不要补充新知识，也不要改写客服回答。

必须只输出 JSON，不要输出 Markdown 代码块或其他文字：
{
  "supported": true 或 false,
  "coverage": "full" 或 "partial" 或 "none",
  "reason": "简短说明判断依据",
  "should_transfer": true 或 false
}

判定规则：
- full：回答中的关键结论都能由资料直接支撑
- partial：回答有部分依据，但缺少必要信息或包含无法验证的内容
- none：资料无法支撑回答，或回答与资料冲突
- coverage 为 none，或 supported 为 false 且无法部分支撑时，should_transfer 必须为 true"""

    sources_text = "\n\n".join(sources)
    user_prompt = (
        f"用户问题：\n{state['query']}\n\n"
        f"知识库资料：\n{sources_text}\n\n"
        f"客服回答：\n{response}"
    )

    try:
        raw_result = router.chat([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        try:
            data = _parse_answerability_json(raw_result)
        except (ValueError, json.JSONDecodeError):
            retry_prompt = (
                f"{user_prompt}\n\n"
                "上次输出无法解析。请只返回一个合法 JSON 对象，不要使用代码块。"
            )
            raw_result = router.chat([
                SystemMessage(content=system_prompt),
                HumanMessage(content=retry_prompt)
            ])
            data = _parse_answerability_json(raw_result)

        supported = bool(data.get("supported", False))
        coverage = str(data.get("coverage", "none")).strip().lower()
        reason = str(data.get("reason", "")).strip() or "未提供判定说明"

        if supported and coverage == "full":
            answerability = "supported"
        elif coverage == "partial":
            answerability = "partial"
        else:
            answerability = "none"
    except Exception as e:
        logger.warning("回答依据校验失败: %s", e)
        answerability = "partial"
        reason = f"回答依据校验失败，按部分依据展示: {e}"

    updates = {
        **state,
        "answerability": answerability,
        "answerability_reason": reason,
        "trace": {
            **state.get("trace", {}),
            "answerability": answerability,
            "answerability_reason": reason
        }
    }

    if answerability == "partial":
        updates["response"] = response + build_answerability_hint("partial")

    return updates


def _parse_answerability_json(raw_result: str) -> dict:
    """解析模型返回的 JSON，兼容 Markdown 代码围栏。"""
    text = (raw_result or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        return json.loads(fenced.group(1))

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("LLM 未返回 JSON")
    return json.loads(match.group(0))


def route_after_classifier(state: AgentState) -> str:
    """分类器后的路由"""
    intent = state.get("intent", "knowledge")
    if intent == "product":
        return "product"
    elif intent == "order":
        return "order"
    elif intent == "human":
        return "human"
    else:
        return "knowledge"


def route_after_knowledge(state: AgentState) -> str:
    """知识节点后的路由：有候选资料才生成，否则转人工。"""
    has_results = bool(state.get("rag_results"))
    logger.debug(
        "route_after_knowledge: results=%s, score=%.4f (%s)",
        len(state.get("rag_results", [])),
        state.get("retrieval_score", 0.0),
        state.get("score_type", "") or "none"
    )
    return "generator" if has_results else "human"


def route_after_product(state: AgentState) -> str:
    """商品节点后的路由（始终生成回复）"""
    return "generator"


def route_after_generator(state: AgentState) -> str:
    """知识回复生成后校验依据，其他意图直接结束。"""
    return "answerability" if state.get("intent") == "knowledge" else "end"


def route_after_answerability(state: AgentState) -> str:
    """回答没有资料依据时转人工，否则结束。"""
    return "human" if state.get("answerability") == "none" else "end"


# ============================================================
# 构建 Graph
# ============================================================

def build_customer_agent(checkpointer=None):
    """构建 CustomerAgent 图

    Args:
        checkpointer: 状态持久化器（可选，用于多轮对话）
    """
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("resolve", _resolve_query)
    graph.add_edge("resolve", "classify")
    graph.add_node("classify", classify_intent)
    graph.add_node("product", product_node)
    graph.add_node("order", order_node)
    graph.add_node("knowledge", knowledge_node)
    graph.add_node("answerability", answerability_node)
    graph.add_node("human", human_node)
    graph.add_node("generator", generator_node)

    # 设置入口
    graph.set_entry_point("resolve")

    # 添加边
    graph.add_edge("human", END)

    # 条件边：classify -> ?
    graph.add_conditional_edges(
        "classify",
        route_after_classifier,
        {
            "product": "product",
            "order": "order",
            "knowledge": "knowledge",
            "human": "human",
        }
    )

    # product -> generator
    graph.add_conditional_edges(
        "product",
        route_after_product,
        {"generator": "generator"}
    )

    # order -> generator
    graph.add_edge("order", "generator")

    # knowledge -> generator 或 human（是否存在候选资料）
    graph.add_conditional_edges(
        "knowledge",
        route_after_knowledge,
        {
            "generator": "generator",
            "human": "human",
        }
    )

    # generator -> answerability（知识）或 END（商品/订单）
    graph.add_conditional_edges(
        "generator",
        route_after_generator,
        {
            "answerability": "answerability",
            "end": END,
        }
    )

    # answerability -> human 或 END
    graph.add_conditional_edges(
        "answerability",
        route_after_answerability,
        {
            "human": "human",
            "end": END,
        }
    )

    # 编译图（可选传入 checkpointer 用于持久化）
    compile_kwargs = {}
    if checkpointer:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


# 全局单例
_customer_agent = None
_checkpointer = MemorySaver()


def get_customer_agent():
    """获取 CustomerAgent 单例"""
    global _customer_agent
    if _customer_agent is None:
        _customer_agent = build_customer_agent(checkpointer=_checkpointer)
    return _customer_agent


def run_agent(query: str, history: list = None, thread_id: str = None) -> dict:
    """
    运行 Agent（同步版，返回完整回复）
    """
    agent = get_customer_agent()

    initial_state: AgentState = {
        "query": query,
        "history": history or [],
        "intent": "",
        "routing_source": "",
        "product_results": [],
        "order_results": [],
        "rag_results": [],
        "retrieval_score": 0.0,
        "score_type": "",
        "answerability": "",
        "answerability_reason": "",
        "need_phone": False,
        "response": "",
        "need_human": False,
        "trace": {"query": query, "history_len": len(history or [])}
    }

    # 如果没有 thread_id，生成一个临时的
    if not thread_id:
        thread_id = str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = agent.invoke(initial_state, config=config)
        return {
            "response": result.get("response", ""),
            "intent": result.get("intent", ""),
            "routing_source": result.get("routing_source", ""),
            "need_human": result.get("need_human", False),
            "retrieval_score": result.get("retrieval_score", 0.0),
            "score_type": result.get("score_type", ""),
            "answerability": result.get("answerability", ""),
            "answerability_reason": result.get("answerability_reason", ""),
            "trace": result.get("trace", {})
        }
    except Exception as e:
        logger.error("Agent 执行失败: %s", e)
        return {
            "response": "抱歉，服务暂时不可用，请稍后重试。",
            "intent": "error",
            "routing_source": "",
            "need_human": False,
            "retrieval_score": 0.0,
            "score_type": "",
            "answerability": "",
            "answerability_reason": "",
            "trace": {"error": str(e)}
        }

