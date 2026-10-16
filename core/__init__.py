# -*- coding: utf-8 -*-
"""核心业务逻辑层"""

# infra
from core.infra.logger import LogManager, setup_logging, get_logger
from core.infra.performance import PerformanceMonitor
from core.infra.rate_limiter import RateLimiter
from core.infra.file_lock import FileLock
from core.infra.llm_providers import (
    BaseLLMProvider, LLMResponse, LLMProviderError,
    get_provider, get_all_providers, PROVIDER_REGISTRY
)

# config_utils
from core.config_utils.config import Config
from core.config_utils.permissions import PermissionManager, get_permission_manager, require_permission
from core.config_utils.validators import Validators, ProductValidator, KnowledgeValidator

# data
from core.data.product_db import ProductDB, get_product_db
from core.data.order_db import OrderDB, get_order_db
from core.data.conversation_db import ConversationDB, get_conversation_db
from core.data.redis_cache import RedisCache, get_redis_cache
from core.data.backup import BackupManager

# rag
from core.rag.vector_store import VectorStore, get_vector_store
from core.rag.embedding import EmbeddingClient
from core.rag.document_parser import DocumentParser, parse_markdown, Chunk
from core.rag.doc_loader import DocLoader, get_doc_loader
from core.rag.rag_chain import RAGChain, get_rag_chain
from core.rag.message_builder import MessageBuilder, get_message_builder
from core.rag.prompt import (
    BASE_ROLE, RAG_RULES, AGENT_ROUTING_PROMPT, PRODUCT_REPLY_RULES,
    build_rag_system_prompt, build_product_reply_prompt,
    build_agent_routing_prompt, build_answerability_hint
)

# agents
from core.agents.customer_agent import get_customer_agent, AgentState
from core.agents.context_engine import ContextEngine, get_context_engine
from core.agents.model_router import ModelRouter, get_model_router
from core.agents.conversation import ConversationManager, Conversation, Message

# shared
from core.shared_data import KnowledgeStore, ProductStore, KnowledgeItem, ProductItem

# standalone
from core.statistics import StatisticsManager, get_statistics_manager
