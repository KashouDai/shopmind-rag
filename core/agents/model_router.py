# -*- coding: utf-8 -*-
"""
模型路由模块
基于 LangChain BaseChatModel 统一各厂商接口，从 Config 读取配置驱动
"""

import logging
from typing import Optional, List, Dict, Any

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatZhipuAI

from core.config_utils.config import Config
from core.infra.llm_providers import get_provider, LLMProviderError

logger = logging.getLogger(__name__)


class ModelRouter:
    """模型路由"""

    # 提供商映射
    PROVIDER_MAP = {
        "openai": "openai",
        "siliconflow": "openai",  # SiliconFlow 使用 OpenAI 兼容接口
        "qwen": "openai",  # 通义千问使用 OpenAI 兼容接口
        "zhipu": "zhipu",
        "deepseek": "openai",  # DeepSeek 使用 OpenAI 兼容接口
    }

    def __init__(self):
        self._config = Config()
        self._model = None
        self._provider_name = None
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        provider_name = self._config.get("llm_provider", "siliconflow")
        api_key = self._config.get("api_key", "")
        api_url = self._config.get("api_base_url", "")
        model_name = self._config.get("model_name", "")
        max_tokens = self._config.get("max_tokens", 2048)
        temperature = self._config.get("temperature", 0.7)
        timeout = self._config.get("api_timeout", 30)

        if not api_key:
            logger.warning("未配置 API 密钥，模型路由不可用")
            return

        try:
            langchain_provider = self.PROVIDER_MAP.get(provider_name, "openai")

            if langchain_provider == "zhipu":
                # 智谱 AI
                self._model = ChatZhipuAI(
                    api_key=api_key,
                    model=model_name or "glm-4-flash",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )
            else:
                # OpenAI 兼容接口（OpenAI / SiliconFlow / Qwen / DeepSeek）
                self._model = ChatOpenAI(
                    api_key=api_key,
                    base_url=api_url or self._get_default_base_url(provider_name),
                    model=model_name or self._get_default_model(provider_name),
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout
                )

            self._provider_name = provider_name
            logger.info("模型路由初始化成功: %s -> %s", provider_name, model_name or "default")

        except Exception as e:
            logger.error("模型路由初始化失败: %s", e)
            raise

    def _get_default_base_url(self, provider: str) -> str:
        """获取默认 API 地址"""
        urls = {
            "openai": "https://api.openai.com/v1",
            "siliconflow": "https://api.siliconflow.cn/v1",
            "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "deepseek": "https://api.deepseek.com/v1",
        }
        return urls.get(provider, "https://api.openai.com/v1")

    def _get_default_model(self, provider: str) -> str:
        """获取默认模型名称"""
        models = {
            "openai": "gpt-3.5-turbo",
            "siliconflow": "Qwen/Qwen3-8B",
            "qwen": "qwen-turbo",
            "zhipu": "glm-4-flash",
            "deepseek": "deepseek-chat",
        }
        return models.get(provider, "gpt-3.5-turbo")

    def chat(self, messages: List[BaseMessage], **kwargs) -> str:
        """调用模型生成回复"""
        if not self._model:
            logger.error("模型未初始化")
            return "服务暂时不可用，请稍后重试。"

        try:
            response = self._model.invoke(messages, **kwargs)
            return response.content
        except Exception as e:
            logger.error("模型调用失败: %s", e)
            return "抱歉，服务暂时不可用，请稍后重试。"

    def is_configured(self) -> bool:
        """检查是否已配置"""
        return self._model is not None

    def reload(self):
        """重新加载模型（配置变更后调用）"""
        self._init_model()

    @property
    def provider_name(self) -> Optional[str]:
        """获取当前提供商名称"""
        return self._provider_name

    @property
    def model(self) -> Optional[str]:
        """获取当前模型名称"""
        if self._model:
            return getattr(self._model, 'model_name', None) or getattr(self._model, 'model', None)
        return None


# 全局单例
_model_router: Optional[ModelRouter] = None


def get_model_router() -> ModelRouter:
    """获取模型路由单例"""
    global _model_router
    if _model_router is None:
        _model_router = ModelRouter()
    return _model_router
