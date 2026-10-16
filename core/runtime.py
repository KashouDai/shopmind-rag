# -*- coding: utf-8 -*-
"""应用运行时资源清理。"""

import importlib
import logging

logger = logging.getLogger(__name__)


def close_runtime_resources(config=None) -> None:
    """关闭已初始化的全局连接，不触发新的连接创建。"""
    if config is not None:
        config.disable_hot_reload()

    singletons = (
        ("core.rag.vector_store", "_vector_store"),
        ("core.data.redis_cache", "_cache"),
        ("core.data.product_db", "_product_db"),
        ("core.data.order_db", "_order_db"),
        ("core.data.conversation_db", "_db"),
    )
    for module_name, attribute in singletons:
        try:
            module = importlib.import_module(module_name)
            instance = getattr(module, attribute, None)
            if instance is not None:
                instance.close()
        except Exception as e:
            logger.warning("关闭运行时资源失败: %s.%s: %s", module_name, attribute, e)
