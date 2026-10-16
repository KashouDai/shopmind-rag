# -*- coding: utf-8 -*-
"""
商品数据访问层 — PostgreSQL + SQLAlchemy ORM
替代 ProductStore 的 JSON 读写
"""

import os
import json
import logging
import re
import uuid
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Integer, DECIMAL, Text, JSON, DateTime, text, or_
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from core.config_utils.config import Config

logger = logging.getLogger(__name__)

Base = declarative_base()


def _tokenize_product_query(query: str) -> List[str]:
    """生成中英文商品检索词，避免把中文拆成无意义的单字。"""
    query = (query or "").strip().lower()
    if not query:
        return []

    tokens = [query]
    tokens.extend(re.findall(r'[a-zA-Z0-9]+', query))
    for segment in re.findall(r'[一-鿿]{2,}', query):
        tokens.append(segment)
        tokens.extend(segment[index:index + 2] for index in range(len(segment) - 1))
    return list(dict.fromkeys(tokens))


def _product_match_score(product: Dict[str, Any], query: str, tokens: List[str]) -> float:
    """按字段权重计算商品匹配分。"""
    query = (query or "").strip().lower()
    name = str(product.get("name", "")).lower()
    category = str(product.get("category", "")).lower()
    description = str(product.get("description", "")).lower()
    keywords = " ".join(
        str(item).lower() for item in (product.get("keywords") or [])
    )

    score = 0.0
    if name == query:
        score += 100.0
    elif query and query in name:
        score += 50.0

    for token in tokens:
        if token in name:
            score += 12.0
        if token in keywords:
            score += 8.0
        if token in category:
            score += 5.0
        if token in description:
            score += 2.0
    return score


class ProductModel(Base):
    """商品表 ORM 模型"""
    __tablename__ = "product"

    id = Column(String(20), primary_key=True, comment="商品ID，如 P001")
    name = Column(String(200), nullable=False, comment="商品名称")
    price = Column(DECIMAL(10, 2), nullable=False, comment="价格")
    category = Column(String(100), nullable=False, comment="分类")
    description = Column(Text, comment="商品描述")
    specifications = Column(JSON, comment="规格参数")
    stock = Column(Integer, default=0, comment="库存数量")
    keywords = Column(String(500), comment="关键词（逗号分隔）")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")


class ProductDB:
    """商品数据访问层"""

    _instance = None
    _engine = None
    _session_factory = None

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
        self._engine = self._create_engine()
        self._session_factory = sessionmaker(bind=self._engine)
        self._create_tables()

    def _get_db_url(self) -> str:
        config = self._config.get_all()
        host = config.get("postgres_host", "localhost")
        port = config.get("postgres_port", 5432)
        database = config.get("postgres_database", "ecom_db")
        user = config.get("postgres_user", "postgres")
        password = config.get("postgres_password", "postgres")
        # 使用 pg8000 驱动，避免 psycopg2 在 Windows 上的 Unicode bug
        return f"postgresql+pg8000://{user}:{password}@{host}:{port}/{database}"

    def _create_engine(self):
        db_url = self._get_db_url()
        try:
            engine = create_engine(db_url, echo=False)
            # 测试连接
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("PostgreSQL 连接成功: %s:%s/%s",
                        self._config.get("postgres_host", "localhost"),
                        self._config.get("postgres_port", 5432),
                        self._config.get("postgres_database", "ecom_db"))
            return engine
        except Exception as e:
            logger.error("PostgreSQL 连接失败: %s", e)
            raise

    def _create_tables(self):
        """建表（幂等）"""
        Base.metadata.create_all(self._engine)
        logger.info("数据库表检查完成")

    def _session(self) -> Session:
        return self._session_factory()

    @staticmethod
    def _product_to_dict(product) -> Dict[str, Any]:
        """将 ORM 对象转为字典，处理 datetime/Decimal 等不可序列化类型"""
        d = {}
        for k in product.__dict__:
            if k.startswith('_'):
                continue
            v = getattr(product, k)
            if hasattr(v, 'isoformat'):
                v = v.isoformat()
            elif hasattr(v, '__float__'):
                v = float(v)
            d[k] = v
        # 解析关键词
        if 'keywords' in d and isinstance(d['keywords'], str):
            d['keywords'] = ProductDB._parse_keywords(d['keywords'])
        return d

    @staticmethod
    def _parse_keywords(kw_str: str) -> List[str]:
        """解析关键词字符串，兼容多种格式（逗号分隔、JSON数组、花括号中文分隔）"""
        import re
        if not kw_str:
            return []
        # 尝试 JSON 解析
        try:
            parsed = json.loads(kw_str)
            if isinstance(parsed, list):
                return [str(k).strip() for k in parsed if k.strip()]
            if isinstance(parsed, str):
                kw_str = parsed
        except (json.JSONDecodeError, TypeError):
            pass
        # 去掉首尾花括号
        inner = kw_str.strip("{}")
        # 按逗号、中文顿号、空格分割
        parts = re.split(r'[,\，、\s]+', inner)
        return [p.strip() for p in parts if p.strip()]

    def add_product(self, name: str, price: float, category: str, description: str,
                    specifications: Dict[str, str] = None, stock: int = 0,
                    keywords: List[str] = None) -> Dict[str, Any]:
        """添加商品"""
        product_id = self._generate_id()
        now = datetime.utcnow()

        product = ProductModel(
            id=product_id,
            name=name,
            price=price,
            category=category,
            description=description,
            specifications=specifications or {},
            stock=stock,
            keywords=", ".join(keywords) if keywords and isinstance(keywords, (list, tuple)) else "",
            created_at=now,
            updated_at=now
        )

        session = self._session()
        try:
            session.add(product)
            session.commit()
            logger.info("商品添加成功: %s - %s", product_id, name)
            # 返回字典格式，与 ProductItem 兼容
            return {
                "id": product.id,
                "name": product.name,
                "price": float(product.price),
                "category": product.category,
                "description": product.description,
                "specifications": product.specifications or {},
                "stock": product.stock,
                "keywords": self._parse_keywords(product.keywords) if product.keywords else [],
                "created_at": product.created_at.isoformat() if product.created_at else "",
                "updated_at": product.updated_at.isoformat() if product.updated_at else ""
            }
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("商品添加失败: %s", e)
            raise
        finally:
            session.close()

    def delete_product(self, product_id: str) -> bool:
        """删除商品"""
        session = self._session()
        try:
            product = session.query(ProductModel).filter_by(id=product_id).first()
            if not product:
                return False
            session.delete(product)
            session.commit()
            logger.info("商品删除成功: %s", product_id)
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("商品删除失败: %s", e)
            raise
        finally:
            session.close()

    def update_product(self, product_id: str, **kwargs) -> bool:
        """更新商品"""
        session = self._session()
        try:
            product = session.query(ProductModel).filter_by(id=product_id).first()
            if not product:
                return False

            for key, value in kwargs.items():
                if hasattr(product, key) and key not in ("id", "created_at"):
                    setattr(product, key, value)
            product.updated_at = datetime.utcnow()

            session.commit()
            logger.info("商品更新成功: %s", product_id)
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("商品更新失败: %s", e)
            raise
        finally:
            session.close()

    def get_product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取商品"""
        session = self._session()
        try:
            product = session.query(ProductModel).filter_by(id=product_id).first()
            if product:
                return self._product_to_dict(product)
            return None
        finally:
            session.close()

    def get_all_products(self) -> List[Dict[str, Any]]:
        """获取所有商品"""
        session = self._session()
        try:
            products = session.query(ProductModel).order_by(ProductModel.created_at.desc()).all()
            return [self._product_to_dict(p) for p in products]
        finally:
            session.close()

    def search_products(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """搜索商品 — 支持多关键词模糊匹配（名称/描述/关键词/分类）"""
        session = self._session()
        try:
            tokens = _tokenize_product_query(query)
            if not tokens:
                return []

            like_conditions = []
            for token in tokens:
                like_query = f"%{token}%"
                like_conditions.append(
                    (ProductModel.name.like(like_query)) |
                    (ProductModel.description.like(like_query)) |
                    (ProductModel.keywords.like(like_query)) |
                    (ProductModel.category.like(like_query))
                )

            products = session.query(ProductModel).filter(
                or_(*like_conditions)
            ).limit(max(limit * 5, 50)).all()
            results = [self._product_to_dict(p) for p in products]
            results.sort(
                key=lambda product: _product_match_score(product, query, tokens),
                reverse=True
            )
            return results[:limit]
        finally:
            session.close()

    def _generate_id(self) -> str:
        """生成商品 ID"""
        return f"P{uuid.uuid4().hex[:12].upper()}"

    def close(self):
        """关闭引擎"""
        if self._engine:
            self._engine.dispose()


# 全局单例
_product_db: Optional[ProductDB] = None


def get_product_db() -> ProductDB:
    """获取商品数据访问层单例"""
    global _product_db
    if _product_db is None:
        _product_db = ProductDB()
    return _product_db
