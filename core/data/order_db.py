# -*- coding: utf-8 -*-
"""
订单数据访问层 — PostgreSQL + SQLAlchemy ORM
"""

import os
import uuid
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

from sqlalchemy import create_engine, Column, String, Integer, DECIMAL, Text, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from core.config_utils.config import Config

logger = logging.getLogger(__name__)

Base = declarative_base()


class OrderModel(Base):
    """订单表 ORM 模型"""
    __tablename__ = "order"

    id = Column(String(30), primary_key=True, comment="订单ID，如 O202609140001")
    order_no = Column(String(50), unique=True, nullable=False, comment="订单号")
    user_phone = Column(String(20), nullable=False, comment="用户手机号")
    user_name = Column(String(100), comment="用户姓名")
    product_id = Column(String(20), comment="商品ID")
    product_name = Column(String(200), comment="商品名称")
    quantity = Column(Integer, default=1, comment="数量")
    unit_price = Column(DECIMAL(10, 2), comment="单价")
    total_amount = Column(DECIMAL(10, 2), comment="总金额")
    status = Column(String(20), default="待发货", comment="订单状态")
    logistics_company = Column(String(100), comment="快递公司")
    tracking_no = Column(String(100), comment="快递单号")
    shipping_address = Column(Text, comment="收货地址")
    remark = Column(Text, comment="备注")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")


class OrderDB:
    """订单数据访问层"""

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
        return f"postgresql+pg8000://{user}:{password}@{host}:{port}/{database}"

    def _create_engine(self):
        db_url = self._get_db_url()
        try:
            engine = create_engine(db_url, echo=False)
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
        logger.info("订单表检查完成")

    def _session(self) -> Session:
        return self._session_factory()

    def add_order(self, order_no: str, user_phone: str, user_name: str,
                  product_id: str, product_name: str, quantity: int,
                  unit_price: float, total_amount: float,
                  shipping_address: str = "", remark: str = "") -> Dict[str, Any]:
        """添加订单"""
        order_id = self._generate_id()
        now = datetime.utcnow()

        order = OrderModel(
            id=order_id,
            order_no=order_no,
            user_phone=user_phone,
            user_name=user_name,
            product_id=product_id,
            product_name=product_name,
            quantity=quantity,
            unit_price=unit_price,
            total_amount=total_amount,
            status="待发货",
            shipping_address=shipping_address,
            remark=remark,
            created_at=now,
            updated_at=now
        )

        session = self._session()
        try:
            session.add(order)
            session.commit()
            logger.info("订单添加成功: %s - %s", order_id, order_no)
            return self._order_to_dict(order)
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("订单添加失败: %s", e)
            raise
        finally:
            session.close()

    def delete_order(self, order_id: str) -> bool:
        """删除订单"""
        session = self._session()
        try:
            order = session.query(OrderModel).filter_by(id=order_id).first()
            if not order:
                return False
            session.delete(order)
            session.commit()
            logger.info("订单删除成功: %s", order_id)
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("订单删除失败: %s", e)
            raise
        finally:
            session.close()

    def update_order(self, order_id: str, **kwargs) -> bool:
        """更新订单"""
        session = self._session()
        try:
            order = session.query(OrderModel).filter_by(id=order_id).first()
            if not order:
                return False

            updatable_fields = [
                "status", "logistics_company", "tracking_no",
                "shipping_address", "remark"
            ]
            for key in updatable_fields:
                if key in kwargs:
                    setattr(order, key, kwargs[key])
            order.updated_at = datetime.utcnow()

            session.commit()
            logger.info("订单更新成功: %s", order_id)
            return True
        except SQLAlchemyError as e:
            session.rollback()
            logger.error("订单更新失败: %s", e)
            raise
        finally:
            session.close()

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """根据 ID 获取订单"""
        session = self._session()
        try:
            order = session.query(OrderModel).filter_by(id=order_id).first()
            return self._order_to_dict(order) if order else None
        finally:
            session.close()

    def get_all_orders(self) -> List[Dict[str, Any]]:
        """获取所有订单"""
        session = self._session()
        try:
            orders = session.query(OrderModel).order_by(OrderModel.created_at.desc()).all()
            return [self._order_to_dict(o) for o in orders]
        finally:
            session.close()

    def search_by_phone(
        self,
        phone: str,
        exact: bool = False,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """根据手机号搜索订单"""
        session = self._session()
        try:
            condition = (
                OrderModel.user_phone == phone
                if exact
                else OrderModel.user_phone.like(f"%{phone}%")
            )
            query = session.query(OrderModel).filter(condition)
            query = query.order_by(OrderModel.created_at.desc())
            if limit is not None:
                query = query.limit(limit)
            orders = query.all()
            return [self._order_to_dict(o) for o in orders]
        finally:
            session.close()

    def search_by_order_no(
        self,
        order_no: str,
        exact: bool = False,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """根据订单号搜索订单"""
        session = self._session()
        try:
            condition = (
                OrderModel.order_no == order_no
                if exact
                else OrderModel.order_no.like(f"%{order_no}%")
            )
            query = session.query(OrderModel).filter(condition)
            query = query.order_by(OrderModel.created_at.desc())
            if limit is not None:
                query = query.limit(limit)
            orders = query.all()
            return [self._order_to_dict(o) for o in orders]
        finally:
            session.close()

    def search_by_tracking_no(
        self,
        tracking_no: str,
        exact: bool = False,
        limit: int = None
    ) -> List[Dict[str, Any]]:
        """根据快递单号搜索订单"""
        session = self._session()
        try:
            condition = (
                OrderModel.tracking_no == tracking_no
                if exact
                else OrderModel.tracking_no.like(f"%{tracking_no}%")
            )
            query = session.query(OrderModel).filter(condition)
            query = query.order_by(OrderModel.created_at.desc())
            if limit is not None:
                query = query.limit(limit)
            orders = query.all()
            return [self._order_to_dict(o) for o in orders]
        finally:
            session.close()

    def track_order(
        self,
        identifier: str,
        exact: bool = True,
        limit: int = 1
    ) -> Optional[Dict[str, Any]]:
        """
        订单追踪：根据订单号或快递单号查找订单

        Args:
            identifier: 订单号或快递单号

        Returns:
            订单信息或 None
        """
        # 先按订单号搜索
        results = self.search_by_order_no(identifier, exact=exact, limit=limit)
        if results:
            return results[0]

        # 再按快递单号搜索
        results = self.search_by_tracking_no(identifier, exact=exact, limit=limit)
        if results:
            return results[0]

        return None

    def _order_to_dict(self, order: OrderModel) -> Dict[str, Any]:
        """订单模型转字典"""
        if not order:
            return None
        return {
            "id": order.id,
            "order_no": order.order_no,
            "user_phone": order.user_phone,
            "user_name": order.user_name,
            "product_id": order.product_id,
            "product_name": order.product_name,
            "quantity": order.quantity,
            "unit_price": float(order.unit_price) if order.unit_price else 0,
            "total_amount": float(order.total_amount) if order.total_amount else 0,
            "status": order.status,
            "logistics_company": order.logistics_company or "",
            "tracking_no": order.tracking_no or "",
            "shipping_address": order.shipping_address or "",
            "remark": order.remark or "",
            "created_at": order.created_at.isoformat() if order.created_at else "",
            "updated_at": order.updated_at.isoformat() if order.updated_at else ""
        }

    def _generate_id(self) -> str:
        """生成订单 ID"""
        return f"O{datetime.utcnow():%Y%m%d}{uuid.uuid4().hex[:6].upper()}"

    def close(self):
        """关闭引擎"""
        if self._engine:
            self._engine.dispose()


# 全局单例
_order_db: Optional[OrderDB] = None


def get_order_db() -> OrderDB:
    """获取订单数据访问层单例"""
    global _order_db
    if _order_db is None:
        _order_db = OrderDB()
    return _order_db
