# -*- coding: utf-8 -*-
"""
订单管理界面

功能：
- 订单 CRUD（增删改查）
- 按手机号、订单号、快递单号追踪订单
- 物流状态更新
"""

import os
import sys
import logging
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QScrollArea, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QDialog, QFormLayout,
    QFileDialog, QListWidget, QListWidgetItem, QLabel,
    QTextEdit, QSplitter, QGroupBox, QDateEdit, QComboBox
)
from PySide6.QtCore import Qt, Signal, QThread, QObject, QDate
from PySide6.QtGui import QDesktopServices

from qfluentwidgets import (
    CardWidget, BodyLabel, TitleLabel, SubtitleLabel,
    PushButton, PrimaryPushButton,
    ComboBox, TableWidget, SearchLineEdit, SpinBox,
    LineEdit, TextEdit, FluentIcon, ListWidget,
    MessageBox, InfoBar, InfoBarPosition, ProgressBar,
    Action, DatePicker, PlainTextEdit
)

from core.data.order_db import get_order_db
from core.config_utils.ui_utils import ProgressThrottler

logger = logging.getLogger(__name__)


class AddOrderDialog(QDialog):
    """添加订单对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("➕ 添加订单")
        self.setMinimumSize(580, 680)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        # 标题
        title = TitleLabel("➕ 添加订单")
        layout.addWidget(title)

        # 使用滚动区域容纳所有内容
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; background-color: transparent; }")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        # 基本信息
        info_card = CardWidget()
        info_layout = QFormLayout(info_card)
        info_layout.setContentsMargins(16, 12, 16, 12)
        info_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.order_no_edit = LineEdit()
        self.order_no_edit.setPlaceholderText("如：DD202609140001")
        info_layout.addRow("📋 订单号：", self.order_no_edit)

        phone_row = QHBoxLayout()
        phone_row.addWidget(BodyLabel("📱 手机号："))
        self.phone_edit = LineEdit()
        self.phone_edit.setPlaceholderText("用户手机号")
        phone_row.addWidget(self.phone_edit)
        phone_row.addStretch()
        info_layout.addRow(phone_row)

        name_row = QHBoxLayout()
        name_row.addWidget(BodyLabel("👤 姓名："))
        self.name_edit = LineEdit()
        self.name_edit.setPlaceholderText("用户姓名")
        name_row.addWidget(self.name_edit)
        name_row.addStretch()
        info_layout.addRow(name_row)

        scroll_layout.addWidget(info_card)

        # 商品信息
        product_card = CardWidget()
        product_layout = QFormLayout(product_card)
        product_layout.setContentsMargins(16, 12, 16, 12)
        product_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        pid_row = QHBoxLayout()
        pid_row.addWidget(BodyLabel("📦 商品ID："))
        self.product_id_edit = LineEdit()
        self.product_id_edit.setPlaceholderText("如 P001")
        pid_row.addWidget(self.product_id_edit)
        pid_row.addStretch()
        product_layout.addRow(pid_row)

        pname_row = QHBoxLayout()
        pname_row.addWidget(BodyLabel("📝 商品名称："))
        self.product_name_edit = LineEdit()
        self.product_name_edit.setPlaceholderText("商品名称")
        pname_row.addWidget(self.product_name_edit)
        pname_row.addStretch()
        product_layout.addRow(pname_row)

        qty_row = QHBoxLayout()
        qty_row.addWidget(BodyLabel("🔢 数量："))
        self.quantity_spin = SpinBox()
        self.quantity_spin.setRange(1, 999)
        self.quantity_spin.setValue(1)
        qty_row.addWidget(self.quantity_spin)
        qty_row.addStretch()
        product_layout.addRow(qty_row)

        scroll_layout.addWidget(product_card)

        # 金额信息
        amount_card = CardWidget()
        amount_layout = QFormLayout(amount_card)
        amount_layout.setContentsMargins(16, 12, 16, 12)
        amount_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        price_row = QHBoxLayout()
        price_row.addWidget(BodyLabel("💰 单价："))
        self.price_edit = LineEdit()
        self.price_edit.setPlaceholderText("元")
        price_row.addWidget(self.price_edit)
        price_row.addStretch()
        amount_layout.addRow(price_row)

        total_row = QHBoxLayout()
        total_row.addWidget(BodyLabel("💵 总额："))
        self.total_edit = LineEdit()
        self.total_edit.setPlaceholderText("元")
        total_row.addWidget(self.total_edit)
        total_row.addStretch()
        amount_layout.addRow(total_row)

        scroll_layout.addWidget(amount_card)

        # 物流信息
        logistics_card = CardWidget()
        logistics_layout = QFormLayout(logistics_card)
        logistics_layout.setContentsMargins(16, 12, 16, 12)
        logistics_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        addr_row = QHBoxLayout()
        addr_row.addWidget(BodyLabel("📍 收货地址："))
        self.address_edit = TextEdit()
        self.address_edit.setPlaceholderText("收货地址")
        self.address_edit.setMaximumHeight(60)
        addr_row.addWidget(self.address_edit)
        logistics_layout.addRow(addr_row)

        log_combo_row = QHBoxLayout()
        log_combo_row.addWidget(BodyLabel("🚚 快递公司："))
        self.logistics_combo = ComboBox()
        self.logistics_combo.addItems(["", "顺丰", "中通", "圆通", "韵达", "EMS", "京东物流"])
        log_combo_row.addWidget(self.logistics_combo)
        log_combo_row.addStretch()
        logistics_layout.addRow(log_combo_row)

        tracking_row = QHBoxLayout()
        tracking_row.addWidget(BodyLabel("📦 快递单号："))
        self.tracking_edit = LineEdit()
        self.tracking_edit.setPlaceholderText("自动生成的快递单号（可手动修改）")
        tracking_row.addWidget(self.tracking_edit)
        # 自动生成按钮
        auto_tracking_btn = PushButton(FluentIcon.SYNC, "自动生成")
        auto_tracking_btn.setFixedSize(70, 28)
        auto_tracking_btn.clicked.connect(self._auto_generate_tracking)
        tracking_row.addWidget(auto_tracking_btn)
        tracking_row.addStretch()
        logistics_layout.addRow(tracking_row)

        scroll_layout.addWidget(logistics_card)

        # 状态和备注
        status_card = CardWidget()
        status_layout = QFormLayout(status_card)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self.status_combo = ComboBox()
        self.status_combo.addItems(["待发货", "已发货", "运输中", "已签收", "已退货", "已取消"])
        self.status_combo.setCurrentIndex(0)
        status_layout.addRow("📊 订单状态：", self.status_combo)

        remark_row = QHBoxLayout()
        remark_row.addWidget(BodyLabel("📝 备注："))
        self.remark_edit = LineEdit()
        self.remark_edit.setPlaceholderText("备注（可选）")
        remark_row.addWidget(self.remark_edit)
        remark_row.addStretch()
        status_layout.addRow(remark_row)

        scroll_layout.addWidget(status_card)

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, 1)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = PushButton("取消")
        cancel_btn.setFixedWidth(100)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.submit_btn = PrimaryPushButton(FluentIcon.DOWNLOAD, "提交")
        self.submit_btn.setFixedWidth(100)
        self.submit_btn.clicked.connect(self._submit)
        btn_layout.addWidget(self.submit_btn)

        layout.addLayout(btn_layout)

    def _auto_generate_tracking(self):
        """自动生成快递单号"""
        import random
        # 生成随机快递单号
        prefix = random.choice(["SF", "ZTO", "YTO", "YD", "EMS"])
        tracking = prefix + str(random.randint(100000000000, 999999999999))
        self.tracking_edit.setText(tracking)

    def _submit(self):
        """提交订单"""
        order_no = self.order_no_edit.text().strip()
        phone = self.phone_edit.text().strip()
        name = self.name_edit.text().strip()
        product_id = self.product_id_edit.text().strip()
        product_name = self.product_name_edit.text().strip()
        quantity = self.quantity_spin.value()
        price_str = self.price_edit.text().strip()
        total_str = self.total_edit.text().strip()
        address = self.address_edit.toPlainText().strip()
        logistics = self.logistics_combo.currentText()
        tracking = self.tracking_edit.text().strip()
        status = self.status_combo.currentText()
        remark = self.remark_edit.text().strip()

        # 验证必填字段
        if not order_no or not phone or not product_id or not price_str or not total_str:
            InfoBar.warning(
                title="缺少必填项",
                content="请填写订单号、手机号、商品ID、单价和总额",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        try:
            unit_price = float(price_str)
            total_amount = float(total_str)
        except ValueError:
            InfoBar.warning(
                title="格式错误",
                content="单价和总额必须为数字",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        # 保存订单
        order_db = get_order_db()
        try:
            order = order_db.add_order(
                order_no=order_no,
                user_phone=phone,
                user_name=name,
                product_id=product_id,
                product_name=product_name,
                quantity=quantity,
                unit_price=unit_price,
                total_amount=total_amount,
                shipping_address=address,
                remark=remark
            )

            # 如果填写了物流信息，更新订单
            if logistics or tracking:
                update_kwargs = {}
                if logistics:
                    update_kwargs["logistics_company"] = logistics
                if tracking:
                    update_kwargs["tracking_no"] = tracking
                if status != "待发货":
                    update_kwargs["status"] = status
                if update_kwargs:
                    order_db.update_order(order["id"], **update_kwargs)
                    order = order_db.get_order(order["id"])

            InfoBar.success(
                title="添加成功",
                content=f"订单 {order_no} 已添加",
                parent=self,
                position=InfoBarPosition.TOP
            )
            self.accept()
        except Exception as e:
            logger.error("添加订单失败: %s", e)
            InfoBar.error(
                title="添加失败",
                content=str(e),
                parent=self,
                position=InfoBarPosition.TOP
            )


class OrderListPanel(QFrame):
    """订单列表面板"""

    order_selected = Signal(object)
    order_updated = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        self.order_db = get_order_db()
        self.current_orders = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 标题行
        top = QHBoxLayout()
        self.title = SubtitleLabel("📋 订单列表")
        top.addWidget(self.title)
        top.addStretch()

        # 添加订单按钮
        self.add_btn = PrimaryPushButton(FluentIcon.ADD, "添加订单")
        self.add_btn.clicked.connect(self._add_order)
        top.addWidget(self.add_btn)

        layout.addLayout(top)

        # 搜索框
        self.search = SearchLineEdit()
        self.search.setPlaceholderText("搜索订单号/手机号...")
        self.search.setFixedWidth(220)
        self.search.textChanged.connect(self._on_search)
        layout.addWidget(self.search)

        # 物流追踪
        track_card = CardWidget()
        track_layout = QHBoxLayout(track_card)
        track_layout.setContentsMargins(12, 8, 12, 8)
        track_layout.addWidget(BodyLabel("🔍 物流追踪："))
        self.tracking_edit = LineEdit()
        self.tracking_edit.setPlaceholderText("输入订单号或快递单号")
        self.tracking_edit.setFixedWidth(200)
        track_layout.addWidget(self.tracking_edit)
        self.track_btn = PushButton(FluentIcon.SEARCH, "查询")
        self.track_btn.clicked.connect(self._track_order)
        track_layout.addWidget(self.track_btn)
        track_layout.addStretch()
        layout.addWidget(track_card)

        # 订单表格
        self.order_table = TableWidget()
        self.order_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.order_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.order_table.verticalHeader().setVisible(False)
        self.order_table.setColumnCount(10)
        self.order_table.setHorizontalHeaderLabels([
            "订单号", "手机号", "商品", "金额", "状态", "快递公司", "快递单号", "收货地址", "创建时间", "操作"
        ])

        header = self.order_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        header.setSectionResizeMode(8, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(9, QHeaderView.Fixed)

        self.order_table.setColumnWidth(9, 100)
        self.order_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        layout.addWidget(self.order_table, 1)

        self._load_all()

    def _load_all(self):
        """加载所有订单"""
        orders = self.order_db.get_all_orders()
        self.current_orders = orders
        self._render_table(orders)

    def _render_table(self, orders: List[dict]):
        """渲染表格"""
        self.order_table.setRowCount(0)

        for order in orders:
            row_count = self.order_table.rowCount()
            self.order_table.insertRow(row_count)

            # 订单号
            order_no_item = QTableWidgetItem(order.get("order_no", ""))
            order_no_item.setTextAlignment(Qt.AlignCenter)
            self.order_table.setItem(row_count, 0, order_no_item)

            # 手机号
            phone_item = QTableWidgetItem(order.get("user_phone", ""))
            phone_item.setTextAlignment(Qt.AlignCenter)
            self.order_table.setItem(row_count, 1, phone_item)

            # 商品名称
            product_item = QTableWidgetItem(order.get("product_name", order.get("product_id", "")))
            product_item.setTextAlignment(Qt.AlignCenter)
            self.order_table.setItem(row_count, 2, product_item)

            # 金额
            amount_item = QTableWidgetItem(f"¥{order.get('total_amount', 0):.2f}")
            amount_item.setTextAlignment(Qt.AlignCenter)
            self.order_table.setItem(row_count, 3, amount_item)

            # 状态（带颜色）
            status = order.get("status", "待发货")
            status_item = QTableWidgetItem(status)
            status_item.setTextAlignment(Qt.AlignCenter)
            self._set_status_color(status_item, status)
            self.order_table.setItem(row_count, 4, status_item)

            # 快递公司
            company = order.get("logistics_company", "") or "-"
            company_item = QTableWidgetItem(company)
            company_item.setTextAlignment(Qt.AlignCenter)
            self.order_table.setItem(row_count, 5, company_item)

            # 快递单号
            tracking = order.get("tracking_no", "") or "-"
            tracking_item = QTableWidgetItem(tracking)
            tracking_item.setTextAlignment(Qt.AlignCenter)
            self.order_table.setItem(row_count, 6, tracking_item)

            # 收货地址
            address = order.get("shipping_address", "") or "-"
            address_item = QTableWidgetItem(address)
            address_item.setTextAlignment(Qt.AlignLeft)
            self.order_table.setItem(row_count, 7, address_item)

            # 创建时间
            created = order.get("created_at", "")[:10] if order.get("created_at") else ""
            time_item = QTableWidgetItem(created)
            time_item.setTextAlignment(Qt.AlignCenter)
            self.order_table.setItem(row_count, 8, time_item)

            # 操作按钮
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.setSpacing(4)

            detail_btn = PushButton("详情")
            detail_btn.setFixedSize(50, 26)
            detail_btn.clicked.connect(lambda checked, o=order: self._show_order_detail(o))
            btn_layout.addWidget(detail_btn)

            delete_btn = PushButton("删除")
            delete_btn.setFixedSize(50, 26)
            delete_btn.clicked.connect(lambda checked, o=order: self._delete_order(o))
            btn_layout.addWidget(delete_btn)

            self.order_table.setCellWidget(row_count, 9, btn_widget)

        # 更新统计
        self.order_updated.emit()

    def _set_status_color(self, item: QTableWidgetItem, status: str):
        """设置状态颜色"""
        from PySide6.QtGui import QColor
        colors = {
            "待发货": "#FF9800",
            "已发货": "#2196F3",
            "运输中": "#03A9F4",
            "已签收": "#4CAF50",
            "已退货": "#9C27B0",
            "已取消": "#9E9E9E"
        }
        color = colors.get(status, "#666666")
        item.setForeground(QColor(color))

    def _add_order(self):
        """添加订单"""
        dialog = AddOrderDialog(self.window())
        if dialog.exec():
            self._load_all()

    def _refresh(self):
        """刷新订单列表"""
        self._load_all()

    def _on_search(self, text: str):
        """搜索订单"""
        if not text:
            self._render_table(self.current_orders)
            return

        query = text.lower()
        filtered = []
        for order in self.current_orders:
            if (query in order.get("order_no", "").lower() or
                query in order.get("user_phone", "").lower() or
                query in order.get("product_name", "").lower()):
                filtered.append(order)

        self._render_table(filtered)

    def _track_order(self):
        """物流追踪"""
        identifier = self.tracking_edit.text().strip()
        if not identifier:
            InfoBar.warning(
                title="请输入查询条件",
                content="请输入订单号或快递单号",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        order = self.order_db.track_order(identifier)
        if order:
            self._show_order_detail(order)
        else:
            InfoBar.warning(
                title="未找到订单",
                content=f"未找到订单号或快递单号为「{identifier}」的订单",
                parent=self,
                position=InfoBarPosition.TOP
            )

    def _show_order_detail(self, order: dict):
        """显示订单详情"""
        dialog = QDialog(self)
        dialog.setWindowTitle(f"📋 订单详情 - {order.get('order_no', '')}")
        dialog.setFixedSize(550, 520)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # 标题
        title = SubtitleLabel(f"📋 {order.get('order_no', '')}")
        layout.addWidget(title)

        # 基本信息
        info_card = CardWidget()
        info_layout = QFormLayout(info_card)
        info_layout.setContentsMargins(16, 12, 16, 12)

        info_layout.addRow("📱 手机号：", BodyLabel(order.get("user_phone", "")))
        info_layout.addRow("👤 姓名：", BodyLabel(order.get("user_name", "") or "-"))
        info_layout.addRow("📦 商品：", BodyLabel(order.get("product_name", order.get("product_id", ""))))
        info_layout.addRow("🔢 数量：", BodyLabel(str(order.get("quantity", 1))))
        info_layout.addRow("💰 单价：", BodyLabel(f"¥{order.get('unit_price', 0):.2f}"))
        info_layout.addRow("💵 总额：", BodyLabel(f"¥{order.get('total_amount', 0):.2f}"))
        info_layout.addRow("📊 状态：", BodyLabel(order.get("status", "")))
        info_layout.addRow("🚚 快递：", BodyLabel(order.get("logistics_company", "") or "-"))
        info_layout.addRow("📦 单号：", BodyLabel(order.get("tracking_no", "") or "-"))
        info_layout.addRow("📍 地址：", BodyLabel(order.get("shipping_address", "") or "-"))
        info_layout.addRow("📝 备注：", BodyLabel(order.get("remark", "") or "-"))
        info_layout.addRow("🕐 创建：", BodyLabel(order.get("created_at", "")[:19] if order.get("created_at") else "-"))

        layout.addWidget(info_card)

        # 更新状态按钮
        update_layout = QHBoxLayout()
        update_layout.addWidget(BodyLabel("更新状态："))
        status_combo = ComboBox()
        status_combo.addItems(["待发货", "已发货", "运输中", "已签收", "已退货", "已取消"])
        current_status = order.get("status", "待发货")
        idx = status_combo.findText(current_status)
        if idx >= 0:
            status_combo.setCurrentIndex(idx)
        update_layout.addWidget(status_combo)
        update_layout.addStretch()
        layout.addLayout(update_layout)

        # 更新和关闭按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        close_btn = PushButton("关闭")
        close_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(close_btn)

        save_btn = PrimaryPushButton(FluentIcon.SAVE, "保存修改")
        save_btn.clicked.connect(lambda: self._save_order_update(order, status_combo.currentText()))
        btn_layout.addWidget(save_btn)

        layout.addLayout(btn_layout)

        dialog.exec()

    def _save_order_update(self, order: dict, new_status: str):
        """保存订单更新"""
        try:
            self.order_db.update_order(order["id"], status=new_status)
            InfoBar.success(
                title="保存成功",
                content=f"订单 {order['order_no']} 状态已更新为 {new_status}",
                parent=self,
                position=InfoBarPosition.TOP
            )
            self._load_all()
        except Exception as e:
            logger.error("更新订单失败: %s", e)
            InfoBar.error(
                title="保存失败",
                content=str(e),
                parent=self,
                position=InfoBarPosition.TOP
            )

    def _delete_order(self, order: dict):
        """删除订单"""
        w = MessageBox(
            "确认删除",
            f"确定要删除订单「{order.get('order_no', '')}」吗？\n\n删除后数据将无法恢复。",
            self
        )
        if w.exec():
            try:
                self.order_db.delete_order(order["id"])
                InfoBar.success(
                    title="删除成功",
                    content=f"订单 {order.get('order_no', '')} 已删除",
                    parent=self,
                    position=InfoBarPosition.TOP
                )
                self._load_all()
            except Exception as e:
                logger.error("删除订单失败: %s", e)
                InfoBar.error(
                    title="删除失败",
                    content=str(e),
                    parent=self,
                    position=InfoBarPosition.TOP
                )

    def _on_cell_double_clicked(self, row: int, col: int):
        """单元格双击事件"""
        order_no_item = self.order_table.item(row, 0)
        if order_no_item:
            order_no = order_no_item.text()
            for order in self.current_orders:
                if order.get("order_no") == order_no:
                    self._show_order_detail(order)
                    break


class OrderToolbar(QFrame):
    """底部工具栏"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("""
            OrderToolbar {
                border: none;
                border-top: 1px solid rgba(0,0,0,0.1);
                background-color: #FAFAFA;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(16)

        self.stats_label = BodyLabel("📋 订单加载中...")
        self.stats_label.setStyleSheet("color: gray;")
        layout.addWidget(self.stats_label)

        layout.addStretch()

    def refresh_stats(self):
        """刷新统计"""
        try:
            order_db = get_order_db()
            orders = order_db.get_all_orders()
            total = len(orders)
            pending = sum(1 for o in orders if o.get("status") == "待发货")
            shipped = sum(1 for o in orders if o.get("status") == "已发货")
            delivered = sum(1 for o in orders if o.get("status") == "已签收")
            self.stats_label.setText(
                f"📋 共 {total} 个订单 | 待发货: {pending} | 已发货: {shipped} | 已签收: {delivered}"
            )
        except Exception:
            self.stats_label.setText("📋 订单统计加载中...")


class OrderInterface(QWidget):
    """订单管理界面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("order_interface")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # 中间：订单列表
        self.order_list_panel = OrderListPanel()
        content_layout.addWidget(self.order_list_panel, 1)

        layout.addWidget(content, 1)

        # 底部工具栏
        self.toolbar = OrderToolbar()
        layout.addWidget(self.toolbar)

        # 初始刷新统计
        self.toolbar.refresh_stats()

    def on_order_updated(self):
        """订单变更后刷新"""
        self.toolbar.refresh_stats()
