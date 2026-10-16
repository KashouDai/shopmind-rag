# -*- coding: utf-8 -*-
"""
知识库管理界面 - Markdown 文档上传版本

改造说明 (v4.0.0):
- 从 Q&A 手动录入 → Markdown 文档上传自动切片
- 左侧面板显示分类列表（兼容旧 UI）
- 中间面板显示文档内容和上传/刷新按钮
- 右侧面板显示 RAG 配置和索引状态
"""

import os
import sys
import logging
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QScrollArea, QTableWidgetItem, QHeaderView,
    QAbstractItemView, QDialog, QFormLayout,
    QFileDialog, QListWidget, QListWidgetItem, QLabel,
    QTextEdit, QSplitter, QToolButton
)
from PySide6.QtCore import Qt, Signal, QThread, QObject, QUrl
from PySide6.QtGui import QDesktopServices, QIcon

from qfluentwidgets import (
    CardWidget, BodyLabel, TitleLabel, SubtitleLabel,
    PushButton, PrimaryPushButton,
    ComboBox, TableWidget, SearchLineEdit, SpinBox,
    LineEdit, TextEdit, FluentIcon, ListWidget,
    MessageBox, InfoBar, InfoBarPosition, ProgressBar,
    Action
)

from core.shared_data import KnowledgeStore, KnowledgeItem
from core.config_utils.ui_utils import ProgressThrottler
from core.rag.doc_loader import get_doc_loader
from core.rag.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class UploadDocWorker(QObject):
    """上传文档工作线程"""
    finished = Signal(bool, str, dict)
    failed = Signal(str)
    progress = Signal(str, int, int)

    def __init__(self, file_path: str, category: str = ""):
        super().__init__()
        self.file_path = file_path
        self.category = category
        self._throttler = None

    def run(self):
        try:
            self._throttler = ProgressThrottler(
                callback=lambda stage, current, total: self.progress.emit(str(stage), int(current), int(total)),
                min_interval=0.1,
                min_progress_change=0.02
            )

            def throttled_cb(stage: str, current: int, total: int):
                self._throttler.update(stage, current, total)

            loader = get_doc_loader()
            result = loader.load_from_file(self.file_path, self.category)

            if self._throttler:
                self._throttler.finish()

            self.finished.emit(True, "上传成功", result)
        except Exception as e:
            logger.exception("文档上传失败")
            self.failed.emit(str(e))


class RebuildIndexWorker(QObject):
    """重建索引工作线程"""
    finished = Signal(bool, str)
    failed = Signal(str)
    progress = Signal(str, int, int)

    def __init__(self):
        super().__init__()
        self._throttler = None

    def run(self):
        try:
            self._throttler = ProgressThrottler(
                callback=lambda stage, current, total: self.progress.emit(str(stage), int(current), int(total)),
                min_interval=0.1,
                min_progress_change=0.02
            )

            def throttled_cb(stage: str, current: int, total: int):
                self._throttler.update(stage, current, total)

            ks = KnowledgeStore()
            success, message = ks.rebuild_vector_index(progress_callback=throttled_cb)

            if self._throttler:
                self._throttler.finish()

            self.finished.emit(bool(success), str(message))
        except Exception as e:
            self.failed.emit(str(e))


class UploadDocDialog(QDialog):
    """上传 Markdown 文档对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📤 上传知识库文档")
        self.setFixedSize(500, 320)
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        self.setStyleSheet("""
            QDialog { background-color: #FAFAFA; }
            QLabel { color: #333333; }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        # 标题
        title = TitleLabel("📤 上传知识库文档")
        layout.addWidget(title)

        # 文件选择
        file_card = CardWidget()
        file_layout = QVBoxLayout(file_card)
        file_layout.setContentsMargins(16, 12, 16, 12)
        file_layout.addWidget(BodyLabel("📄 选择 Markdown 文件："))

        self.file_path_edit = LineEdit()
        self.file_path_edit.setPlaceholderText("选择或拖拽 .md 文件到这里...")
        self.file_path_edit.setReadOnly(True)
        file_layout.addWidget(self.file_path_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        browse_btn = PushButton(FluentIcon.FOLDER, "浏览...")
        browse_btn.clicked.connect(self._browse_file)
        btn_row.addWidget(browse_btn)
        file_layout.addLayout(btn_row)
        layout.addWidget(file_card)

        # 分类
        meta_card = CardWidget()
        meta_layout = QFormLayout(meta_card)
        meta_layout.setContentsMargins(16, 12, 16, 12)
        self.category_combo = ComboBox()
        self.category_combo.addItems(["通用", "售后政策", "物流配送", "促销活动", "商品咨询", "支付问题", "服务咨询", "订单咨询"])
        meta_layout.addRow("📁 分类：", self.category_combo)
        layout.addWidget(meta_card)

        # 进度条（初始隐藏）
        self.progress_bar = ProgressBar()
        self.progress_bar.setHidden(True)
        layout.addWidget(self.progress_bar)

        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = PushButton("取消")
        cancel_btn.setFixedWidth(100)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        self.upload_btn = PrimaryPushButton(FluentIcon.DOWNLOAD, "上传")
        self.upload_btn.setFixedWidth(100)
        self.upload_btn.clicked.connect(self._upload)
        btn_layout.addWidget(self.upload_btn)

        layout.addLayout(btn_layout)

    def _browse_file(self):
        """浏览文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 Markdown 文件", "",
            "Markdown 文件 (*.md);;所有文件 (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)

    def _upload(self):
        """执行上传"""
        file_path = self.file_path_edit.text().strip()
        if not file_path or not os.path.exists(file_path):
            InfoBar.warning(
                title="选择文件",
                content="请先选择一个有效的 Markdown 文件",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        if not file_path.lower().endswith('.md'):
            InfoBar.warning(
                title="文件格式",
                content="仅支持 .md (Markdown) 格式的文档",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        self.upload_btn.setEnabled(False)
        self.progress_bar.setHidden(False)
        self.progress_bar.setValue(0)

        self.category = self.category_combo.currentText()

        # 启动上传线程
        self.worker_thread = QThread()
        self.worker = UploadDocWorker(file_path, self.category)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_upload_finished)
        self.worker.failed.connect(self._on_upload_failed)
        self.worker.progress.connect(self._on_upload_progress)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self._cleanup_thread)

        self.worker_thread.start()

    def _on_upload_finished(self, success: bool, message: str, stats: dict):
        self.upload_btn.setEnabled(True)
        self.progress_bar.setHidden(True)

        if success:
            InfoBar.success(
                title="上传成功",
                content=f"{message}（新增 {stats.get('added', 0)} 个 chunk）",
                parent=self,
                position=InfoBarPosition.TOP
            )
            self.accept()
        else:
            InfoBar.error(
                title="上传失败",
                content=message,
                parent=self,
                position=InfoBarPosition.TOP
            )

    def _on_upload_failed(self, message: str):
        self.upload_btn.setEnabled(True)
        self.progress_bar.setHidden(True)
        InfoBar.error(
            title="上传失败",
            content=message,
            parent=self,
            position=InfoBarPosition.TOP
        )

    def _on_upload_progress(self, stage: str, current: int, total: int):
        if total > 0:
            self.progress_bar.setValue(int(current / total * 100))

    def _cleanup_thread(self):
        if hasattr(self, 'worker'):
            self.worker.deleteLater()
        if hasattr(self, 'worker_thread'):
            self.worker_thread.deleteLater()


class KnowledgeTreePanel(QFrame):
    """知识库分类面板"""

    category_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setStyleSheet("""
            QFrame {
                border-right: 1px solid rgba(0,0,0,0.1);
                background-color: #FAFAFA;
            }
        """)

        self.knowledge_store = KnowledgeStore()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(16)

        # 标题
        title = TitleLabel("📚 知识分类")
        layout.addWidget(title)

        # 分类列表
        self.list_widget = ListWidget()
        self.list_widget.setStyleSheet("""
            ListWidget {
                border: none;
                background-color: transparent;
                outline: none;
            }
            ListWidget::item {
                padding: 8px 16px;
                border-radius: 8px;
                margin: 1px 0;
                border: none;
                outline: none;
            }
            ListWidget::item:hover {
                background-color: rgba(0, 120, 212, 0.1);
            }
            ListWidget::item:selected {
                background-color: rgba(0, 120, 212, 0.2);
                color: #0078d4;
            }
        """)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget, 1)

        self._load_data()

    def _load_data(self):
        """加载分类列表"""
        self.list_widget.clear()

        # 全部
        self.list_widget.addItem("📋 全部知识")

        # 分类图标（删除商品信息、订单咨询）
        cat_icons = {
            "售后政策": "🔄",
            "物流配送": "🚚",
            "促销活动": "🎉",
            "商品咨询": "📦",
            "支付问题": "💳",
            "服务咨询": "💁",
            "通用": "📝"
        }

        for cat, icon in cat_icons.items():
            self.list_widget.addItem(f"{icon} {cat}")

    def refresh(self):
        self._load_data()

    def _on_item_clicked(self, item):
        text = item.text()
        if "全部知识" in text:
            self.category_selected.emit("全部")
        else:
            # 提取分类名（去掉图标）
            cat = text.split(" ", 1)[1] if " " in text else text
            self.category_selected.emit(cat)


class DocumentListPanel(QFrame):
    """文档列表面板"""

    chunk_selected = Signal(object)
    document_opened = Signal(str)
    document_deleted = Signal(str)
    document_updated = Signal(str)  # 新增：文档更新信号
    doc_selected = Signal(str)  # 文档被选中时发出文档名

    def __init__(self, parent=None):
        super().__init__(parent)

        self.doc_loader = get_doc_loader()
        self.current_chunks = []
        self.current_doc_name = ""
        self.current_category = "全部"  # 当前选中的分类

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 标题行（搜索框 + 上传按钮同行）
        top = QHBoxLayout()
        self.title = SubtitleLabel("📄 文档列表")
        top.addWidget(self.title)
        top.addStretch()

        # 搜索框
        self.search = SearchLineEdit()
        self.search.setPlaceholderText("搜索文档...")
        self.search.setFixedWidth(180)
        self.search.textChanged.connect(self._on_search)
        top.addWidget(self.search)

        # 上传按钮
        self.upload_btn = PrimaryPushButton(FluentIcon.ADD, "上传文档")
        self.upload_btn.clicked.connect(self._upload_doc)
        top.addWidget(self.upload_btn)

        layout.addLayout(top)

        # 文档表格
        self.doc_table = TableWidget()
        # 允许通过按键编辑分类和版本号，双击用于打开文档
        self.doc_table.setEditTriggers(QAbstractItemView.EditKeyPressed)
        self.doc_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.doc_table.verticalHeader().setVisible(False)
        self.doc_table.setColumnCount(4)  # 文档名称、分类、版本、操作
        self.doc_table.setHorizontalHeaderLabels(["文档名称", "分类", "版本号", "操作"])

        # 设置列宽
        header = self.doc_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Fixed)
        header.setSectionResizeMode(2, QHeaderView.Fixed)
        header.setSectionResizeMode(3, QHeaderView.Fixed)
        self.doc_table.setColumnWidth(1, 120)
        self.doc_table.setColumnWidth(2, 80)
        self.doc_table.setColumnWidth(3, 60)

        self.doc_table.cellDoubleClicked.connect(self._on_cell_double_clicked)
        self.doc_table.cellChanged.connect(self._on_cell_changed)
        self.doc_table.currentCellChanged.connect(self._on_row_selected)
        self.doc_table.selectionModel().selectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.doc_table, 1)

        self._load_all()

    def _on_category_selected(self, category: str):
        """处理分类选择"""
        self.current_category = category
        self._load_all(category)

    @staticmethod
    def _infer_category_from_filename(filename: str) -> str:
        """从文件名推断分类"""
        name = filename.replace('.md', '').replace('.Md', '')
        cat_map = {
            '售后政策': '售后政策',
            '物流配送': '物流配送',
            '促销活动': '促销活动',
            '商品咨询': '商品咨询',
            '支付问题': '支付问题',
            '服务咨询': '服务咨询',
            '订单咨询': '订单咨询',
        }
        for keyword, category in cat_map.items():
            if keyword in name:
                return category
        return '通用'

    def _load_all(self, category_filter: str = None):
        """加载所有文档（从 documents 目录和 doc_index.json）"""
        # 获取 documents 目录中的所有 .md 文件
        doc_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "documents"
        )
        self.current_docs = {}

        if os.path.exists(doc_dir):
            for filename in os.listdir(doc_dir):
                if filename.endswith('.md'):
                    # 从 doc_index.json 获取元数据
                    stats = self.doc_loader.get_document_stats()
                    if filename in stats.get('documents', {}):
                        self.current_docs[filename] = stats['documents'][filename]
                    else:
                        # 文档存在但未索引，尝试从文件名推断分类
                        inferred_cat = self._infer_category_from_filename(filename)
                        self.current_docs[filename] = {
                            'category': inferred_cat,
                            'chunk_count': 0,
                            'file_hash': '',
                            'updated_at': '',
                            'version': '1.0'
                        }

        # 如果指定了分类筛选
        if category_filter and category_filter != "全部":
            filtered_docs = {}
            for doc_name, doc_info in self.current_docs.items():
                if doc_info.get('category', '通用') == category_filter:
                    filtered_docs[doc_name] = doc_info
            self.current_docs = filtered_docs

        # 清空表格
        self.doc_table.setRowCount(0)

        # 填充表格
        for doc_name, doc_info in self.current_docs.items():
            row_count = self.doc_table.rowCount()
            self.doc_table.insertRow(row_count)

            # 文档名称（不可编辑，双击打开）
            name_item = QTableWidgetItem(doc_name)
            name_item.setToolTip(doc_name)
            name_item.setTextAlignment(Qt.AlignCenter)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.doc_table.setItem(row_count, 0, name_item)

            # 分类（可编辑）
            category = doc_info.get('category', '') or self._infer_category_from_filename(doc_name)
            cat_item = QTableWidgetItem(category)
            cat_item.setTextAlignment(Qt.AlignCenter)
            cat_item.setFlags(cat_item.flags() | Qt.ItemIsEditable)
            self.doc_table.setItem(row_count, 1, cat_item)

            # 版本号（可编辑）
            version = doc_info.get('version', '1.0')
            version_item = QTableWidgetItem(version)
            version_item.setTextAlignment(Qt.AlignCenter)
            version_item.setFlags(version_item.flags() | Qt.ItemIsEditable)
            self.doc_table.setItem(row_count, 2, version_item)

            # 删除按钮（纯图标）
            delete_btn = QToolButton()
            delete_btn.setIcon(FluentIcon.DELETE.icon())
            delete_btn.setToolTip("删除")
            delete_btn.setFixedSize(28, 28)
            delete_btn.clicked.connect(lambda checked, name=doc_name: self._delete_document(name))
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.addWidget(delete_btn)
            self.doc_table.setCellWidget(row_count, 3, btn_widget)

        # 选中第一行
        if self.doc_table.rowCount() > 0:
            self.doc_table.selectRow(0)

    def _view_document(self, doc_name: str):
        """查看文档详情"""
        self.current_doc_name = doc_name
        stats = self.doc_loader.get_document_stats()

        if doc_name not in stats.get('documents', {}):
            return

        doc_info = stats['documents'][doc_name]
        chunk_count = doc_info.get('chunk_count', 0)

        # 从 Milvus 获取实际内容
        from core.rag.vector_store import get_vector_store
        vs = get_vector_store()
        all_chunks = vs.get_all_chunks(source_file=doc_name)

        # 创建详情对话框
        dialog = QDialog(self)
        dialog.setWindowTitle(f"📄 {doc_name}")
        dialog.setFixedSize(700, 500)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 标题
        title = SubtitleLabel(f"📄 {doc_name}")
        layout.addWidget(title)

        # 统计信息
        info_label = BodyLabel(f"分类: {doc_info.get('category', '通用')} | Chunk 数: {chunk_count} | 版本: {doc_info.get('version', '1.0')}")
        layout.addWidget(info_label)

        # 内容区域
        if all_chunks:
            content_parts = []
            for chunk_data in all_chunks:
                heading = chunk_data.get('section', '')
                content = chunk_data.get('content', '')
                if heading:
                    content_parts.append(f"## {heading}\n{content}")
                else:
                    content_parts.append(content)
            content_text = "\n\n".join(content_parts)
        else:
            content_text = "暂无内容"

        content_edit = TextEdit()
        content_edit.setReadOnly(True)
        content_edit.setPlainText(content_text)
        layout.addWidget(content_edit, 1)

        # 关闭按钮
        close_btn = PushButton("关闭")
        close_btn.clicked.connect(dialog.reject)
        layout.addWidget(close_btn)

        dialog.exec()

    def _refresh(self):
        """刷新文档列表"""
        self._load_all(self.current_category)
        self.document_deleted.emit("")

    def _upload_doc(self):
        """上传文档"""
        dialog = UploadDocDialog(self.window())
        if dialog.exec():
            self._load_all()
            self.document_deleted.emit("")

    def _on_search(self, text: str):
        """搜索文档"""
        # 清空表格
        self.doc_table.setRowCount(0)

        if not text:
            # 重新加载全部
            self._load_all(self.current_category)
            return

        stats = self.doc_loader.get_document_stats()
        query_lower = text.lower()

        # 过滤文档
        filtered_docs = {}
        for doc_name, doc_info in stats.get('documents', {}).items():
            # 搜索文档名
            if query_lower in doc_name.lower():
                filtered_docs[doc_name] = doc_info
                continue
            # 搜索章节标题 - 从 Milvus 获取内容
            from core.rag.vector_store import get_vector_store
            vs = get_vector_store()
            chunks = vs.get_all_chunks(source_file=doc_name)
            for chunk_data in chunks:
                heading = chunk_data.get('section', '').lower()
                content = chunk_data.get('content', '').lower()
                if query_lower in heading or query_lower in content:
                    filtered_docs[doc_name] = doc_info
                    break

        # 填充表格
        for doc_name, doc_info in filtered_docs.items():
            row_count = self.doc_table.rowCount()
            self.doc_table.insertRow(row_count)

            # 文档名称
            name_item = QTableWidgetItem(doc_name)
            name_item.setToolTip(doc_name)
            name_item.setTextAlignment(Qt.AlignCenter)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.doc_table.setItem(row_count, 0, name_item)

            # 分类（可编辑）
            category = doc_info.get('category', '') or self._infer_category_from_filename(doc_name)
            cat_item = QTableWidgetItem(category)
            cat_item.setTextAlignment(Qt.AlignCenter)
            cat_item.setFlags(cat_item.flags() | Qt.ItemIsEditable)
            self.doc_table.setItem(row_count, 1, cat_item)

            # 版本号（可编辑）
            version = doc_info.get('version', '1.0')
            version_item = QTableWidgetItem(version)
            version_item.setTextAlignment(Qt.AlignCenter)
            version_item.setFlags(version_item.flags() | Qt.ItemIsEditable)
            self.doc_table.setItem(row_count, 2, version_item)

            # 删除按钮（纯图标）
            delete_btn = QToolButton()
            delete_btn.setIcon(FluentIcon.DELETE.icon())
            delete_btn.setToolTip("删除")
            delete_btn.setFixedSize(28, 28)
            delete_btn.clicked.connect(lambda checked, name=doc_name: self._delete_document(name))
            btn_widget = QWidget()
            btn_layout = QHBoxLayout(btn_widget)
            btn_layout.setContentsMargins(0, 0, 0, 0)
            btn_layout.addWidget(delete_btn)
            self.doc_table.setCellWidget(row_count, 3, btn_widget)

        # 选中第一行
        if self.doc_table.rowCount() > 0:
            self.doc_table.selectRow(0)
            first_doc = list(filtered_docs.keys())[0]
            self._view_document(first_doc)

    def _on_row_selected(self, row: int, col: int):
        """行被选中时发出信号"""
        if row >= 0:
            name_item = self.doc_table.item(row, 0)
            if name_item:
                self.doc_selected.emit(name_item.text())

    def _on_selection_changed(self):
        """selection 变更时触发"""
        current = self.doc_table.currentRow()
        if current >= 0:
            name_item = self.doc_table.item(current, 0)
            if name_item:
                self.doc_selected.emit(name_item.text())

    def _on_cell_double_clicked(self, row: int, col: int):
        """单元格双击事件 — 打开原文件"""
        name_item = self.doc_table.item(row, 0)
        if name_item:
            doc_name = name_item.text()
            self._open_document(doc_name)

    def _on_cell_changed(self, row: int, col: int):
        """单元格内容改变事件"""
        name_item = self.doc_table.item(row, 0)
        if not name_item:
            return

        doc_name = name_item.text()

        if col == 1:  # 分类
            new_category = self.doc_table.item(row, 1).text()
            self._update_doc_category(doc_name, new_category)
        elif col == 2:  # 版本号
            new_version = self.doc_table.item(row, 2).text()
            self._update_doc_version(doc_name, new_version)

    def _update_doc_category(self, doc_name: str, new_category: str):
        """更新文档分类"""
        stats = self.doc_loader.get_document_stats()
        if doc_name in stats.get('documents', {}):
            doc_info = stats['documents'][doc_name]
            # 更新 doc_index.json
            doc_info['category'] = new_category
            self.doc_loader._index[doc_name]['category'] = new_category
            self.doc_loader._save_index()
            self.document_updated.emit(doc_name)

    def _update_doc_version(self, doc_name: str, new_version: str):
        """更新文档版本号"""
        stats = self.doc_loader.get_document_stats()
        if doc_name in stats.get('documents', {}):
            # 更新 doc_index.json
            self.doc_loader._index[doc_name]['version'] = new_version
            self.doc_loader._save_index()
            self.document_updated.emit(doc_name)

    def _open_document(self, doc_name: str = None):
        """打开原文件"""
        if not doc_name:
            # 从选中行获取
            selected_rows = self.doc_table.selectedItems()
            if selected_rows:
                name_item = self.doc_table.item(selected_rows[0].row(), 0)
                if name_item:
                    doc_name = name_item.text()
            if not doc_name:
                return

        # 从 documents 目录打开
        doc_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data", "documents"
        )
        file_path = os.path.join(doc_dir, doc_name)

        if os.path.exists(file_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))
            self.document_opened.emit(file_path)
        else:
            InfoBar.warning(
                title="文件不存在",
                content=f"文档文件 {doc_name} 不存在",
                parent=self,
                position=InfoBarPosition.TOP
            )

    def _delete_document(self, doc_name: str = None):
        """删除文档"""
        if not doc_name:
            # 从选中行获取
            selected_rows = self.doc_table.selectedItems()
            if selected_rows:
                name_item = self.doc_table.item(selected_rows[0].row(), 0)
                if name_item:
                    doc_name = name_item.text()
            if not doc_name:
                return

        w = MessageBox("确认删除", f"确定要删除文档「{doc_name}」吗？\n\n删除后相关知识 chunk 将从 Milvus 中移除。", self)
        if w.exec():
            self.doc_loader.remove_document(doc_name)
            self._load_all()
            self.document_deleted.emit(doc_name)
            InfoBar.success(
                title="删除成功",
                content=f"文档 {doc_name} 已删除",
                parent=self,
                position=InfoBarPosition.TOP
            )


class RightPanel(QFrame):
    """右侧面板：上半部分文档信息，下半部分RAG配置"""

    index_rebuilt = Signal()
    category_changed = Signal(str, str)
    version_changed = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setStyleSheet("QFrame { border-left: 1px solid rgba(0,0,0,0.1); background: #f8f8f8; }")
        self._initialized = False

        from core.config_utils.config import Config
        self.config = Config()
        self.doc_loader = get_doc_loader()

        self._rebuild_running = False
        self._rebuild_thread: QThread | None = None
        self._rebuild_worker: RebuildIndexWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ===== 上半部分：文档信息 =====
        info_card = CardWidget()
        info_inner = QVBoxLayout(info_card)
        info_inner.setContentsMargins(14, 14, 14, 10)
        info_inner.setSpacing(8)

        info_title = SubtitleLabel("📄 文档信息")
        info_inner.addWidget(info_title)

        self.info_form = QFormLayout()
        self.info_form.setSpacing(8)
        self.info_form.setFormAlignment(Qt.AlignTop)
        self.info_form.setLabelAlignment(Qt.AlignRight)

        self.name_label = BodyLabel("未选择文档")
        self.name_label.setStyleSheet("font-weight: bold; color: #333;")
        self.info_form.addRow("文档名称:", self.name_label)

        self.category_combo = ComboBox()
        self.category_combo.addItems(["通用", "售后政策", "物流配送", "促销活动", "商品咨询", "支付问题", "服务咨询", "订单咨询"])
        self.category_combo.currentTextChanged.connect(self._on_category_changed)
        self.info_form.addRow("分类:", self.category_combo)

        self.version_edit = LineEdit()
        self.version_edit.setFixedWidth(120)
        self.version_edit.textChanged.connect(self._on_version_changed)
        self.info_form.addRow("版本号:", self.version_edit)

        self.chunk_label = BodyLabel("-")
        self.chunk_label.setStyleSheet("color: #666;")
        self.info_form.addRow("Chunk 数:", self.chunk_label)

        self.updated_label = BodyLabel("-")
        self.updated_label.setStyleSheet("color: #999; font-size: 10px;")
        self.info_form.addRow("更新时间:", self.updated_label)

        info_inner.addLayout(self.info_form)
        info_inner.addStretch()
        layout.addWidget(info_card)

        # ===== 下半部分：RAG 配置 =====
        config_card = CardWidget()
        config_inner = QVBoxLayout(config_card)
        config_inner.setContentsMargins(14, 10, 14, 14)
        config_inner.setSpacing(8)

        config_title = SubtitleLabel("⚙️ RAG 配置")
        config_inner.addWidget(config_title)

        self.config_form = QFormLayout()
        self.config_form.setSpacing(10)
        self.config_form.setFormAlignment(Qt.AlignTop)
        self.config_form.setLabelAlignment(Qt.AlignRight)

        self.chunk_size = SpinBox()
        self.chunk_size.setRange(100, 2000)
        self.chunk_size.setValue(self.config.get("chunk_size", 500))
        self.chunk_size.valueChanged.connect(self._save_config)
        self.config_form.addRow("Chunk 大小:", self.chunk_size)

        self.overlap = SpinBox()
        self.overlap.setRange(0, 500)
        self.overlap.setValue(self.config.get("chunk_overlap", 50))
        self.overlap.valueChanged.connect(self._save_config)
        self.config_form.addRow("重叠大小:", self.overlap)

        self.model_combo = ComboBox()
        self.model_combo.addItems(["bge-large-zh", "m3e-base", "text-embedding-ada-002"])
        current_model = self.config.get("embedding_model", "bge-large-zh")
        index = self.model_combo.findText(current_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.config_form.addRow("Embedding 模型:", self.model_combo)

        self.rebuild_btn = PushButton(FluentIcon.SYNC, "重建向量索引")
        self.rebuild_btn.setFixedHeight(32)
        self.rebuild_btn.clicked.connect(self._rebuild_index)
        self.config_form.addRow("", self.rebuild_btn)

        self.index_status = BodyLabel("📊 索引状态: 加载中...")
        self.index_status.setStyleSheet("color: gray; font-size: 11px;")
        self.config_form.addRow("", self.index_status)

        config_inner.addLayout(self.config_form)
        layout.addWidget(config_card)

        self._initialized = True

    def update_doc(self, doc_name: str):
        if not doc_name or not getattr(self, '_initialized', False):
            return
        if not doc_name:
            self.name_label.setText("未选择文档")
            self.chunk_label.setText("-")
            self.updated_label.setText("-")
            return

        self.name_label.setText(doc_name)
        stats = self.doc_loader.get_document_stats()
        if doc_name in stats.get('documents', {}):
            doc_info = stats['documents'][doc_name]
            cat = doc_info.get('category', '') or self._infer_category_from_filename(doc_name)
            version = doc_info.get('version', '1.0')
            chunk_count = doc_info.get('chunk_count', 0)
            updated_at = doc_info.get('updated_at', '-')
            if updated_at != '-':
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
                    updated_at = dt.strftime('%Y-%m-%d %H:%M')
                except Exception:
                    pass
            else:
                updated_at = '-'

            idx = self.category_combo.findText(cat)
            if idx >= 0:
                self.category_combo.setCurrentIndex(idx)
            self.version_edit.setText(version)
            self.chunk_label.setText(str(chunk_count))
            self.updated_label.setText(updated_at)
        else:
            self.category_combo.setCurrentIndex(0)
            self.version_edit.setText("1.0")
            self.chunk_label.setText("0")
            self.updated_label.setText("-")

    def _on_category_changed(self, category: str):
        name = self.name_label.text()
        if name and name != "未选择文档":
            self.category_changed.emit(name, category)

    def _on_version_changed(self, version: str):
        name = self.name_label.text()
        if name and name != "未选择文档":
            self.version_changed.emit(name, version)

    @staticmethod
    def _infer_category_from_filename(filename: str) -> str:
        name = filename.replace('.md', '').replace('.Md', '')
        cat_map = {
            '售后政策': '售后政策', '物流配送': '物流配送', '促销活动': '促销活动',
            '商品咨询': '商品咨询', '支付问题': '支付问题', '服务咨询': '服务咨询',
            '订单咨询': '订单咨询',
        }
        for keyword, category in cat_map.items():
            if keyword in name:
                return category
        return '通用'

    def _save_config(self):
        self.config.set("chunk_size", self.chunk_size.value())
        self.config.set("chunk_overlap", self.overlap.value())

    def _on_model_changed(self, model: str):
        old_model = self.config.get("embedding_model", "bge-large-zh")
        if model == old_model:
            self._save_config()
            return

        from core.rag.vector_store import VectorStore
        vs = VectorStore()
        index_model = getattr(vs, "embedding_model", None)

        warn_parts = [
            f"当前索引基于模型: {index_model or old_model}",
            f"新选择的模型: {model}",
            "建议切换模型后立即重建向量索引。"
        ]
        box = MessageBox("切换 Embedding 模型", "\n".join(warn_parts), self)
        box.yesButton.setText("保存并重建索引")
        box.cancelButton.setText("仅保存")

        if box.exec():
            self.config.set("embedding_model", model)
            self._rebuild_index()
        else:
            self.config.set("embedding_model", model)
        self._update_index_status()

    def _rebuild_index(self):
        if self._rebuild_running:
            return
        self._rebuild_running = True
        self.rebuild_btn.setEnabled(False)
        self.rebuild_btn.setText("重建中...")
        self.index_status.setText("📊 索引状态: 重建中...")

        self._rebuild_thread = QThread()
        self._rebuild_worker = RebuildIndexWorker()
        self._rebuild_worker.moveToThread(self._rebuild_thread)
        self._rebuild_thread.started.connect(self._rebuild_worker.run)
        self._rebuild_worker.finished.connect(self._on_rebuild_finished)
        self._rebuild_worker.failed.connect(self._on_rebuild_failed)
        self._rebuild_worker.progress.connect(self._on_rebuild_progress)
        self._rebuild_worker.finished.connect(self._rebuild_thread.quit)
        self._rebuild_worker.failed.connect(self._rebuild_thread.quit)
        self._rebuild_thread.finished.connect(self._cleanup_rebuild_thread)
        self._rebuild_thread.start()

    def _on_rebuild_progress(self, stage: str, current: int, total: int):
        if total <= 0:
            total = 1
        self.index_status.setText(f"📊 索引状态: {stage} {max(0,min(current,total))}/{total}")

    def _on_rebuild_finished(self, success: bool, message: str):
        bar = InfoBar.success if success else InfoBar.error
        bar(title="重建成功" if success else "重建失败", content=message, parent=self, position=InfoBarPosition.TOP)
        if success:
            self.index_rebuilt.emit()
        self._finish_rebuild_ui()

    def _on_rebuild_failed(self, message: str):
        InfoBar.error(title="重建失败", content=message, parent=self, position=InfoBarPosition.TOP)
        self._finish_rebuild_ui()

    def _finish_rebuild_ui(self):
        self._rebuild_running = False
        self.rebuild_btn.setEnabled(True)
        self.rebuild_btn.setText("重建向量索引")
        self._update_index_status()

    def _cleanup_rebuild_thread(self):
        if self._rebuild_worker is not None:
            self._rebuild_worker.deleteLater()
            self._rebuild_worker = None
        if self._rebuild_thread is not None:
            self._rebuild_thread.deleteLater()
            self._rebuild_thread = None

    def _update_index_status(self):
        try:
            vs = get_vector_store()
            count = vs.count
            dim = getattr(vs, "dimension", None)
            built_model = getattr(vs, "embedding_model", None)
            current_model = self.config.get("embedding_model", "bge-large-zh")
            if count > 0 and built_model and built_model != current_model:
                self.index_status.setText(f"⚠️ 需要重建索引: {built_model} → {current_model}")
            elif dim:
                self.index_status.setText(f"📊 索引状态: {count} 条向量（维度 {dim}）")
            else:
                self.index_status.setText(f"📊 索引状态: {count} 条向量")
        except Exception:
            self.index_status.setText("📊 索引状态: 未初始化")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setStyleSheet("QFrame { border-left: 1px solid rgba(0,0,0,0.1); }")

        from core.config_utils.config import Config
        self.config = Config()
        self.doc_loader = get_doc_loader()

        self._rebuild_running = False
        self._rebuild_thread: QThread | None = None
        self._rebuild_worker: RebuildIndexWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        self.title = SubtitleLabel("⚙️ RAG 配置")
        layout.addWidget(self.title)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")

        self.content = QWidget()
        self.content_layout = QFormLayout(self.content)
        self.content_layout.setContentsMargins(12, 4, 12, 12)
        self.content_layout.setSpacing(10)
        self.content_layout.setFormAlignment(Qt.AlignTop)
        self.content_layout.setLabelAlignment(Qt.AlignVCenter)
        self.content_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)

        # Chunk 大小
        self.chunk_size = SpinBox()
        self.chunk_size.setRange(100, 2000)
        self.chunk_size.setValue(self.config.get("chunk_size", 500))
        self.chunk_size.valueChanged.connect(self._save_config)
        self.content_layout.addRow("Chunk 大小:", self.chunk_size)

        # 重叠大小
        self.overlap = SpinBox()
        self.overlap.setRange(0, 500)
        self.overlap.setValue(self.config.get("chunk_overlap", 50))
        self.overlap.valueChanged.connect(self._save_config)
        self.content_layout.addRow("重叠大小:", self.overlap)

        # Embedding 模型
        self.model_combo = ComboBox()
        self.model_combo.addItems(["bge-large-zh", "m3e-base", "text-embedding-ada-002"])
        current_model = self.config.get("embedding_model", "bge-large-zh")
        index = self.model_combo.findText(current_model)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        self.content_layout.addRow("Embedding 模型:", self.model_combo)

        # 重建索引按钮
        self.rebuild_btn = PushButton(FluentIcon.SYNC, "重建向量索引")
        self.rebuild_btn.setFixedHeight(32)
        self.rebuild_btn.clicked.connect(self._rebuild_index)
        self.content_layout.addRow("", self.rebuild_btn)

        # 索引状态
        self.index_status = BodyLabel("📊 索引状态: 加载中...")
        self.index_status.setStyleSheet("color: gray; font-size: 11px;")
        self.content_layout.addRow("", self.index_status)

        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)

    def _save_config(self):
        """保存配置"""
        self.config.set("chunk_size", self.chunk_size.value())
        self.config.set("chunk_overlap", self.overlap.value())
        InfoBar.success(
            title="配置已保存",
            content="RAG 配置已更新",
            parent=self,
            position=InfoBarPosition.TOP,
            duration=1500
        )

    def _on_model_changed(self, model: str):
        old_model = self.config.get("embedding_model", "bge-large-zh")
        if model == old_model:
            self._save_config()
            return

        from core.rag.vector_store import VectorStore
        vs = VectorStore()
        index_model = getattr(vs, "embedding_model", None)

        warn_parts = []
        warn_parts.append(f"当前索引基于模型: {index_model or old_model}")
        warn_parts.append(f"新选择的模型: {model}")
        warn_parts.append("建议切换模型后立即重建向量索引。")

        msg = "\n".join(warn_parts)

        box = MessageBox("切换 Embedding 模型", msg, self)
        box.yesButton.setText("保存并重建索引")
        box.cancelButton.setText("仅保存")

        if box.exec():
            self.config.set("embedding_model", model)
            self._rebuild_index()
        else:
            self.config.set("embedding_model", model)
        self._update_index_status()

    def _rebuild_index(self):
        """重建向量索引"""
        if self._rebuild_running:
            return

        self._rebuild_running = True
        self.rebuild_btn.setEnabled(False)
        self.rebuild_btn.setText("重建中...")
        self.index_status.setText("📊 索引状态: 重建中...")

        self._rebuild_thread = QThread()
        self._rebuild_worker = RebuildIndexWorker()
        self._rebuild_worker.moveToThread(self._rebuild_thread)

        self._rebuild_thread.started.connect(self._rebuild_worker.run)
        self._rebuild_worker.finished.connect(self._on_rebuild_finished)
        self._rebuild_worker.failed.connect(self._on_rebuild_failed)
        self._rebuild_worker.progress.connect(self._on_rebuild_progress)
        self._rebuild_worker.finished.connect(self._rebuild_thread.quit)
        self._rebuild_worker.failed.connect(self._rebuild_thread.quit)
        self._rebuild_thread.finished.connect(self._cleanup_rebuild_thread)

        self._rebuild_thread.start()

    def _on_rebuild_progress(self, stage: str, current: int, total: int):
        if total <= 0:
            total = 1
        current = max(0, min(current, total))
        self.index_status.setText(f"📊 索引状态: {stage} {current}/{total}")

    def _on_rebuild_finished(self, success: bool, message: str):
        if success:
            InfoBar.success(
                title="重建成功",
                content=message,
                parent=self,
                position=InfoBarPosition.TOP
            )
            self.index_rebuilt.emit()
        else:
            InfoBar.error(
                title="重建失败",
                content=message,
                parent=self,
                position=InfoBarPosition.TOP
            )
        self._finish_rebuild_ui()

    def _on_rebuild_failed(self, message: str):
        InfoBar.error(
            title="重建失败",
            content=message,
            parent=self,
            position=InfoBarPosition.TOP
        )
        self._finish_rebuild_ui()

    def _finish_rebuild_ui(self):
        self._rebuild_running = False
        self.rebuild_btn.setEnabled(True)
        self.rebuild_btn.setText("重建向量索引")
        self._update_index_status()

    def _cleanup_rebuild_thread(self):
        if self._rebuild_worker is not None:
            self._rebuild_worker.deleteLater()
            self._rebuild_worker = None
        if self._rebuild_thread is not None:
            self._rebuild_thread.deleteLater()
            self._rebuild_thread = None

    def _update_index_status(self):
        """更新索引状态"""
        try:
            vs = get_vector_store()
            count = vs.count
            dim = getattr(vs, "dimension", None)
            built_model = getattr(vs, "embedding_model", None)
            current_model = self.config.get("embedding_model", "bge-large-zh")
            if count > 0 and built_model and built_model != current_model:
                self.index_status.setText(f"⚠️ 需要重建索引: {built_model} → {current_model}")
            else:
                if dim:
                    self.index_status.setText(f"📊 索引状态: {count} 条向量（维度 {dim}）")
                else:
                    self.index_status.setText(f"📊 索引状态: {count} 条向量")
        except Exception:
            self.index_status.setText("📊 索引状态: 未初始化")


class PublishToolbar(QFrame):
    """底部工具栏"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setStyleSheet("""
            PublishToolbar {
                border: none;
                border-top: 1px solid rgba(0,0,0,0.1);
                background-color: #FAFAFA;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 10, 20, 10)
        layout.setSpacing(16)

        self.stats_label = BodyLabel("📚 知识库加载中...")
        self.stats_label.setStyleSheet("color: gray;")
        layout.addWidget(self.stats_label)

        layout.addStretch()

    def refresh_stats(self):
        """刷新统计"""
        try:
            loader = get_doc_loader()
            stats = loader.get_document_stats()
            total_docs = stats.get('total_documents', 0)
            total_chunks = stats.get('total_chunks', 0)
            self.stats_label.setText(f"📚 共 {total_docs} 篇文档，{total_chunks} 个 chunk")
        except Exception:
            self.stats_label.setText("📚 知识库统计加载中...")


class KnowledgeInterface(QWidget):
    """知识库管理界面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("knowledge_interface")
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        content = QWidget()
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # 左侧：分类面板
        self.tree_panel = KnowledgeTreePanel()
        content_layout.addWidget(self.tree_panel)

        # 中间：文档内容面板
        self.doc_list_panel = DocumentListPanel()
        content_layout.addWidget(self.doc_list_panel, 1)

        # 连接分类选择信号
        self.tree_panel.category_selected.connect(self.doc_list_panel._on_category_selected)

        # 右侧：文档信息 + RAG配置（上下分块）
        self.right_panel = RightPanel()
        self.right_panel.category_changed.connect(self._on_doc_category_changed)
        self.right_panel.version_changed.connect(self._on_doc_version_changed)
        self.doc_list_panel.doc_selected.connect(self.right_panel.update_doc)
        self.right_panel.index_rebuilt.connect(self._on_index_rebuilt)
        content_layout.addWidget(self.right_panel)

        layout.addWidget(content, 1)

        # 底部工具栏
        self.toolbar = PublishToolbar()
        layout.addWidget(self.toolbar)

        # 初始刷新统计
        self.toolbar.refresh_stats()

    def _on_index_rebuilt(self):
        """索引重建完成后刷新"""
        self.doc_list_panel._load_all()
        self.toolbar.refresh_stats()

    def _on_doc_category_changed(self, doc_name: str, new_category: str):
        """从文档信息面板更新分类"""
        from core.rag.doc_loader import get_doc_loader
        loader = get_doc_loader()
        stats = loader.get_document_stats()
        if doc_name in stats.get('documents', {}):
            loader._index[doc_name]['category'] = new_category
            loader._save_index()
        self.doc_list_panel._load_all(self.doc_list_panel.current_category)

    def _on_doc_version_changed(self, doc_name: str, new_version: str):
        """从文档信息面板更新版本号"""
        from core.rag.doc_loader import get_doc_loader
        loader = get_doc_loader()
        stats = loader.get_document_stats()
        if doc_name in stats.get('documents', {}):
            loader._index[doc_name]['version'] = new_version
            loader._save_index()
        self.doc_list_panel._load_all(self.doc_list_panel.current_category)
