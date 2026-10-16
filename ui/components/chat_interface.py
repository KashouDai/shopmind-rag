# -*- coding: utf-8 -*-
"""
客户问答界面 - 集成知识库和会话同步
"""

import re
import logging
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QScrollArea, QLabel, QStackedWidget, QSizePolicy, QApplication
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread, QObject, QSize
from PySide6.QtGui import QFontMetrics, QDesktopServices

from qfluentwidgets import (
    PushButton, PrimaryPushButton, TextEdit, LineEdit,
    CardWidget, BodyLabel, TitleLabel,
    SubtitleLabel, FluentIcon, TransparentToolButton,
    InfoBar, InfoBarPosition
)

from core.agents.conversation import ConversationManager, Conversation
from core.shared_data import KnowledgeStore
from core.statistics import StatisticsManager


logger = logging.getLogger(__name__)


class MarkdownRenderer:
    """简易 Markdown 渲染器，将 Markdown 转换为 HTML"""
    
    @staticmethod
    def render(text: str) -> str:
        """将 Markdown 文本转换为 HTML"""
        if not text:
            return ""
        
        # 转义 HTML 特殊字符（但保留我们要处理的 Markdown 语法）
        text = text.replace("&", "&amp;")
        text = text.replace("<", "&lt;")
        text = text.replace(">", "&gt;")
        
        # 处理代码块 ```code```
        def replace_code_block(match):
            lang = match.group(1) or ""
            code = match.group(2).strip()
            return f'<pre style="background-color: #f5f5f5; padding: 10px; border-radius: 6px; font-family: Consolas, Monaco, monospace; font-size: 13px; margin: 6px 0; white-space: pre-wrap; word-wrap: break-word;"><code>{code}</code></pre>'
        
        text = re.sub(r'```(\w*)\n(.*?)```', replace_code_block, text, flags=re.DOTALL)
        
        # 处理行内代码 `code`
        text = re.sub(
            r'`([^`]+)`',
            r'<code style="background-color: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-family: Consolas, Monaco, monospace; font-size: 13px;">\1</code>',
            text
        )
        
        # 处理粗体 **text** 或 __text__
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
        
        # 处理斜体 *text* 或 _text_（注意不要和粗体冲突）
        text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'(?<!_)_(?!_)(.+?)(?<!_)_(?!_)', r'<i>\1</i>', text)
        
        # 处理标题
        text = re.sub(r'^### (.+)$', r'<div style="font-weight: bold; font-size: 14px; margin: 6px 0;">\1</div>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'<div style="font-weight: bold; font-size: 15px; margin: 6px 0;">\1</div>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', r'<div style="font-weight: bold; font-size: 16px; margin: 6px 0;">\1</div>', text, flags=re.MULTILINE)
        
        # 处理无序列表
        text = re.sub(r'^[\-\*] (.+)$', r'• \1<br>', text, flags=re.MULTILINE)
        
        # 处理有序列表
        text = re.sub(r'^(\d+)\. (.+)$', r'\1. \2<br>', text, flags=re.MULTILINE)
        
        # 处理链接 [text](url)
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" style="color: #1976d2;">\1</a>', text)
        
        # 处理分隔线
        text = re.sub(r'^---+$', r'<hr style="border: none; border-top: 1px solid #e0e0e0; margin: 8px 0;">', text, flags=re.MULTILINE)
        
        # 处理换行
        text = text.replace('\n\n', '<br><br>')
        text = text.replace('\n', '<br>')
        
        # 清理多余的换行
        text = re.sub(r'(<br>){3,}', '<br><br>', text)
        
        return text


class ChatWorker(QObject):
    """聊天工作线程"""
    finished = Signal(str, str, float, dict)
    failed = Signal(str)

    def __init__(self, conv_id: str, text: str, history: list):
        super().__init__()
        self._conv_id = conv_id
        self._text = text
        self._history = history
        self._is_cancelled = False

    def cancel(self):
        """取消任务"""
        self._is_cancelled = True

    def run(self):
        if self._is_cancelled:
            return

        try:
            from core.agents.customer_agent import run_agent
            result = run_agent(
                self._text,
                history=self._history,
                thread_id=self._conv_id
            )
            if self._is_cancelled:
                return
            response = result.get("response", "")
            retrieval_score = result.get("retrieval_score", 0.0)
            intent = result.get("intent", "")
            need_human = result.get("need_human", False)
            routing_source = result.get("routing_source", "")
            score_type = result.get("score_type", "")
            answerability = result.get("answerability", "")
            answerability_reason = result.get("answerability_reason", "")
            trace = result.get("trace", {})
            rag_trace = {
                **trace,
                "query": self._text,
                "intent": intent,
                "need_human": need_human,
                "routing_source": routing_source,
                "retrieval_score": retrieval_score,
                "score_type": score_type,
                "answerability": answerability,
                "answerability_reason": answerability_reason,
                "trace": trace
            }
            self.finished.emit(self._conv_id, response, retrieval_score, rag_trace)
        except Exception as e:
            logger.error("Agent 执行失败: %s", e)
            if not self._is_cancelled:
                self.failed.emit(str(e))


class ConversationListItem(CardWidget):
    """对话列表项"""
    
    clicked = Signal(str)
    delete_clicked = Signal(str)
    
    def __init__(self, conversation: Conversation, parent=None):
        super().__init__(parent)
        self.conv_id = conversation.id
        self.title = conversation.title
        
        self.setFixedHeight(50)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumWidth(0)
        self.setCursor(Qt.PointingHandCursor)
        
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(12, 8, 8, 8)
        self._layout.setSpacing(8)
        
        # 对话图标 - 使用📝
        self.icon_label = BodyLabel("📝")
        self.icon_label.setFixedWidth(24)
        self._layout.addWidget(self.icon_label)
        
        # 标题
        self.title_label = BodyLabel(self.title)
        self.title_label.setStyleSheet("color: inherit;")
        self.title_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.title_label.setMinimumWidth(0)
        self._layout.addWidget(self.title_label, 1)
        
        # 删除按钮 - 使用BodyLabel作为按钮避免边框问题
        self.delete_btn = BodyLabel("✕")
        self.delete_btn.setFixedSize(24, 24)
        self.delete_btn.setAlignment(Qt.AlignCenter)
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.setStyleSheet("""
            QLabel {
                color: #999;
                font-size: 14px;
                border-radius: 4px;
            }
            QLabel:hover {
                color: #ff4d4f;
                background-color: rgba(255, 77, 79, 0.1);
            }
        """)
        self.delete_btn.setVisible(False)
        self.delete_btn.mousePressEvent = lambda e: self.delete_clicked.emit(self.conv_id)
        self._layout.addWidget(self.delete_btn)

        self._update_title_elide()

    def _update_title_elide(self):
        margins = self._layout.contentsMargins()
        available = self.width() - margins.left() - margins.right()
        available -= self.icon_label.width()
        available -= self._layout.spacing()
        if self.delete_btn.isVisible():
            available -= self.delete_btn.width()
            available -= self._layout.spacing()
        if available < 0:
            available = 0
        fm = QFontMetrics(self.title_label.font())
        self.title_label.setText(fm.elidedText(self.title, Qt.ElideRight, available))
    
    def enterEvent(self, event):
        self.delete_btn.setVisible(True)
        self._update_title_elide()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        self.delete_btn.setVisible(False)
        self._update_title_elide()
        super().leaveEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_title_elide()
    
    def mousePressEvent(self, event):
        # 不调用父类的 mousePressEvent，避免触发 CardWidget 的点击效果
        pass
    
    def mouseReleaseEvent(self, event):
        # 重写以阻止 CardWidget 发出无参数的 clicked 信号
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.conv_id)


class Sidebar(QFrame):
    """简化版侧边栏 - 只有新建对话和历史列表"""
    
    new_conversation = Signal()
    conversation_selected = Signal(str)
    conversation_deleted = Signal(str)
    transfer_to_human = Signal()  # 转人工信号
    admin_clicked = Signal()  # 管理员入口信号
    
    def _ensure_valid_font_point_size(self, widget: QWidget) -> None:
        font = widget.font()
        if font.pointSize() > 0:
            return
        base = QApplication.font()
        point_size = base.pointSize()
        if point_size <= 0:
            point_size = 10
        font.setPointSize(point_size)
        widget.setFont(font)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.conv_items = {}
        self.current_id = None
        
        self.setFixedWidth(260)
        self.setStyleSheet("""
            QFrame {
                background-color: transparent;
                border-right: 1px solid rgba(0,0,0,0.1);
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 20, 16, 20)
        layout.setSpacing(16)
        
        # 标题 - 使用机器人图标
        title = TitleLabel("🤖 shopmind-rag")
        layout.addWidget(title)
        
        # 新建对话按钮
        self.new_btn = PrimaryPushButton(FluentIcon.ADD, "新建对话")
        self.new_btn.setFixedHeight(40)
        self.new_btn.clicked.connect(self.new_conversation.emit)
        layout.addWidget(self.new_btn)
        
        # 转人工按钮
        self.transfer_btn = PushButton(FluentIcon.HEADPHONE, "转人工客服")
        self.transfer_btn.setFixedHeight(36)
        self.transfer_btn.clicked.connect(self.transfer_to_human.emit)
        layout.addWidget(self.transfer_btn)
        
        # 历史对话标签
        history_label = SubtitleLabel("历史对话")
        history_label.setStyleSheet("color: gray; margin-top: 8px;")
        layout.addWidget(history_label)
        
        # 历史对话列表区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        self.list_container = QWidget()
        self.list_container.setStyleSheet("background: transparent;")
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(6)
        self.list_layout.addStretch()
        
        self.scroll_area.setWidget(self.list_container)
        layout.addWidget(self.scroll_area, 1)
        
        # 管理员入口按钮（底部）
        self.admin_btn = TransparentToolButton(FluentIcon.PEOPLE)
        self.admin_btn.setFixedSize(36, 36)
        self.admin_btn.setIconSize(QSize(16, 16))
        self._ensure_valid_font_point_size(self.admin_btn)
        self.admin_btn.setToolTip("管理员入口")
        self.admin_btn.clicked.connect(self.admin_clicked.emit)
        
        admin_layout = QHBoxLayout()
        admin_layout.addWidget(self.admin_btn)
        admin_layout.addStretch()
        layout.addLayout(admin_layout)
    
    def set_conversations(self, conversations: list):
        """设置对话列表"""
        for item in self.conv_items.values():
            item.deleteLater()
        self.conv_items.clear()
        
        while self.list_layout.count() > 0:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for conv in conversations:
            self._add_item(conv)
        self.list_layout.addStretch()
    
    def add_conversation(self, conv: Conversation):
        """添加新对话"""
        stretch = self.list_layout.takeAt(self.list_layout.count() - 1)
        self._add_item(conv, insert_at=0)
        self.list_layout.addStretch()
        self._select(conv.id)
    
    def _add_item(self, conv: Conversation, insert_at=-1):
        """添加项"""
        item = ConversationListItem(conv)
        item.clicked.connect(self._on_clicked)
        item.delete_clicked.connect(self._on_delete)
        
        if insert_at >= 0:
            self.list_layout.insertWidget(insert_at, item)
        else:
            self.list_layout.addWidget(item)
        self.conv_items[conv.id] = item
    
    def _on_clicked(self, conv_id: str):
        self._select(conv_id)
        self.conversation_selected.emit(conv_id)
    
    def _on_delete(self, conv_id: str):
        if conv_id in self.conv_items:
            self.conv_items[conv_id].deleteLater()
            del self.conv_items[conv_id]
        self.conversation_deleted.emit(conv_id)
    
    def _select(self, conv_id: str):
        if self.current_id and self.current_id in self.conv_items:
            self.conv_items[self.current_id].setStyleSheet("")
        if conv_id in self.conv_items:
            self.conv_items[conv_id].setStyleSheet(
                "CardWidget { border: 2px solid #0078d4; }"
            )
        self.current_id = conv_id


class MessageBubble(QFrame):
    """消息气泡 - 支持 Markdown 渲染，宽度自适应"""

    def __init__(self, role: str, content: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.content = content
        self._content_label = None

        # 设置大小策略为根据内容调整
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        if role == "user":
            # 用户消息 - 浅蓝色气泡，纯文本
            self.setStyleSheet("""
                QFrame {
                    background-color: #bbdefb;
                    border-radius: 16px;
                    border: 1px solid #90caf9;
                }
                QLabel {
                    border: none;
                    background: transparent;
                }
            """)
            label = BodyLabel(content)
            label.setWordWrap(True)
            label.setStyleSheet("color: #000000; font-size: 14px; border: none; background: transparent;")
            layout.addWidget(label)
        else:
            # AI/人工客服消息 - 白色气泡，支持 Markdown
            self.setStyleSheet("""
                QFrame {
                    background-color: #ffffff;
                    border-radius: 16px;
                    border: 1px solid #e0e0e0;
                }
                QLabel {
                    border: none;
                    background: transparent;
                }
            """)

            # 根据消息内容判断是人工客服还是AI
            if "[人工客服]" in content:
                ai_label = BodyLabel("👨‍💼 人工客服")
                ai_label.setStyleSheet("color: #722ed1; font-weight: bold; font-size: 13px; border: none; background: transparent;")
            else:
                ai_label = BodyLabel("🤖 智能客服")
                ai_label.setStyleSheet("color: #1976d2; font-weight: bold; font-size: 13px; border: none; background: transparent;")
            layout.addWidget(ai_label)

            # 使用 QLabel 显示 Markdown 渲染后的 HTML
            content_label = QLabel()
            content_label.setWordWrap(True)
            content_label.setTextFormat(Qt.RichText)
            content_label.setOpenExternalLinks(True)
            content_label.setStyleSheet("""
                QLabel {
                    border: none;
                    background: transparent;
                    color: #424242;
                    font-size: 14px;
                    line-height: 1.5;
                }
            """)

            # 渲染 Markdown
            html_content = MarkdownRenderer.render(content)
            content_label.setText(html_content)
            self._content_label = content_label

            layout.addWidget(content_label)


class WelcomeWidget(QWidget):
    """欢迎界面"""
    
    question_clicked = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 60, 40, 40)
        layout.setSpacing(32)
        
        layout.addStretch(2)
        
        greeting = TitleLabel("有什么我能帮你的吗？")
        greeting.setAlignment(Qt.AlignCenter)
        greeting.setStyleSheet("font-size: 28px;")
        layout.addWidget(greeting)
        
        questions_widget = QWidget()
        q_layout = QVBoxLayout(questions_widget)
        q_layout.setSpacing(12)
        
        questions = [
            "如何申请退货退款？",
            "我的订单什么时候发货？",
            "物流信息在哪里查看？",
            "有什么优惠活动吗？",
            "商品尺码怎么选择？",
            "支持哪些支付方式？",
            "如何联系人工客服？",
            "商品质量有保障吗？",
        ]
        
        row = None
        for i, q in enumerate(questions):
            if i % 4 == 0:
                if row:
                    row.addStretch()
                    q_layout.addLayout(row)
                row = QHBoxLayout()
                row.setSpacing(10)
                row.addStretch()
            
            btn = PushButton(q)
            btn.clicked.connect(lambda checked, text=q: self.question_clicked.emit(text))
            row.addWidget(btn)
        
        if row:
            row.addStretch()
            q_layout.addLayout(row)
        
        layout.addWidget(questions_widget)
        layout.addStretch(3)


class ChatArea(QWidget):
    """聊天区域"""
    
    message_sent = Signal(str)
    question_clicked = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.stack = QStackedWidget()
        
        self.welcome = WelcomeWidget()
        self.welcome.question_clicked.connect(self.question_clicked.emit)
        self.stack.addWidget(self.welcome)
        
        self.chat_widget = QWidget()
        chat_layout = QVBoxLayout(self.chat_widget)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        
        self.msg_container = QWidget()
        self.msg_layout = QVBoxLayout(self.msg_container)
        self.msg_layout.setContentsMargins(40, 24, 40, 24)
        self.msg_layout.setSpacing(16)
        self.msg_layout.addStretch()
        
        self.scroll.setWidget(self.msg_container)
        chat_layout.addWidget(self.scroll)
        
        self.stack.addWidget(self.chat_widget)
        layout.addWidget(self.stack, 1)
        
        # 输入区
        input_card = CardWidget()
        input_layout = QHBoxLayout(input_card)
        input_layout.setContentsMargins(16, 8, 16, 8)
        input_layout.setSpacing(12)
        
        self.input_edit = LineEdit()
        self.input_edit.setPlaceholderText("输入您的问题...")
        self.input_edit.setFixedHeight(44)
        self.input_edit.setClearButtonEnabled(True)
        self.input_edit.returnPressed.connect(self._send)
        self.input_edit.setStyleSheet("""
            LineEdit {
                font-size: 14px;
                padding: 0 12px;
            }
        """)
        input_layout.addWidget(self.input_edit, 1)
        
        self.send_btn = PrimaryPushButton(FluentIcon.SEND, "发送")
        self.send_btn.setFixedSize(80, 38)
        self.send_btn.clicked.connect(self._send)
        input_layout.addWidget(self.send_btn)
        
        input_container = QWidget()
        ic_layout = QHBoxLayout(input_container)
        ic_layout.setContentsMargins(40, 16, 40, 24)
        ic_layout.addWidget(input_card)
        layout.addWidget(input_container)
    
    def show_welcome(self):
        self.stack.setCurrentIndex(0)
        self._clear_messages()
    
    def show_chat(self):
        self.stack.setCurrentIndex(1)
    
    def _clear_messages(self):
        while self.msg_layout.count() > 1:
            item = self.msg_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def add_message(self, role: str, content: str):
        if self.stack.currentIndex() == 0:
            self.show_chat()

        stretch = self.msg_layout.takeAt(self.msg_layout.count() - 1)

        row = QHBoxLayout()
        row.setSpacing(0)

        bubble = MessageBubble(role, content)
        # 设置最大宽度为聊天区域的70%，最小100px
        max_width = max(100, int(self.scroll.viewport().width() * 0.7))
        bubble.setMaximumWidth(max_width)

        if role == "user":
            row.addStretch()
            row.addWidget(bubble)
        else:
            row.addWidget(bubble)
            row.addStretch()

        container = QWidget()
        container.setLayout(row)
        self.msg_layout.addWidget(container)
        self.msg_layout.addStretch()

        QTimer.singleShot(50, self._scroll_bottom)

    def load_conversation(self, conv: Conversation):
        self._clear_messages()
        if conv.messages:
            for msg in conv.messages:
                self.add_message(msg.role, msg.content)
            self.show_chat()
        else:
            self.show_welcome()
    
    def _scroll_bottom(self):
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())
    
    def _send(self):
        text = self.input_edit.text().strip()
        if text:
            self.message_sent.emit(text)
            self.input_edit.clear()


class ChatInterface(QWidget):
    """客户问答界面 - 集成知识库检索"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chat_interface")
        
        self.conv_manager = ConversationManager()
        self.knowledge_store = KnowledgeStore()

        self._sending = False
        self._chat_thread: QThread | None = None
        self._chat_worker: ChatWorker | None = None

        self._current_conv_id: str | None = None
        self._rendered_message_count: int = 0
        self._human_refresh_timer = QTimer(self)
        
        self._init_ui()
        self._connect_signals()
        self._load_conversations()

        self._human_refresh_timer.timeout.connect(self._refresh_human_conversation)
        self._human_refresh_timer.start(1000)

        # 定时刷新对话列表（跨进程同步）
        self._history_refresh_timer = QTimer(self)
        self._history_refresh_timer.timeout.connect(self._refresh_history)
        self._history_refresh_timer.start(5000)  # 每5秒刷新一次
    
    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.sidebar = Sidebar()
        layout.addWidget(self.sidebar)
        
        self.chat_area = ChatArea()
        layout.addWidget(self.chat_area, 1)
    
    def _connect_signals(self):
        self.sidebar.new_conversation.connect(self._new_conv)
        self.sidebar.conversation_selected.connect(self._select_conv)
        self.sidebar.conversation_deleted.connect(self._delete_conv)
        self.sidebar.transfer_to_human.connect(self._transfer_to_human)
        self.chat_area.message_sent.connect(self._send_message)
        self.chat_area.question_clicked.connect(self._send_message)
    
    def _load_conversations(self):
        convs = self.conv_manager.get_all_conversations()
        self.sidebar.set_conversations(convs)

    def _refresh_history(self):
        """定时刷新历史对话列表，确保跨进程同步"""
        try:
            current_ids = set(self.sidebar.conv_items.keys())
            convs = self.conv_manager.get_all_conversations()
            db_ids = {c.id for c in convs}
            # 如果有新对话（来自其他进程），刷新列表
            if db_ids - current_ids:
                self._load_conversations()
                # 恢复当前选中的对话
                if self._current_conv_id and self._current_conv_id in db_ids:
                    self._select_conv(self._current_conv_id)
        except Exception:
            pass
    
    def _new_conv(self):
        conv = self.conv_manager.create_conversation()
        self.sidebar.add_conversation(conv)
        self.chat_area.show_welcome()
    
    def _select_conv(self, conv_id: str):
        conv = self.conv_manager.set_current_conversation(conv_id)
        if conv:
            self.chat_area.load_conversation(conv)
            self._current_conv_id = conv.id
            self._rendered_message_count = len(conv.messages)
    
    def _delete_conv(self, conv_id: str):
        self.conv_manager.delete_conversation(conv_id)
        self._load_conversations()
        if self.conv_manager.current_conversation is None:
            self.chat_area.show_welcome()
            self._current_conv_id = None
            self._rendered_message_count = 0
        elif self._current_conv_id == conv_id:
            self._current_conv_id = self.conv_manager.current_conversation.id
            self._rendered_message_count = len(self.conv_manager.current_conversation.messages)
            self.chat_area.load_conversation(self.conv_manager.current_conversation)
    
    def _transfer_to_human(self):
        """转人工客服"""
        conv = self.conv_manager.current_conversation
        if not conv:
            InfoBar.warning(
                title="无法转接",
                content="请先创建或选择一个对话",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return
        
        if not conv.messages:
            InfoBar.warning(
                title="无法转接",
                content="请至少发送一条消息后再转人工",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return
        
        # 标记对话为待人工处理
        conv.transfer_to_human()
        self.conv_manager.db.update_conversation_status(conv.id, status=conv.status, human_agent_id=conv.human_agent_id)
        
        # 添加系统消息
        self.conv_manager.add_message("assistant", "📞 您的对话已转接至人工客服，请稍候...")
        self.chat_area.add_message("assistant", "📞 您的对话已转接至人工客服，请稍候...")
        
        InfoBar.success(
            title="转接成功",
            content="已将对话转接至人工客服队列",
            parent=self,
            position=InfoBarPosition.TOP
        )
        self._current_conv_id = conv.id
        self._rendered_message_count = len(conv.messages)
    
    def _send_message(self, text: str):
        """发送消息"""
        if self._sending:
            return

        # 记录问题到统计
        try:
            stats_manager = StatisticsManager()
            stats_manager.record_question(text)
        except Exception:
            pass

        # 如果没有当前会话，自动创建
        if self.conv_manager.current_conversation is None:
            conv = self.conv_manager.create_conversation()
            self.sidebar.add_conversation(conv)

        # 检查是否处于人工服务状态
        conv = self.conv_manager.current_conversation
        if conv.status in [conv.STATUS_PENDING_HUMAN, conv.STATUS_HUMAN_HANDLING]:
            self.conv_manager.add_message("user", text)
            self.chat_area.add_message("user", text)
            self._load_conversations()
            self._current_conv_id = conv.id
            self._rendered_message_count = len(conv.messages)
            return

        conv_id = self.conv_manager.current_conversation.id

        # 添加用户消息（通过 manager 确保内存同步）
        self.conv_manager.add_message("user", text)
        conv = self.conv_manager.current_conversation
        self.chat_area.add_message("user", text)
        self._load_conversations()

        # 构建历史
        history = []
        for msg in conv.messages[:-1]:  # 不包含刚添加的用户消息
            history.append({"role": msg.role, "content": msg.content})

        # 创建工作线程
        self._chat_thread = QThread()
        self._chat_worker = ChatWorker(conv_id, text, history)
        self._chat_worker.moveToThread(self._chat_thread)

        self._chat_thread.started.connect(self._chat_worker.run)
        self._chat_worker.finished.connect(self._on_ai_finished)
        self._chat_worker.failed.connect(self._on_ai_failed)
        self._chat_worker.finished.connect(self._chat_thread.quit)
        self._chat_worker.failed.connect(self._chat_thread.quit)
        self._chat_thread.finished.connect(self._cleanup_chat_thread)

        self._set_sending(True)
        self._chat_thread.start()

    def _set_sending(self, sending: bool):
        self._sending = sending
        self.chat_area.input_edit.setEnabled(not sending)
        self.chat_area.send_btn.setEnabled(not sending)
        self.chat_area.input_edit.setPlaceholderText("输入您的问题...")

    def _on_ai_finished(self, conv_id: str, response: str, retrieval_score: float, rag_trace: dict = None):
        # 通过 manager 添加消息，确保内存同步
        # confidence 列保留为兼容字段，当前实际保存检索匹配分。
        compatible_score = round(float(retrieval_score or 0.0), 6)
        self.conv_manager.add_message(
            "assistant",
            response,
            confidence=compatible_score,
            rag_trace=rag_trace
        )
        # 清除缓存
        self.conv_manager.cache.invalidate_session(conv_id)
        self.conv_manager.cache.invalidate_session_list()

        self._load_conversations()
        if self.conv_manager.current_conversation:
            self._current_conv_id = self.conv_manager.current_conversation.id
            self._rendered_message_count = len(self.conv_manager.current_conversation.messages)
            self.chat_area.load_conversation(self.conv_manager.current_conversation)
        self._set_sending(False)

    def _on_ai_failed(self, message: str):
        InfoBar.error(
            title="发送失败",
            content=message,
            parent=self,
            position=InfoBarPosition.TOP
        )
        self._set_sending(False)

    def _cleanup_chat_thread(self):
        """清理聊天线程资源 - 确保正确释放内存"""
        if self._chat_worker is not None:
            try:
                self._chat_worker.cancel()  # 取消任务
                # 断开所有信号连接
                try:
                    self._chat_worker.finished.disconnect()
                except:
                    pass
                try:
                    self._chat_worker.failed.disconnect()
                except:
                    pass
                self._chat_worker.deleteLater()
            except:
                pass
            self._chat_worker = None
            
        if self._chat_thread is not None:
            try:
                if self._chat_thread.isRunning():
                    self._chat_thread.quit()
                    self._chat_thread.wait(1000)  # 等待最多1秒
                self._chat_thread.deleteLater()
            except:
                pass
            self._chat_thread = None

    def _refresh_human_conversation(self):
        if self._sending:
            return

        conv = self.conv_manager.current_conversation
        if not conv or not conv.id:
            return

        if conv.status not in [Conversation.STATUS_PENDING_HUMAN, Conversation.STATUS_HUMAN_HANDLING]:
            return

        current_id = conv.id
        if self._current_conv_id != current_id:
            self._current_conv_id = current_id
            self._rendered_message_count = len(conv.messages)
            # 检测到人工客服介入，提示用户
            if conv.status == Conversation.STATUS_HUMAN_HANDLING:
                InfoBar.info(
                    title="人工客服已接入",
                    content="人工客服正在为您服务，请描述您的问题...",
                    parent=self,
                    position=InfoBarPosition.TOP
                )
            return

        self.conv_manager._load_conversations()
        latest = self.conv_manager.get_conversation(current_id)
        if not latest:
            return

        # 检测状态变化
        if latest.status != conv.status:
            conv.status = latest.status
            if latest.status == Conversation.STATUS_HUMAN_HANDLING:
                InfoBar.info(
                    title="人工客服已接入",
                    content="人工客服正在为您服务，请描述您的问题...",
                    parent=self,
                    position=InfoBarPosition.TOP
                )
            elif latest.status == Conversation.STATUS_HUMAN_CLOSED:
                InfoBar.warning(
                    title="人工服务已结束",
                    content="人工客服已结束服务，您可以继续与AI对话",
                    parent=self,
                    position=InfoBarPosition.TOP
                )
                self._current_conv_id = None
                self._rendered_message_count = 0
                return

        new_count = len(latest.messages)
        if new_count == self._rendered_message_count:
            return

        if new_count < self._rendered_message_count:
            self.chat_area.load_conversation(latest)
            self._rendered_message_count = new_count
            self._current_conv_id = current_id
            return

        for msg in latest.messages[self._rendered_message_count:]:
            self.chat_area.add_message(msg.role, msg.content)

        self._rendered_message_count = new_count
        self._load_conversations()
