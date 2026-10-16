# -*- coding: utf-8 -*-
"""
客服AI工作台界面 - 完善版（会话按对话分组显示）
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame,
    QScrollArea, QLabel
)
from PySide6.QtCore import Qt, Signal, QTimer

from qfluentwidgets import (
    CardWidget, BodyLabel, TitleLabel, SubtitleLabel,
    PushButton, PrimaryPushButton, TransparentPushButton,
    ComboBox, InfoBar, InfoBarPosition, FluentIcon,
    MessageBox
)

from core.shared_data import KnowledgeStore
from core.agents.conversation import ConversationManager


class ConversationSessionItem(CardWidget):
    """对话会话列表项 - 显示整个对话"""
    
    item_clicked = Signal(object)  # 重命名避免与CardWidget的clicked冲突
    delete_clicked = Signal(str)
    
    def __init__(self, conv_data: dict, parent=None):
        super().__init__(parent)
        self.conv_data = conv_data
        self.conv_id = conv_data.get("id", "")
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(90)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        
        # 顶部：状态 + 时间 + 删除按钮
        top = QHBoxLayout()
        
        status = conv_data.get("status", "进行中")
        status_colors = {
            "进行中": "#52c41a",
            "待处理": "#faad14", 
            "转人工": "#722ed1",
            "已解决": "#1890ff"
        }
        status_label = BodyLabel(status)
        status_label.setStyleSheet(f"""
            background-color: {status_colors.get(status, '#999')};
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
        """)
        top.addWidget(status_label)
        
        # 消息数量
        msg_count = len(conv_data.get("messages", []))
        count_label = BodyLabel(f"💬 {msg_count//2}轮对话")
        count_label.setStyleSheet("color: gray; font-size: 11px;")
        top.addWidget(count_label)
        
        top.addStretch()
        
        # 删除按钮（已解决时显示）
        if status == "已解决":
            self.delete_btn = BodyLabel("🗑️")
            self.delete_btn.setFixedSize(24, 24)
            self.delete_btn.setAlignment(Qt.AlignCenter)
            self.delete_btn.setCursor(Qt.PointingHandCursor)
            self.delete_btn.setStyleSheet("""
                QLabel {
                    color: #999;
                    font-size: 14px;
                }
                QLabel:hover {
                    color: #ff4d4f;
                }
            """)
            self.delete_btn.setToolTip("删除会话")
            self.delete_btn.mousePressEvent = lambda e: self.delete_clicked.emit(self.conv_id)
            top.addWidget(self.delete_btn)
        
        time_label = BodyLabel(conv_data.get("timestamp", ""))
        time_label.setStyleSheet("color: gray; font-size: 11px;")
        top.addWidget(time_label)
        
        layout.addLayout(top)
        
        # 对话标题/第一条消息预览
        title = conv_data.get("title", "新对话")
        title_text = title[:25] + "..." if len(title) > 25 else title
        title_label = BodyLabel(f"📋 {title_text}")
        title_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(title_label)
        
        # 最近一次知识检索的匹配分（rerank 与 rrf 量纲不同，不统一换算百分比）
        retrieval_score = conv_data.get("retrieval_score")
        score_type = conv_data.get("score_type", "")
        if score_type:
            retrieval_text = f"最近检索分: {float(retrieval_score):.4f} ({score_type})"
        elif conv_data.get("retrieval_triggered"):
            retrieval_text = "最近检索分: 无候选资料"
        else:
            retrieval_text = "最近检索分: -"
        retrieval_label = BodyLabel(retrieval_text)
        retrieval_label.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(retrieval_label)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.item_clicked.emit(self.conv_data)
        super().mousePressEvent(event)


class SessionListPanel(QFrame):
    """会话列表面板 - 按对话分组"""
    
    session_selected = Signal(object)
    session_deleted = Signal(str)
    sessions_reloaded = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(300)
        self.setStyleSheet("QFrame { border-right: 1px solid rgba(0,0,0,0.1); }")
        
        self.conv_manager = ConversationManager()
        self._refresh_debounce = QTimer(self)
        self._refresh_debounce.setSingleShot(True)
        self._refresh_debounce.timeout.connect(self.refresh)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(12)
        
        title = SubtitleLabel("📊 会话列表")
        layout.addWidget(title)
        
        self.filter_combo = ComboBox()
        self.filter_combo.addItems(["全部", "进行中", "待处理", "转人工", "已解决"])
        self.filter_combo.currentTextChanged.connect(self._filter_changed)
        layout.addWidget(self.filter_combo)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        
        self.list_container = QWidget()
        self.list_layout = QVBoxLayout(self.list_container)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(8)
        self.list_layout.addStretch()
        
        self.scroll.setWidget(self.list_container)
        layout.addWidget(self.scroll, 1)

        self.refresh()

    def _load_sessions(self):
        """加载会话数据 - 从对话历史中获取"""
        self.session_data = []
        
        # 从对话管理器获取所有对话
        convs = self.conv_manager.get_all_conversations()
        
        for conv in convs:
            built = self._build_session_data(conv)
            if built:
                self.session_data.append(built)
        
        self._refresh_list()

    def _build_session_data(self, conv):
        if not getattr(conv, "messages", None):
            return None

        retrieval_score = None
        score_type = ""
        retrieval_triggered = False
        for msg in conv.messages:
            if msg.role != "assistant" or not isinstance(msg.rag_trace, dict):
                continue
            trace = msg.rag_trace
            if trace.get("intent") != "knowledge":
                continue
            retrieval_triggered = True
            retrieval_score = float(trace.get("retrieval_score", 0.0) or 0.0)
            score_type = str(trace.get("score_type", "") or "")

        from core.agents.conversation import Conversation as ConvClass
        if conv.status == ConvClass.STATUS_PENDING_HUMAN or conv.status == ConvClass.STATUS_HUMAN_HANDLING:
            status = "转人工"
        elif conv.status == ConvClass.STATUS_HUMAN_CLOSED:
            status = "已解决"
        else:
            status = "进行中"

        return {
            "id": conv.id,
            "title": conv.title,
            "messages": [m.to_dict() for m in conv.messages],
            "timestamp": conv.updated_at,
            "retrieval_score": retrieval_score,
            "score_type": score_type,
            "retrieval_triggered": retrieval_triggered,
            "status": status,
        }
    
    def _filter_changed(self, filter_text: str):
        self._refresh_list(filter_text)
    
    def _refresh_list(self, filter_text: str = "全部"):
        while self.list_layout.count() > 0:
            item = self.list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        for data in self.session_data:
            if filter_text == "全部" or data.get("status") == filter_text:
                item = ConversationSessionItem(data)
                item.item_clicked.connect(self._on_item_clicked)
                item.delete_clicked.connect(self._on_delete_clicked)
                self.list_layout.addWidget(item)
        
        self.list_layout.addStretch()
    
    def _on_item_clicked(self, conv_data: dict):
        self.session_selected.emit(conv_data)
    
    def _on_delete_clicked(self, conv_id: str):
        """删除会话"""
        w = MessageBox("确认删除", "确定要删除这个会话记录吗？", self)
        if w.exec():
            # 从列表中移除
            self.session_data = [s for s in self.session_data if s.get("id") != conv_id]
            self._refresh_list(self.filter_combo.currentText())
            self.session_deleted.emit(conv_id)
            InfoBar.success(
                title="删除成功",
                content="会话记录已删除",
                parent=self,
                position=InfoBarPosition.TOP
            )
    
    def _delete_session_status(self, conv_id: str):
        pass

    def update_session_status(self, conv_id: str, status: str):
        """更新会话状态（仅更新内存列表，PG中状态由Conversation.status管理）"""
        for data in self.session_data:
            if data.get("id") == conv_id:
                data["status"] = status
                break
        self._refresh_list(self.filter_combo.currentText())
    
    def refresh(self):
        """刷新列表"""
        self.conv_manager._load_conversations()
        self._load_sessions()
        self.sessions_reloaded.emit()

    def get_conv_data(self, conv_id: str) -> dict | None:
        self.conv_manager._load_conversations()
        conv = self.conv_manager.get_conversation(conv_id)
        return self._build_session_data(conv) if conv else None


class ConversationPanel(QFrame):
    """对话窗口面板 - 显示完整对话"""
    
    status_changed = Signal(str, str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_conv = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        self.title = SubtitleLabel("💬 对话详情")
        layout.addWidget(self.title)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("QScrollArea { border: 1px solid rgba(0,0,0,0.1); border-radius: 8px; }")
        
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(16, 16, 16, 16)
        self.content_layout.setSpacing(12)
        self.content_layout.addStretch()
        
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)
        
        # 操作按钮
        btn_layout = QHBoxLayout()
        
        self.resolve_btn = PrimaryPushButton(FluentIcon.ACCEPT, "标记解决")
        self.resolve_btn.clicked.connect(self._mark_resolved)
        btn_layout.addWidget(self.resolve_btn)
        
        self.transfer_btn = PushButton(FluentIcon.PEOPLE, "转人工")
        self.transfer_btn.clicked.connect(self._transfer_to_human)
        btn_layout.addWidget(self.transfer_btn)
        
        btn_layout.addStretch()
        
        self.like_btn = TransparentPushButton("👍")
        self.like_btn.setFixedSize(40, 40)
        self.like_btn.clicked.connect(lambda: self._set_feedback("👍"))
        btn_layout.addWidget(self.like_btn)
        
        self.dislike_btn = TransparentPushButton("👎")
        self.dislike_btn.setFixedSize(40, 40)
        self.dislike_btn.clicked.connect(lambda: self._set_feedback("👎"))
        btn_layout.addWidget(self.dislike_btn)
        
        layout.addLayout(btn_layout)
    
    def load_conversation(self, conv_data: dict):
        """加载完整对话"""
        self.current_conv = conv_data
        
        while self.content_layout.count() > 0:
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.title.setText(f"💬 {conv_data.get('title', '对话详情')}")
        
        messages = conv_data.get("messages", [])
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            card = CardWidget()
            if role == "user":
                card.setStyleSheet("CardWidget { background-color: #e6f7ff; border-radius: 8px; }")
                label_text = "👤 用户"
                label_color = "#1890ff"
            else:
                card.setStyleSheet("CardWidget { background-color: #f6ffed; border-radius: 8px; }")
                label_text = "🤖 AI客服"
                label_color = "#52c41a"
            
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            
            role_label = BodyLabel(label_text)
            role_label.setStyleSheet(f"font-weight: bold; color: {label_color};")
            card_layout.addWidget(role_label)
            
            content_label = BodyLabel(content)
            content_label.setWordWrap(True)
            card_layout.addWidget(content_label)
            
            self.content_layout.addWidget(card)
        
        self.content_layout.addStretch()
    
    def _mark_resolved(self):
        if self.current_conv:
            conv_id = self.current_conv.get("id")
            InfoBar.success(
                title="操作成功",
                content="已标记为解决",
                parent=self,
                position=InfoBarPosition.TOP
            )
            self.status_changed.emit(conv_id, "已解决")
    
    def _transfer_to_human(self):
        """转人工客服 - 实际转接功能"""
        if not self.current_conv:
            InfoBar.warning(
                title="无法转接",
                content="请先选择一个会话",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return
        
        conv_id = self.current_conv.get("id")
        
        # 获取对话并更新状态
        from core.agents.conversation import ConversationManager, Conversation
        conv_manager = ConversationManager()
        conv = conv_manager.get_conversation(conv_id)
        
        if conv:
            conv.transfer_to_human()
            conv.add_message("assistant", "📞 该对话已转接至人工客服，请稍候...")
            conv_manager.db.update_conversation_status(conv.id, status=conv.status, human_agent_id=conv.human_agent_id)
        
        InfoBar.success(
            title="转接成功",
            content="已转接人工客服",
            parent=self,
            position=InfoBarPosition.TOP
        )
        self.status_changed.emit(conv_id, "转人工")
    
    def _set_feedback(self, feedback: str):
        InfoBar.success(
            title="感谢反馈",
            content=f"已记录您的反馈 {feedback}",
            parent=self,
            position=InfoBarPosition.TOP
        )


class RAGTracePanel(QFrame):
    """RAG追溯面板 - 支持选择查看每条用户消息的追溯"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(320)
        self.setStyleSheet("QFrame { border-left: 1px solid rgba(0,0,0,0.1); }")
        
        self.knowledge_store = KnowledgeStore()
        self.current_messages = []  # 存储当前对话的用户消息
        self._current_trace_data = None
        self._current_query = ""
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        title = SubtitleLabel("🔍 RAG 追溯")
        layout.addWidget(title)
        
        # 消息选择下拉框
        self.msg_selector = ComboBox()
        self.msg_selector.setPlaceholderText("选择要追溯的问题...")
        self.msg_selector.currentIndexChanged.connect(self._on_message_selected)
        layout.addWidget(self.msg_selector)
        
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setStyleSheet("QScrollArea { border: none; }")
        
        self.content = QWidget()
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(12)
        
        self.scroll.setWidget(self.content)
        layout.addWidget(self.scroll, 1)
        
        self.edit_kb_btn = PrimaryPushButton(FluentIcon.EDIT, "一键改知识")
        self.edit_kb_btn.clicked.connect(self._edit_knowledge)
        layout.addWidget(self.edit_kb_btn)
    
    def load_trace(self, conv_data: dict):
        """加载对话的RAG追溯 - 填充消息选择器"""
        self._clear_content()
        self.msg_selector.clear()
        self.current_messages = []
        self._current_trace_data = None
        self._current_query = ""
        
        messages = conv_data.get("messages", [])
        if not messages:
            self._add_section("📭 无数据", "该对话无消息记录")
            self.content_layout.addStretch()
            return

        parsed: list[dict] = []
        pending: dict | None = None
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role")
            if role == "user":
                if pending:
                    parsed.append(pending)
                pending = {"query": msg.get("content", ""), "trace": None}
            elif role == "assistant":
                if pending and msg.get("rag_trace"):
                    pending["trace"] = msg.get("rag_trace")
                    parsed.append(pending)
                    pending = None

        if pending:
            parsed.append(pending)

        parsed = [p for p in parsed if (p.get("query") or "").strip()]
        if not parsed:
            self._add_section("📭 无数据", "该对话无可追溯的用户消息")
            self.content_layout.addStretch()
            return

        self.current_messages = parsed
        
        for i, item in enumerate(parsed):
            content = item.get("query", "")
            display_text = f"Q{i+1}: {content[:25]}..." if len(content) > 25 else f"Q{i+1}: {content}"
            self.msg_selector.addItem(display_text)
        
        # 默认选择第一条
        if parsed:
            self.msg_selector.setCurrentIndex(0)
            first = parsed[0]
            self._show_trace_for_query(first.get("query", ""), first.get("trace"))
    
    def _on_message_selected(self, index: int):
        """当选择消息改变时显示对应的追溯"""
        if 0 <= index < len(self.current_messages):
            item = self.current_messages[index]
            self._show_trace_for_query(item.get("query", ""), item.get("trace"))
    
    def _show_trace_for_query(self, query: str, trace: dict = None):
        """显示指定查询的RAG追溯"""
        self._clear_content()
        self._current_query = query or ""
        self._current_trace_data = trace
        
        if not query:
            self._add_section("📭 无数据", "查询内容为空")
            self.content_layout.addStretch()
            return

        trace_data = trace
        if not trace_data:
            self._add_section("📭 无追溯数据", "该消息未保存 rag_trace，无法保证追溯一致性")
            self.content_layout.addStretch()
            return

        if trace_data:
            # ① Query Rewrite
            raw_query = str(trace_data.get("query", query) or query)
            rewritten = str(trace_data.get("rewritten_query", "") or "")
            query_display = raw_query[:40] + "..." if len(raw_query) > 40 else raw_query
            rewrite_display = rewritten[:40] + "..." if len(rewritten) > 40 else rewritten
            self._add_section(
                "① Query Rewrite", 
                f"原始：{query_display}\n改写：{rewrite_display}"
            )
            
            # ② Retriever
            from core.config_utils.config import Config
            config = Config()
            top_k = config.get("retrieval_top_k", 5)
            threshold = config.get("similarity_threshold", 0.4)
            
            score_type = str(trace_data.get("score_type", "") or "")
            retriever_text = (
                f"TopK: {top_k}  重排阈值（仅 rerank）: {threshold}\n"
                f"分数来源: {score_type or '无'}\n\n"
            )
            
            retrieved_items = trace_data.get("retrieved_items") or []
            if retrieved_items:
                for item in retrieved_items[:5]:
                    score = float(item.get("score", 0.0)) if isinstance(item, dict) else 0.0
                    item_score_type = str(item.get("score_type", "") or score_type) if isinstance(item, dict) else score_type
                    if item_score_type == "rerank":
                        score_color = "🟢" if score >= 0.7 else ("🟡" if score >= 0.4 else "🔴")
                    else:
                        score_color = "⚪"
                    q = ""
                    if isinstance(item, dict):
                        q = str(item.get("question", ""))
                    question_preview = q[:20] + "..." if len(q) > 20 else q
                    retriever_text += (
                        f"{score_color} {score:.4f} "
                        f"({item_score_type or 'unknown'}) {question_preview}\n"
                    )
            else:
                retriever_text += "⚠️ 未找到匹配的知识"
            
            self._add_section("② Retriever", retriever_text.strip())
            
            # ③ Context
            context_text = str(trace_data.get("context_text", "") or "")
            if context_text:
                context_preview = context_text[:150] + "..." if len(context_text) > 150 else context_text
                self._add_section("③ Context（注入的上下文）", context_preview, highlight=True)
            else:
                self._add_section("③ Context", "⚠️ 无命中片段，未注入上下文")
            
            # ④ 检索与依据校验
            nested_trace = trace_data.get("trace", {})
            if not isinstance(nested_trace, dict):
                nested_trace = {}
            retrieval_score = float(
                trace_data.get("retrieval_score", nested_trace.get("retrieval_score", 0.0)) or 0.0
            )
            score_type = str(
                trace_data.get("score_type", nested_trace.get("score_type", "")) or ""
            )
            answerability = str(
                trace_data.get("answerability", nested_trace.get("answerability", "")) or ""
            )
            answerability_reason = str(
                trace_data.get("answerability_reason", nested_trace.get("answerability_reason", "")) or ""
            )

            answerability_text = {
                "supported": "有资料直接支撑",
                "partial": "部分有资料支撑",
                "none": "没有资料支撑，已转人工",
            }.get(answerability, "未执行或未记录")
            score_text = f"{retrieval_score:.4f} ({score_type})" if score_type else "无候选资料"
            check_text = (
                f"检索匹配分：{score_text}\n"
                f"回答依据：{answerability_text}"
            )
            if answerability_reason:
                check_text += f"\n判定说明：{answerability_reason}"
            if trace_data.get("need_human") or nested_trace.get("need_human"):
                check_text += "\n最终动作：转人工"
            self._add_section("④ 检索与依据校验", check_text)
            
            # ⑤ 最终提示词（可折叠，默认关闭）
            final_prompt = str(trace_data.get("final_prompt", "") or "")
            if final_prompt:
                self._add_collapsible_section(
                    "⑤ Final Prompt（发送给LLM）", 
                    final_prompt,
                    collapsed=True
                )
            else:
                self._add_section("⑤ Final Prompt", "⚠️ 未生成提示词")
        else:
            self._add_section("📭 无追溯数据", "未执行RAG检索")
        
        self.content_layout.addStretch()
    
    def _clear_content(self):
        """清空内容区域"""
        while self.content_layout.count() > 0:
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
    
    def _add_section(self, title: str, content: str, highlight: bool = False):
        card = CardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        
        title_label = BodyLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #1890ff;")
        layout.addWidget(title_label)
        
        content_label = BodyLabel(content)
        content_label.setWordWrap(True)
        if highlight:
            content_label.setStyleSheet("background-color: #fffbe6; padding: 8px; border-radius: 4px;")
        layout.addWidget(content_label)
        
        self.content_layout.addWidget(card)
    
    def _add_collapsible_section(self, title: str, content: str, collapsed: bool = True):
        """添加可折叠的区块"""
        card = CardWidget()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)
        
        # 标题行（可点击）
        header = QHBoxLayout()
        
        # 折叠图标
        arrow_label = BodyLabel("▶" if collapsed else "▼")
        arrow_label.setFixedWidth(20)
        header.addWidget(arrow_label)
        
        title_label = BodyLabel(title)
        title_label.setStyleSheet("font-weight: bold; color: #1890ff;")
        header.addWidget(title_label)
        header.addStretch()
        
        header_widget = QWidget()
        header_widget.setLayout(header)
        header_widget.setCursor(Qt.CursorShape.PointingHandCursor)
        layout.addWidget(header_widget)
        
        # 内容区域
        content_label = BodyLabel(content)
        content_label.setWordWrap(True)
        content_label.setStyleSheet("background-color: #f5f5f5; padding: 8px; border-radius: 4px;")
        content_label.setVisible(not collapsed)
        layout.addWidget(content_label)
        
        # 点击切换显示/隐藏
        def toggle_content():
            is_visible = content_label.isVisible()
            content_label.setVisible(not is_visible)
            arrow_label.setText("▼" if not is_visible else "▶")
        
        header_widget.mousePressEvent = lambda e: toggle_content()
        
        self.content_layout.addWidget(card)
    
    def _edit_knowledge(self):
        """一键编辑检索到的知识条目"""
        trace_data = self._current_trace_data
        if not isinstance(trace_data, dict):
            InfoBar.warning(
                title="无可编辑内容",
                content="请先选择一条用户消息进行RAG追溯",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return

        retrieved_items = trace_data.get("retrieved_items") or []
        retrieved_items = [x for x in retrieved_items if isinstance(x, dict) and x.get("id")]
        if not retrieved_items:
            InfoBar.warning(
                title="无可编辑内容",
                content="该条追溯没有检索到可编辑的知识条目",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return
        
        # 如果只有一条结果，直接编辑
        if len(retrieved_items) == 1:
            item_id = str(retrieved_items[0].get("id", "")).strip()
            item = self.knowledge_store.get_item_by_id(item_id)
            if not item:
                InfoBar.warning(
                    title="无法编辑",
                    content="对应知识条目不存在或已被删除",
                    parent=self,
                    position=InfoBarPosition.TOP
                )
                return
            self._open_edit_dialog(item)
            return
        
        # 多条结果，弹出选择对话框
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QDialogButtonBox
        
        dialog = QDialog(self.window())
        dialog.setWindowTitle("选择要编辑的知识条目")
        dialog.setMinimumWidth(500)
        dialog.setMinimumHeight(300)
        
        layout = QVBoxLayout(dialog)
        
        hint = BodyLabel("检索到以下知识条目，请选择要编辑的：")
        layout.addWidget(hint)
        
        list_widget = QListWidget()
        list_widget.setStyleSheet("QListWidget::item { padding: 8px; }")
        
        for item in retrieved_items[:20]:
            item_id = str(item.get("id", "")).strip()
            score = float(item.get("score", 0.0) or 0.0)
            score_type = str(item.get("score_type", "") or "")
            question = str(item.get("question", "") or "")
            question_preview = question[:50] + "..." if len(question) > 50 else question
            score_label = f"{score:.4f} ({score_type})" if score_type else f"{score:.4f}"
            list_item = QListWidgetItem(f"[{score_label}] {question_preview}")
            list_item.setData(Qt.ItemDataRole.UserRole, item_id)
            list_widget.addItem(list_item)
        
        list_widget.setCurrentRow(0)
        layout.addWidget(list_widget)
        
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        if dialog.exec() == QDialog.Accepted:
            selected = list_widget.currentItem()
            if selected:
                item_id = selected.data(Qt.ItemDataRole.UserRole)
                item = self.knowledge_store.get_item_by_id(item_id)
                if not item:
                    InfoBar.warning(
                        title="无法编辑",
                        content="对应知识条目不存在或已被删除",
                        parent=self,
                        position=InfoBarPosition.TOP
                    )
                    return
                self._open_edit_dialog(item)
    
    def _open_edit_dialog(self, knowledge_item):
        """打开知识编辑对话框"""
        from ui.pages.knowledge_interface import AddKnowledgeDialog
        
        dialog = AddKnowledgeDialog(self.window(), knowledge_item)
        result = dialog.exec()
        
        if result:
            data = dialog.get_data()
            try:
                self.knowledge_store.update_item(
                    knowledge_item.id,
                    question=data["question"],
                    answer=data["answer"],
                    keywords=data["keywords"],
                    category=data["category"]
                )
                index_error = getattr(self.knowledge_store, "last_vector_index_error", None)
                if isinstance(index_error, dict) and index_error.get("type") == "dimension_mismatch":
                    InfoBar.success(
                        title="编辑成功",
                        content="知识条目已更新（向量索引未更新）",
                        parent=self,
                        position=InfoBarPosition.TOP
                    )
                    InfoBar.warning(
                        title="需要重建索引",
                        content="检测到Embedding维度变化，请在知识库页面点击“重建向量索引”",
                        parent=self,
                        position=InfoBarPosition.TOP
                    )
                else:
                    InfoBar.success(
                        title="编辑成功",
                        content=f"知识条目已更新",
                        parent=self,
                        position=InfoBarPosition.TOP
                    )
                
                # 刷新当前追溯显示
                if self.current_messages and self.msg_selector.currentIndex() >= 0:
                    item = self.current_messages[self.msg_selector.currentIndex()]
                    query = item.get("query", "")
                    trace = item.get("trace")
                    if query:
                        self._show_trace_for_query(query, trace)
                        
            except Exception as e:
                InfoBar.error(
                    title="编辑失败",
                    content=str(e),
                    parent=self,
                    position=InfoBarPosition.TOP
                )


class WorkbenchInterface(QWidget):
    """客服AI工作台界面"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("workbench_interface")
        self._current_conv_id: str | None = None
        self._init_ui()
    
    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        self.session_panel = SessionListPanel()
        self.session_panel.session_selected.connect(self._on_session_selected)
        self.session_panel.sessions_reloaded.connect(self._on_sessions_reloaded)
        layout.addWidget(self.session_panel)
        
        self.conv_panel = ConversationPanel()
        self.conv_panel.status_changed.connect(self._on_status_changed)
        layout.addWidget(self.conv_panel, 1)
        
        self.trace_panel = RAGTracePanel()
        layout.addWidget(self.trace_panel)
    
    def _on_session_selected(self, conv_data: dict):
        self._current_conv_id = conv_data.get("id")
        self.conv_panel.load_conversation(conv_data)
        self.trace_panel.load_trace(conv_data)
    
    def _on_sessions_reloaded(self):
        if not self._current_conv_id:
            return
        latest = self.session_panel.get_conv_data(self._current_conv_id)
        if not latest:
            self._current_conv_id = None
            return
        self.conv_panel.load_conversation(latest)
        self.trace_panel.load_trace(latest)

    def _on_status_changed(self, conv_id: str, new_status: str):
        self.session_panel.update_session_status(conv_id, new_status)
    
    def showEvent(self, event):
        """显示时刷新数据"""
        super().showEvent(event)
        self.session_panel.refresh()
