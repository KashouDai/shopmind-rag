# -*- coding: utf-8 -*-
"""
设置对话框模块 - 使用 Fluent Widgets
"""

from PySide6.QtWidgets import (
    QVBoxLayout, QHBoxLayout, QWidget, QFormLayout
)
from PySide6.QtCore import Qt, Signal

import os

from qfluentwidgets import (
    MessageBoxBase, SubtitleLabel, BodyLabel,
    SpinBox, LineEdit, ComboBox, Slider,
    PrimaryPushButton, PushButton, PasswordLineEdit,
    CardWidget, InfoBar, InfoBarPosition
)

from core.config_utils.config import Config
from core.agents.model_router import get_model_router
from core.infra.llm_providers import get_all_providers


class SettingsDialog(MessageBoxBase):
    """设置对话框"""
    
    settings_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = Config()
        
        self.titleLabel = SubtitleLabel("设置")
        self.viewLayout.addWidget(self.titleLabel)
        
        self._init_ui()
        self._load_settings()
        
        self.yesButton.setText("保存")
        self.cancelButton.setText("取消")
        
        self.yesButton.clicked.connect(self._save_settings)
        
        self.widget.setMinimumWidth(500)
    
    def _init_ui(self):
        """初始化UI"""
        # API设置
        api_card = CardWidget()
        api_layout = QVBoxLayout(api_card)
        api_layout.setContentsMargins(16, 16, 16, 16)
        api_layout.setSpacing(12)
        
        api_title = BodyLabel("🔗 API配置")
        api_title.setStyleSheet("font-weight: bold;")
        api_layout.addWidget(api_title)
        
        # LLM提供商选择
        provider_row = QHBoxLayout()
        provider_row.addWidget(BodyLabel("服务商:"))
        self.provider_combo = ComboBox()
        self._providers = get_all_providers()
        for p in self._providers:
            self.provider_combo.addItem(p["display_name"], p["name"])
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        provider_row.addWidget(self.provider_combo)
        provider_row.addStretch()
        api_layout.addLayout(provider_row)
        
        # API地址
        url_row = QHBoxLayout()
        url_row.addWidget(BodyLabel("API地址:"))
        self.api_url = LineEdit()
        self.api_url.setPlaceholderText("使用默认地址")
        url_row.addWidget(self.api_url, 1)
        api_layout.addLayout(url_row)
        
        # API密钥
        key_row = QHBoxLayout()
        key_row.addWidget(BodyLabel("API密钥:"))
        self.api_key = PasswordLineEdit()
        self.api_key.setPlaceholderText("sk-...")
        key_row.addWidget(self.api_key, 1)
        api_layout.addLayout(key_row)
        
        # 模型选择
        model_row = QHBoxLayout()
        model_row.addWidget(BodyLabel("模型:"))
        self.model_combo = ComboBox()
        model_row.addWidget(self.model_combo, 1)
        api_layout.addLayout(model_row)
        
        # 自定义模型名称
        custom_model_row = QHBoxLayout()
        custom_model_row.addWidget(BodyLabel("自定义模型:"))
        self.custom_model_input = LineEdit()
        self.custom_model_input.setPlaceholderText("留空则使用上方选择的模型")
        custom_model_row.addWidget(self.custom_model_input, 1)
        api_layout.addLayout(custom_model_row)
        
        # 测试连接按钮
        test_row = QHBoxLayout()
        test_row.addStretch()
        self.test_btn = PushButton("测试连接")
        self.test_btn.clicked.connect(self._test_connection)
        test_row.addWidget(self.test_btn)
        api_layout.addLayout(test_row)
        
        self.viewLayout.addWidget(api_card)
        
        # 高级设置
        adv_card = CardWidget()
        adv_layout = QVBoxLayout(adv_card)
        adv_layout.setContentsMargins(16, 16, 16, 16)
        adv_layout.setSpacing(12)
        
        adv_title = BodyLabel("⚙️ 高级设置")
        adv_title.setStyleSheet("font-weight: bold;")
        adv_layout.addWidget(adv_title)
        
        # 最大令牌
        tokens_row = QHBoxLayout()
        tokens_row.addWidget(BodyLabel("最大令牌:"))
        self.max_tokens = SpinBox()
        self.max_tokens.setRange(256, 8192)
        self.max_tokens.setValue(2048)
        self.max_tokens.setSingleStep(256)
        tokens_row.addWidget(self.max_tokens)
        tokens_row.addStretch()
        adv_layout.addLayout(tokens_row)
        
        # 温度
        temp_row = QHBoxLayout()
        temp_row.addWidget(BodyLabel("温度:"))
        self.temp_slider = Slider(Qt.Horizontal)
        self.temp_slider.setRange(0, 100)
        self.temp_slider.setValue(70)
        self.temp_slider.setFixedWidth(150)
        temp_row.addWidget(self.temp_slider)
        self.temp_label = BodyLabel("0.7")
        self.temp_label.setFixedWidth(30)
        self.temp_slider.valueChanged.connect(
            lambda v: self.temp_label.setText(f"{v/100:.1f}")
        )
        temp_row.addWidget(self.temp_label)
        temp_row.addStretch()
        adv_layout.addLayout(temp_row)
        
        self.viewLayout.addWidget(adv_card)
    
    def _on_provider_changed(self, index: int):
        """提供商切换时更新模型列表和默认API地址"""
        if index < 0 or index >= len(self._providers):
            return
        
        provider = self._providers[index]
        
        # 更新API地址占位符
        self.api_url.setPlaceholderText(provider["default_api_url"])
        
        # 更新模型列表
        self.model_combo.clear()
        self.model_combo.addItems(provider["supported_models"])
        
        # 设置默认模型
        default_model = provider["default_model"]
        idx = self.model_combo.findText(default_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)
    
    def _load_settings(self):
        """加载设置"""
        # 加载提供商
        current_provider = self.config.get("llm_provider", "siliconflow")
        for i, p in enumerate(self._providers):
            if p["name"] == current_provider:
                self.provider_combo.setCurrentIndex(i)
                break
        
        # 触发一次提供商变更以更新模型列表
        self._on_provider_changed(self.provider_combo.currentIndex())
        
        # 加载API地址
        self.api_url.setText(self.config.get("api_base_url", "", include_env=False))
        
        # 加载API密钥状态
        configured_key = bool(self.config.get("api_key", "", include_env=True))
        self.api_key.setText("")
        self.api_key.setPlaceholderText("已配置（不显示）" if configured_key else "sk-...")
        
        # 加载模型
        model_name = self.config.get("model_name", "")
        if model_name:
            idx = self.model_combo.findText(model_name)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
                self.custom_model_input.setText("")
            else:
                # 模型不在列表中，放到自定义输入框
                self.custom_model_input.setText(model_name)
        
        # 加载高级设置
        self.max_tokens.setValue(self.config.get("max_tokens", 2048))
        temp = int(self.config.get("temperature", 0.7) * 100)
        self.temp_slider.setValue(temp)
        self.temp_label.setText(f"{temp/100:.1f}")
    
    def _save_settings(self):
        """保存设置"""
        # 获取选中的提供商
        provider_index = self.provider_combo.currentIndex()
        provider_name = self._providers[provider_index]["name"] if provider_index >= 0 else "siliconflow"
        
        # 优先使用自定义模型名称
        custom_model = self.custom_model_input.text().strip()
        model_name = custom_model if custom_model else self.model_combo.currentText().strip()
        
        settings: dict = {
            "llm_provider": provider_name,
            "api_base_url": self.api_url.text().strip(),
            "model_name": model_name,
            "max_tokens": self.max_tokens.value(),
            "temperature": self.temp_slider.value() / 100
        }

        api_key = self.api_key.text().strip()
        if api_key:
            self.config.set("api_key", api_key)
            self.api_key.setText("")
            self.api_key.setPlaceholderText("已配置（已保存）")
        
        self.config.update(settings)
        
        # 重新加载模型路由
        get_model_router().reload()
        
        self.settings_changed.emit()
    
    def _test_connection(self):
        """测试API连接"""
        # 临时应用当前设置进行测试
        provider_index = self.provider_combo.currentIndex()
        if provider_index < 0:
            InfoBar.error(
                title="测试失败",
                content="请选择服务商",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return
        
        provider_name = self._providers[provider_index]["name"]
        api_key = self.api_key.text().strip() or self.config.get("api_key", "")
        api_url = self.api_url.text().strip() or None
        
        # 优先使用自定义模型
        custom_model = self.custom_model_input.text().strip()
        model = custom_model if custom_model else self.model_combo.currentText().strip() or None
        
        if not api_key:
            InfoBar.error(
                title="测试失败",
                content="请输入API密钥",
                parent=self,
                position=InfoBarPosition.TOP
            )
            return
        
        self.test_btn.setEnabled(False)
        self.test_btn.setText("测试中...")
        
        try:
            # 创建临时提供商进行测试
            from core.infra.llm_providers import get_provider
            provider_class = get_provider(provider_name)
            provider = provider_class(api_key=api_key, api_url=api_url, model=model)
            success, message = provider.test_connection()
            
            if success:
                InfoBar.success(
                    title="连接成功",
                    content=message,
                    parent=self,
                    position=InfoBarPosition.TOP
                )
            else:
                InfoBar.error(
                    title="连接失败",
                    content=message,
                    parent=self,
                    position=InfoBarPosition.TOP
                )
        except Exception as e:
            InfoBar.error(
                title="测试失败",
                content=str(e),
                parent=self,
                position=InfoBarPosition.TOP
            )
        finally:
            self.test_btn.setEnabled(True)
            self.test_btn.setText("测试连接")
