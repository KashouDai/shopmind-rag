# -*- coding: utf-8 -*-
"""
登录对话框 - 管理后台身份验证
"""

import os
import json
import base64
import secrets
import time
import logging

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QStackedWidget
)
from PySide6.QtCore import Qt, Signal

from qfluentwidgets import (
    LineEdit, PasswordLineEdit, PrimaryPushButton, PushButton,
    TitleLabel, SubtitleLabel, BodyLabel, CardWidget,
    InfoBar, InfoBarPosition, FluentIcon
)

# 导入密码哈希库
try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError as e:
    CRYPTO_AVAILABLE = False
    logging.warning(f"cryptography库导入失败: {e}")

logger = logging.getLogger(__name__)


class AuthManager:
    """用户认证管理器"""
    
    def __init__(self):
        self._users_file = self._get_users_file()
        self._failed_attempts = {}
        self._lock_until = {}
        self._ensure_default_admin()
    
    def _get_users_file(self):
        """获取用户数据文件路径"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(base_dir, "data", "users.json")
    
    def _ensure_default_admin(self):
        """确保默认管理员账户存在"""
        if not os.path.exists(self._users_file):
            os.makedirs(os.path.dirname(self._users_file), exist_ok=True)
            # 创建默认管理员 admin/admin123
            default_users = {
                "admin": {
                    "password": self._make_password_record("admin123"),
                    "role": "admin",
                    "name": "管理员",
                    "must_change_password": True
                }
            }
            with open(self._users_file, 'w', encoding='utf-8') as f:
                json.dump(default_users, f, ensure_ascii=False, indent=2)

    def _make_password_record(self, password: str, iterations: int = 150_000) -> dict:
        """使用PBKDF2创建密码记录"""
        if not CRYPTO_AVAILABLE:
            raise RuntimeError("cryptography库未安装或导入失败，无法创建安全密码。请运行: pip install cryptography")
        
        try:
            salt = secrets.token_bytes(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=int(iterations),
            )
            digest = kdf.derive(password.encode("utf-8"))
            
            return {
                "algo": "pbkdf2_sha256",
                "salt": base64.b64encode(salt).decode("ascii"),
                "iterations": int(iterations),
                "hash": base64.b64encode(digest).decode("ascii"),
            }
        except Exception as e:
            raise RuntimeError(f"密码加密失败: {e}")

    def _verify_password_record(self, password: str, record: dict) -> bool:
        """验证PBKDF2密码"""
        if not CRYPTO_AVAILABLE:
            return False
        
        if not isinstance(record, dict):
            return False
        algo = record.get("algo")
        if algo != "pbkdf2_sha256":
            return False

        try:
            salt = base64.b64decode(record.get("salt", ""), validate=True)
            expected = base64.b64decode(record.get("hash", ""), validate=True)
            iterations = int(record.get("iterations", 0))
        except Exception:
            return False

        if not salt or not expected or iterations <= 0:
            return False

        try:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=iterations,
            )
            got = kdf.derive(password.encode("utf-8"))
            return secrets.compare_digest(got, expected)
        except Exception:
            return False

    def _is_weak_password(self, username: str, password: str) -> bool:
        pwd = (password or "").strip()
        if len(pwd) < 8:
            return True
        if username and pwd.lower() == username.lower():
            return True
        weak_list = {
            "admin123",
            "123456",
            "password",
            "qwerty",
            "000000",
            "111111",
            "abcdefg",
        }
        if pwd.lower() in weak_list:
            return True
        if pwd.isdigit() and len(pwd) <= 10:
            return True
        return False

    def _is_locked(self, username: str) -> tuple:
        until = self._lock_until.get(username)
        now = time.time()
        if until is None or now >= until:
            if until is not None:
                self._lock_until.pop(username, None)
            return False, 0
        return True, int(until - now)

    def _mark_failed(self, username: str):
        now = time.time()
        items = self._failed_attempts.get(username, [])
        items = [t for t in items if now - t <= 300]
        items.append(now)
        self._failed_attempts[username] = items
        if len(items) >= 5:
            self._lock_until[username] = now + 300

    def _clear_failed(self, username: str):
        self._failed_attempts.pop(username, None)
        self._lock_until.pop(username, None)
    
    def _load_users(self) -> dict:
        """加载用户数据"""
        try:
            with open(self._users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    
    def _save_users(self, users: dict):
        """保存用户数据"""
        os.makedirs(os.path.dirname(self._users_file), exist_ok=True)
        with open(self._users_file, 'w', encoding='utf-8') as f:
            json.dump(users, f, ensure_ascii=False, indent=2)
    
    def login(self, username: str, password: str) -> tuple:
        """登录验证
        Returns: (success: bool, message: str, require_password_change: bool, role: str)
        """
        if not username or not password:
            return False, "请输入用户名和密码", False, ""

        locked, seconds = self._is_locked(username)
        if locked:
            return False, f"登录失败次数过多，请 {seconds}s 后重试", False, ""
        
        users = self._load_users()
        if username not in users:
            self._mark_failed(username)
            return False, "用户名不存在", False, ""

        user = users.get(username, {})
        stored_record = user.get("password")

        # 只支持PBKDF2，不再兼容SHA256
        if not stored_record or not isinstance(stored_record, dict):
            self._mark_failed(username)
            logger.warning(f"用户 {username} 使用过时的密码格式，需要重置密码")
            return False, "密码格式过时，请联系管理员重置密码", False, ""

        ok = self._verify_password_record(password, stored_record)

        if not ok:
            self._mark_failed(username)
            return False, "密码错误", False, ""

        self._clear_failed(username)

        require_change = bool(user.get("must_change_password")) or self._is_weak_password(username, password)
        if username == "admin" and password == "admin123":
            require_change = True

        role = user.get("role", "cs")  # 默认为客服角色
        return True, user.get("name", username), require_change, role
    
    def register(self, username: str, password: str, confirm_password: str, role: str = "cs") -> tuple:
        """提交注册申请
        Returns: (success: bool, message: str)
        """
        if not username or not password:
            return False, "请填写完整信息"
        
        if len(username) < 3:
            return False, "用户名至少3个字符"
        
        if self._is_weak_password(username, password):
            return False, "密码过弱，请设置更复杂的密码（至少8位，避免常见密码）"
        
        if password != confirm_password:
            return False, "两次密码不一致"
        
        users = self._load_users()
        if username in users:
            return False, "用户名已存在"
        
        # 检查是否已有待审核申请
        pending_file = os.path.join(os.path.dirname(self._users_file), "pending_registrations.json")
        try:
            with open(pending_file, 'r', encoding='utf-8') as f:
                pending = json.load(f)
        except:
            pending = {}
        
        if username in pending:
            return False, "该用户名已有待审核的注册申请"
        
        # 添加到待审核列表
        from datetime import datetime
        pending[username] = {
            "password": self._make_password_record(password),
            "role": role,
            "name": username,
            "apply_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(pending_file, 'w', encoding='utf-8') as f:
            json.dump(pending, f, ensure_ascii=False, indent=2)
        
        return True, "注册申请已提交，请等待管理员审核"

    def change_password(self, username: str, old_password: str, new_password: str, confirm_password: str) -> tuple:
        if not username:
            return False, "用户名无效"
        if not old_password or not new_password:
            return False, "请填写完整信息"
        if new_password != confirm_password:
            return False, "两次新密码不一致"
        if self._is_weak_password(username, new_password):
            return False, "新密码过弱，请设置更复杂的密码"

        users = self._load_users()
        if username not in users:
            return False, "用户不存在"

        user = users.get(username, {})

        # 只验证PBKDF2格式
        record = user.get("password")
        if not record or not isinstance(record, dict):
            return False, "密码格式过时，请联系管理员重置"

        ok = self._verify_password_record(old_password, record)

        if not ok:
            return False, "原密码不正确"

        user["password"] = self._make_password_record(new_password)
        user.pop("password_hash", None)  # 清理可能残留的旧字段
        user["must_change_password"] = False
        users[username] = user
        self._save_users(users)
        return True, "密码修改成功"


class ChangePasswordDialog(QDialog):
    def __init__(self, username: str, auth: AuthManager, parent=None):
        super().__init__(parent)
        self._username = username
        self._auth = auth

        self.setWindowTitle("修改密码")
        self.setFixedSize(420, 320)
        self.setWindowFlags(Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowTitleHint)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)

        title = SubtitleLabel("为保障安全，请修改密码")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        user_label = BodyLabel(f"账号：{username}")
        user_label.setAlignment(Qt.AlignCenter)
        user_label.setStyleSheet("color: gray;")
        layout.addWidget(user_label)

        self.old_pwd = PasswordLineEdit()
        self.old_pwd.setPlaceholderText("原密码")
        self.old_pwd.setFixedHeight(44)
        layout.addWidget(self.old_pwd)

        self.new_pwd = PasswordLineEdit()
        self.new_pwd.setPlaceholderText("新密码（至少8位，避免常见密码）")
        self.new_pwd.setFixedHeight(44)
        layout.addWidget(self.new_pwd)

        self.new_pwd2 = PasswordLineEdit()
        self.new_pwd2.setPlaceholderText("确认新密码")
        self.new_pwd2.setFixedHeight(44)
        self.new_pwd2.returnPressed.connect(self._submit)
        layout.addWidget(self.new_pwd2)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.submit_btn = PrimaryPushButton("确 认 修 改")
        self.submit_btn.setFixedHeight(40)
        self.submit_btn.clicked.connect(self._submit)
        btn_row.addWidget(self.submit_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def _submit(self):
        ok, msg = self._auth.change_password(
            self._username,
            self.old_pwd.text(),
            self.new_pwd.text(),
            self.new_pwd2.text(),
        )
        if ok:
            InfoBar.success(
                title="修改成功",
                content=msg,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=1500,
            )
            self.accept()
        else:
            InfoBar.error(
                title="修改失败",
                content=msg,
                parent=self,
                position=InfoBarPosition.TOP,
            )


class LoginDialog(QDialog):
    """登录/注册对话框"""
    
    login_success = Signal(str, str)  # 登录成功信号，传递用户名和角色
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.auth = AuthManager()
        self._login_in_progress = False
        self.setWindowTitle("管理后台登录")
        self.setFixedSize(400, 480)
        # 保留关闭按钮，移除帮助按钮
        self.setWindowFlags(
            Qt.Dialog | Qt.WindowCloseButtonHint | Qt.WindowTitleHint
        )
        
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 32, 32, 32)
        layout.setSpacing(20)
        
        # 标题
        title = TitleLabel("🔐 shopmind-rag")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        subtitle = SubtitleLabel("管理后台")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: gray;")
        layout.addWidget(subtitle)
        
        layout.addSpacing(10)
        
        # 登录/注册切换
        self.stack = QStackedWidget()
        
        # 登录页面
        self.login_page = self._create_login_page()
        self.stack.addWidget(self.login_page)
        
        # 注册页面
        self.register_page = self._create_register_page()
        self.stack.addWidget(self.register_page)
        
        layout.addWidget(self.stack)
        layout.addStretch()
        
        # 默认提示
        hint = BodyLabel("默认账号: admin / admin123")
        hint.setAlignment(Qt.AlignCenter)
        hint.setStyleSheet("color: #999; font-size: 11px;")
        layout.addWidget(hint)
    
    def _create_login_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)
        
        # 用户名
        self.login_username = LineEdit()
        self.login_username.setPlaceholderText("用户名")
        self.login_username.setFixedHeight(44)
        layout.addWidget(self.login_username)
        
        # 密码
        self.login_password = PasswordLineEdit()
        self.login_password.setPlaceholderText("密码")
        self.login_password.setFixedHeight(44)
        layout.addWidget(self.login_password)
        
        # 登录按钮
        self.login_btn = PrimaryPushButton("登 录")
        self.login_btn.setFixedHeight(44)
        self.login_btn.setDefault(True)
        self.login_btn.setAutoDefault(True)
        self.login_btn.clicked.connect(self._do_login)
        layout.addWidget(self.login_btn)
        
        # 切换到注册
        switch_layout = QHBoxLayout()
        switch_layout.addStretch()
        switch_label = BodyLabel("没有账号？")
        switch_layout.addWidget(switch_label)
        switch_btn = PushButton("立即注册")
        switch_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        switch_layout.addWidget(switch_btn)
        switch_layout.addStretch()
        layout.addLayout(switch_layout)
        
        return page
    
    def _create_register_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(20)
        
        # 用户名
        self.reg_username = LineEdit()
        self.reg_username.setPlaceholderText("用户名（至少3个字符）")
        self.reg_username.setFixedHeight(42)
        layout.addWidget(self.reg_username)
        
        # 密码
        self.reg_password = PasswordLineEdit()
        self.reg_password.setPlaceholderText("密码（至少8位，避免常见密码）")
        self.reg_password.setFixedHeight(42)
        layout.addWidget(self.reg_password)
        
        # 确认密码
        self.reg_confirm = PasswordLineEdit()
        self.reg_confirm.setPlaceholderText("确认密码")
        self.reg_confirm.setFixedHeight(42)
        layout.addWidget(self.reg_confirm)
        
        # 角色选择
        from qfluentwidgets import ComboBox
        role_layout = QHBoxLayout()
        role_label = BodyLabel("注册类型：")
        role_layout.addWidget(role_label)
        self.reg_role = ComboBox()
        self.reg_role.addItems(["客服", "管理员"])
        self.reg_role.setFixedHeight(30)
        role_layout.addWidget(self.reg_role)
        role_layout.addStretch()
        layout.addLayout(role_layout)
        
        # 注册按钮
        self.reg_btn = PrimaryPushButton("提交注册申请")
        self.reg_btn.setFixedHeight(42)
        self.reg_btn.clicked.connect(self._do_register)
        layout.addWidget(self.reg_btn)
        
        # 切换到登录
        switch_layout = QHBoxLayout()
        switch_layout.addStretch()
        switch_label = BodyLabel("已有账号？")
        switch_layout.addWidget(switch_label)
        switch_btn = PushButton("返回登录")
        switch_btn.setFixedSize(90,32)
        switch_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        switch_layout.addWidget(switch_btn)
        switch_layout.addStretch()
        layout.addLayout(switch_layout)
        
        return page
    
    def _do_login(self):
        """执行登录"""
        if self._login_in_progress:
            return
        self._login_in_progress = True

        username = self.login_username.text().strip()
        password = self.login_password.text()

        try:
            success, message, require_change, role = self.auth.login(username, password)
            
            if success:
                # 设置权限管理器的当前用户
                from core.config_utils.permissions import get_permission_manager
                pm = get_permission_manager()
                pm.set_current_user(username)
                
                if require_change:
                    dlg = ChangePasswordDialog(username, self.auth, parent=self)
                    if dlg.exec():
                        self.login_success.emit(username, role)
                        self.accept()
                    else:
                        InfoBar.warning(
                            title="需要修改密码",
                            content="为保障安全，请先完成密码修改",
                            parent=self,
                            position=InfoBarPosition.TOP,
                        )
                else:
                    self.login_success.emit(username, role)
                    self.accept()
            else:
                InfoBar.error(
                    title="登录失败",
                    content=message,
                    parent=self,
                    position=InfoBarPosition.TOP
                )
        finally:
            self._login_in_progress = False
    
    def _do_register(self):
        """执行注册"""
        username = self.reg_username.text().strip()
        password = self.reg_password.text()
        confirm = self.reg_confirm.text()
        role_text = self.reg_role.currentText()
        role = "admin" if role_text == "管理员" else "cs"
        
        success, message = self.auth.register(username, password, confirm, role)
        
        if success:
            InfoBar.success(
                title="申请已提交",
                content=message,
                parent=self,
                position=InfoBarPosition.TOP,
                duration=3000
            )
            # 清空表单并切换到登录页
            self.reg_username.clear()
            self.reg_password.clear()
            self.reg_confirm.clear()
            self.reg_role.setCurrentIndex(0)
            self.stack.setCurrentIndex(0)
        else:
            InfoBar.error(
                title="提交失败",
                content=message,
                parent=self,
                position=InfoBarPosition.TOP
            )
