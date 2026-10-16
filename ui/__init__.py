# -*- coding: utf-8 -*-
"""UI组件包"""

from ui.windows.main_window import MainWindow
from ui.windows.chat_window import ChatWindow
from ui.components.chat_interface import ChatInterface
from ui.pages.workbench_interface import WorkbenchInterface
from ui.pages.knowledge_interface import KnowledgeInterface
from ui.pages.product_interface import ProductInterface
from ui.pages.human_service_interface import HumanServiceInterface
from ui.pages.log_interface import LogInterface
from ui.pages.performance_interface import PerformanceInterface
from ui.pages.statistics_interface import StatisticsInterface
from ui.pages.user_interface import UserInterface
from ui.dialogs.settings_dialog import SettingsDialog
from ui.dialogs.login_dialog import LoginDialog

__all__ = [
    "MainWindow",
    "ChatWindow",
    "ChatInterface",
    "WorkbenchInterface",
    "KnowledgeInterface",
    "ProductInterface",
    "HumanServiceInterface",
    "LogInterface",
    "PerformanceInterface",
    "StatisticsInterface",
    "UserInterface",
    "SettingsDialog",
    "LoginDialog",
]
