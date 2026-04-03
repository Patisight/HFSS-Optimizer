"""
HFSS天线优化程序 - PyQt6现代UI v2
完整功能：自检、优化、实时日志、进度显示
"""

import sys
import os
import json
import subprocess
import threading
import time
import traceback
from pathlib import Path
from datetime import datetime
from PyQt6 import QtWidgets, QtCore, QtGui
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QTableWidget, QTableWidgetItem, QPushButton, QLabel,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QGroupBox, QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QHeaderView, QStyleFactory, QApplication, QTabWidget,
    QProgressDialog, QFormLayout, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QTextCursor

from config.surrogate_config import (
    SURROGATE_MODELS, COMMON_PARAMS, get_model_default_config
)
from utils.formula import FormulaValidator

PROJECT_ROOT = Path(__file__).parent
CONFIG_FILE = PROJECT_ROOT / "user_config.json"


class Colors:
    """配色方案"""
    PRIMARY = "#2980b9"
    SUCCESS = "#27ae60"
    WARNING = "#d4a017"
    DANGER = "#c0392b"
    DARK = "#1a1a2e"
    GRAY = "#5a5a6e"
    LIGHT = "#e8e8e8"
    WHITE = "#ffffff"
    BG = "#16213e"


class OptimizationThread(QThread):
    """优化线程"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)  # current, total
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True
        self.process = None
        self.total_evals = 100  # 默认值

    def run(self):
        try:
            self.log_signal.emit("=" * 50)
            self.log_signal.emit("开始优化流程...")
            self.log_signal.emit("=" * 50)

            # 调用run.py
            cmd = [sys.executable, str(PROJECT_ROOT / "run.py"), "--config", str(CONFIG_FILE)]
            self.log_signal.emit(f"命令: {' '.join(cmd)}")

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=str(PROJECT_ROOT)
            )

            for line in iter(self.process.stdout.readline, ''):
                if not self.is_running:
                    break
                line = line.strip()
                if line:
                    # 解析进度信息
                    if line.startswith('[PROGRESS]'):
                        parts = line.split()
                        if len(parts) >= 2:
                            if parts[1].startswith('TOTAL:'):
                                self.total_evals = int(parts[1].split(':')[1])
                            else:
                                current = int(parts[1])
                                self.progress_signal.emit(current, self.total_evals)
                    else:
                        self.log_signal.emit(line)

            self.process.wait()

            if self.is_running:
                self.progress_signal.emit(self.total_evals, self.total_evals)
                self.finished_signal.emit(True, "优化完成!")
            else:
                self.finished_signal.emit(False, "优化已停止")

        except Exception as e:
            self.finished_signal.emit(False, f"错误: {e}")

    def stop(self):
        self.is_running = False
        if self.process:
            self.process.terminate()


class CheckThread(QThread):
    """自检线程"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict, str)

    def __init__(self, config, n_samples=10):
        super().__init__()
        self.config = config
        self.n_samples = n_samples

    def run(self):
        try:
            sys.path.insert(0, str(PROJECT_ROOT / 'tests'))
            from checker import ProjectChecker

            def progress_callback(current, total, message):
                self.progress_signal.emit(f"[{current}/{total}] {message}")

            checker = ProjectChecker(self.config, n_samples=self.n_samples)
            results = checker.run_all_checks(progress_callback)
            report = checker.get_report_text()

            self.finished_signal.emit(results, report)

        except Exception as e:
            self.finished_signal.emit({}, f"自检异常: {e}")


class HFSSOptimizerGUI(QtWidgets.QMainWindow):
    """HFSS天线优化程序主窗口"""

    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.optimization_thread = None
        self.check_thread = None
        self.is_running = False
        self._loading_ui = False  # 防止 load_data_to_ui 触发 itemChanged

        self.init_ui()
        self.load_data_to_ui()

    def init_ui(self):
        """初始化UI"""
        self.setWindowTitle("HFSS天线优化程序 v2.0")
        self.setMinimumSize(1100, 800)

        # 应用样式 - 深色主题
        self.setStyleSheet(f"""
            QMainWindow {{ background: {Colors.BG}; }}
            QLabel {{ color: #e0e0e0; }}
            QGroupBox {{
                font-weight: bold;
                font-size: 12px;
                border: 1px solid #3a3a5c;
                border-radius: 8px;
                margin-top: 12px;
                padding: 10px;
                background: #1e1e3f;
                color: #e0e0e0;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 8px;
                color: #4a9eff;
            }}
            QTabWidget::pane {{
                border: 1px solid #3a3a5c;
                border-radius: 8px;
                background: #1e1e3f;
                margin-top: -1px;
            }}
            QTabBar {{
                background: {Colors.BG};
            }}
            QTabBar::tab {{
                padding: 12px 24px;
                background: #2a2a4a;
                border: none;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: bold;
                color: #a0a0b0;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{ 
                background: #1e1e3f; 
                color: #ffffff;
                border-bottom: 3px solid #4a9eff;
            }}
            QTabBar::tab:hover {{ 
                background: #3a3a5c;
                color: #ffffff;
            }}
            QProgressBar {{
                border: 2px solid #3a3a5c;
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                min-height: 25px;
                background: #1a1a2e;
                color: #ffffff;
            }}
            QProgressBar::chunk {{ background: {Colors.PRIMARY}; border-radius: 6px; }}
            QTextEdit {{
                border: 1px solid #3a3a5c;
                border-radius: 6px;
                background: #0d0d1a;
                color: #00ff88;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                padding: 8px;
            }}
            QTableWidget {{
                border: 1px solid #3a3a5c;
                border-radius: 6px;
                gridline-color: #3a3a5c;
                background: #1a1a2e;
                color: #ffffff;
                alternate-background-color: #252545;
            }}
            QTableWidget::item {{ 
                padding: 6px; 
                color: #ffffff;
                background: #1a1a2e;
            }}
            QTableWidget::item:selected {{ 
                background: {Colors.PRIMARY}; 
                color: #ffffff;
            }}
            QHeaderView::section {{
                background: #2a2a4a;
                padding: 10px;
                border: none;
                border-right: 1px solid #3a3a5c;
                border-bottom: 2px solid #4a9eff;
                font-weight: bold;
                color: #ffffff;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                border: 1px solid #3a3a5c;
                border-radius: 6px;
                padding: 6px 10px;
                background: #2a2a4a;
                color: #ffffff;
            }}
            QLineEdit:focus, QComboBox:focus {{ border: 2px solid {Colors.PRIMARY}; }}
            QLineEdit::placeholder {{ color: #888888; }}
            QComboBox {{ 
                color: #ffffff;
                background: #2a2a4a;
            }}
            QComboBox::drop-down {{ 
                border: none;
                background: #2a2a4a;
            }}
            QComboBox QAbstractItemView {{
                background: #2a2a4a;
                color: #ffffff;
                selection-background-color: {Colors.PRIMARY};
            }}
            QCheckBox {{ 
                spacing: 8px; 
                color: #e0e0e0;
            }}
            QCheckBox::indicator {{ 
                width: 18px; 
                height: 18px;
                background: #2a2a4a;
                border: 1px solid #3a3a5c;
                border-radius: 4px;
            }}
            QCheckBox::indicator:checked {{ 
                background: {Colors.PRIMARY};
            }}
            QSpinBox, QDoubleSpinBox {{
                color: #ffffff;
                background: #2a2a4a;
            }}
            QSpinBox::up-button, QDoubleSpinBox::up-button {{
                background: #3a3a5c;
                border-left: 1px solid #3a3a5c;
            }}
            QSpinBox::down-button, QDoubleSpinBox::down-button {{
                background: #3a3a5a;
                border-left: 1px solid #3a3a5c;
            }}
            QPushButton {{
                background: #3a3a5c;
                border: 1px solid #4a4a6c;
                border-radius: 6px;
                padding: 8px 16px;
                color: #ffffff;
            }}
            QPushButton:hover {{ background: #4a4a6c; }}
            QPushButton:pressed {{ background: #2a2a4c; }}
            QPushButton:disabled {{ background: #2a2a3c; color: #666666; }}
            QPushButton[primary='true'] {{
                background: {Colors.PRIMARY};
                border: none;
                color: white;
                font-weight: bold;
            }}
            QPushButton[primary='true']:hover {{ background: #4a9eff; }}
            QPushButton[primary='true']:disabled {{ background: #3a3a5c; }}
            QPushButton[success='true'] {{
                background: {Colors.SUCCESS};
                border: none;
                color: white;
                font-weight: bold;
            }}
            QPushButton[success='true']:hover {{ background: #2ecc71; }}
            QPushButton[danger='true'] {{
                background: {Colors.DANGER};
                border: none;
                color: white;
                font-weight: bold;
            }}
            QPushButton[danger='true']:hover {{ background: #e74c3c; }}
        """)

        # 中央部件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # 标题栏
        title_layout = QHBoxLayout()
        title_label = QLabel("HFSS天线优化程序")
        title_label.setStyleSheet("font-size: 28px; font-weight: bold; color: #ffffff;")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        self.status_indicator = QLabel("● 就绪")
        self.status_indicator.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 14px; font-weight: bold;")
        title_layout.addWidget(self.status_indicator)
        main_layout.addLayout(title_layout)

        # 标签页
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_project_tab(), "📁 项目配置")
        self.tabs.addTab(self.create_variables_tab(), "⚙️ 优化变量")
        self.tabs.addTab(self.create_objectives_tab(), "🎯 优化目标")
        self.tabs.addTab(self.create_algorithm_tab(), "⚡ 算法配置")
        self.tabs.addTab(self.create_run_tab(), "🚀 运行控制")
        main_layout.addWidget(self.tabs)

    # ==================== 项目配置页 ====================
    def create_project_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # HFSS配置
        hfss_group = QGroupBox("HFSS 连接配置")
        hfss_layout = QGridLayout()

        hfss_layout.addWidget(QLabel("项目路径:"), 0, 0)
        self.project_path_edit = QLineEdit()
        self.project_path_edit.setPlaceholderText("选择HFSS项目文件(.aedt)")
        hfss_layout.addWidget(self.project_path_edit, 0, 1, 1, 2)
        browse_btn = QPushButton("浏览...")
        browse_btn.setProperty('class', 'secondary')
        browse_btn.clicked.connect(self.browse_project)
        hfss_layout.addWidget(browse_btn, 0, 3)

        hfss_layout.addWidget(QLabel("设计名称:"), 1, 0)
        self.design_name_edit = QLineEdit("HFSSDesign1")
        hfss_layout.addWidget(self.design_name_edit, 1, 1, 1, 3)

        hfss_layout.addWidget(QLabel("设置名称:"), 2, 0)
        self.setup_name_edit = QLineEdit("Setup1")
        hfss_layout.addWidget(self.setup_name_edit, 2, 1, 1, 3)

        hfss_layout.addWidget(QLabel("扫频名称:"), 3, 0)
        self.sweep_name_edit = QLineEdit("Sweep")
        hfss_layout.addWidget(self.sweep_name_edit, 3, 1, 1, 3)

        hfss_group.setLayout(hfss_layout)
        layout.addWidget(hfss_group)

        # 说明
        info_group = QGroupBox("使用说明")
        info_layout = QVBoxLayout()
        info_text = QLabel(
            "<b>步骤:</b><br><br>"
            "1. 选择HFSS项目文件并配置设计参数<br><br>"
            "2. 在'优化变量'中添加天线设计参数<br><br>"
            "3. 在'优化目标'中定义目标和权重<br><br>"
            "4. 选择算法并配置参数<br><br>"
            "5. 点击'运行控制'开始优化"
        )
        info_text.setStyleSheet("color: #c0c0c0; line-height: 1.8;")
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        return widget

    def browse_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择HFSS项目", "", "AEDT Files (*.aedt)")
        if path:
            self.project_path_edit.setText(path)

    # ==================== 变量页 ====================
    def create_variables_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        layout.addWidget(QLabel("添加优化变量（天线设计参数，如长度、宽度、角度等）"))

        self.var_table = QTableWidget()
        self.var_table.setColumnCount(4)
        self.var_table.setHorizontalHeaderLabels(['名称', '最小值', '最大值', '单位'])
        self.var_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.var_table.setColumnWidth(1, 100)
        self.var_table.setColumnWidth(2, 100)
        self.var_table.setColumnWidth(3, 80)
        self.var_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.var_table.cellClicked.connect(self._on_var_table_selected)
        layout.addWidget(self.var_table)

        # 表单
        form_group = QGroupBox("添加/修改变量")
        form_layout = QGridLayout()

        form_layout.addWidget(QLabel("名称:"), 0, 0)
        self.var_name_edit = QLineEdit()
        self.var_name_edit.setPlaceholderText("如: length")
        form_layout.addWidget(self.var_name_edit, 0, 1)

        form_layout.addWidget(QLabel("最小值:"), 0, 2)
        self.var_min_edit = QDoubleSpinBox()
        self.var_min_edit.setRange(-1e6, 1e6)
        self.var_min_edit.setDecimals(4)
        self.var_min_edit.setValue(0)
        form_layout.addWidget(self.var_min_edit, 0, 3)

        form_layout.addWidget(QLabel("最大值:"), 0, 4)
        self.var_max_edit = QDoubleSpinBox()
        self.var_max_edit.setRange(-1e6, 1e6)
        self.var_max_edit.setDecimals(4)
        self.var_max_edit.setValue(10)
        form_layout.addWidget(self.var_max_edit, 0, 5)

        form_layout.addWidget(QLabel("单位:"), 0, 6)
        self.var_unit_edit = QLineEdit("mm")
        self.var_unit_edit.setFixedWidth(60)
        form_layout.addWidget(self.var_unit_edit, 0, 7)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # 按钮
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("添加")
        add_btn.setProperty('primary', 'true')
        add_btn.clicked.connect(self.add_variable)
        btn_layout.addWidget(add_btn)

        update_btn = QPushButton("修改选中")
        update_btn.clicked.connect(self.update_variable)
        btn_layout.addWidget(update_btn)

        delete_btn = QPushButton("删除选中")
        delete_btn.clicked.connect(self.delete_variable)
        btn_layout.addWidget(delete_btn)

        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(lambda: self.var_table.setRowCount(0))
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return widget

    def add_variable(self):
        name = self.var_name_edit.text().strip()
        if not name:
            self._show_message("警告", "请输入变量名称", "warning")
            return
        row = self.var_table.rowCount()
        self.var_table.insertRow(row)
        self.var_table.setItem(row, 0, QTableWidgetItem(name))
        self.var_table.setItem(row, 1, QTableWidgetItem(str(self.var_min_edit.value())))
        self.var_table.setItem(row, 2, QTableWidgetItem(str(self.var_max_edit.value())))
        self.var_table.setItem(row, 3, QTableWidgetItem(self.var_unit_edit.text()))
        self.var_name_edit.clear()

    def _on_var_table_selected(self, row, _col):
        """选中变量行时，将行数据回填到编辑表单"""
        if self._loading_ui:
            return
        item_name = self.var_table.item(row, 0)
        item_min = self.var_table.item(row, 1)
        item_max = self.var_table.item(row, 2)
        item_unit = self.var_table.item(row, 3)
        if item_name:
            self.var_name_edit.setText(item_name.text())
        if item_min:
            try:
                self.var_min_edit.setValue(float(item_min.text()))
            except ValueError:
                pass
        if item_max:
            try:
                self.var_max_edit.setValue(float(item_max.text()))
            except ValueError:
                pass
        if item_unit:
            self.var_unit_edit.setText(item_unit.text())

    def update_variable(self):
        row = self.var_table.currentRow()
        if row < 0:
            self._show_message("警告", "请先选中要修改的行", "warning")
            return
        self.var_table.setItem(row, 0, QTableWidgetItem(self.var_name_edit.text()))
        self.var_table.setItem(row, 1, QTableWidgetItem(str(self.var_min_edit.value())))
        self.var_table.setItem(row, 2, QTableWidgetItem(str(self.var_max_edit.value())))
        self.var_table.setItem(row, 3, QTableWidgetItem(self.var_unit_edit.text()))

    def delete_variable(self):
        row = self.var_table.currentRow()
        if row >= 0:
            self.var_table.removeRow(row)

    # ==================== 目标页 ====================
    def create_objectives_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)

        info_label = QLabel("定义优化目标（权重越大表示该目标越重要）")
        info_label.setStyleSheet("color: #a0a0b0;")
        layout.addWidget(info_label)

        self.obj_table = QTableWidget()
        self.obj_table.setColumnCount(7)
        self.obj_table.setHorizontalHeaderLabels(['名称', '类型', '频段(GHz)', '公式', '目标值', '方向', '权重'])
        self.obj_table.setColumnWidth(0, 80)
        self.obj_table.setColumnWidth(1, 70)
        self.obj_table.setColumnWidth(2, 100)
        self.obj_table.setColumnWidth(3, 180)
        self.obj_table.setColumnWidth(4, 70)
        self.obj_table.setColumnWidth(5, 70)
        self.obj_table.setColumnWidth(6, 50)
        self.obj_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.obj_table.cellClicked.connect(self._on_obj_table_selected)
        layout.addWidget(self.obj_table)

        # 表单
        form_group = QGroupBox("添加/修改目标")
        form_layout = QGridLayout()

        form_layout.addWidget(QLabel("名称:"), 0, 0)
        self.obj_name_edit = QLineEdit()
        self.obj_name_edit.setPlaceholderText("如: S11")
        form_layout.addWidget(self.obj_name_edit, 0, 1)

        form_layout.addWidget(QLabel("类型:"), 0, 2)
        self.obj_type_combo = QComboBox()
        self.obj_type_combo.addItems(['S参数', 'Gain', 'peakGain'])
        form_layout.addWidget(self.obj_type_combo, 0, 3)

        form_layout.addWidget(QLabel("目标值:"), 0, 4)
        self.obj_goal_edit = QDoubleSpinBox()
        self.obj_goal_edit.setRange(-1e6, 1e6)
        self.obj_goal_edit.setDecimals(2)
        self.obj_goal_edit.setValue(-10)
        form_layout.addWidget(self.obj_goal_edit, 0, 5)

        form_layout.addWidget(QLabel("方向:"), 0, 6)
        self.obj_target_combo = QComboBox()
        self.obj_target_combo.addItems(['minimize', 'maximize'])
        form_layout.addWidget(self.obj_target_combo, 0, 7)

        form_layout.addWidget(QLabel("频率:"), 1, 0)
        self.obj_freq_edit = QLineEdit("5.1-7.2")
        self.obj_freq_edit.setPlaceholderText("5.1-7.2 或 5.5")
        form_layout.addWidget(self.obj_freq_edit, 1, 1, 1, 2)

        # 公式输入（当类型为 formula 时显示）
        form_layout.addWidget(QLabel("公式:"), 2, 0)
        self.obj_formula_edit = QLineEdit()
        self.obj_formula_edit.setPlaceholderText("例如: dB(S(1,1)) + dB(S(2,1))")
        self.obj_formula_edit.setMinimumWidth(250)
        self.obj_formula_edit.setStyleSheet("background-color: #2a2a4a; color: #a0a0b0;")
        form_layout.addWidget(self.obj_formula_edit, 2, 1, 1, 3)
        
        # 公式验证按钮
        self.obj_formula_validate_btn = QPushButton("验证")
        self.obj_formula_validate_btn.setStyleSheet("background-color: #3498db; color: white; padding: 3px 8px;")
        self.obj_formula_validate_btn.clicked.connect(self.validate_objective_formula)
        form_layout.addWidget(self.obj_formula_validate_btn, 2, 4)
        
        # 公式错误提示
        self.obj_formula_error_label = QLabel("")
        self.obj_formula_error_label.setStyleSheet("color: #e74c3c; font-size: 12px;")
        form_layout.addWidget(self.obj_formula_error_label, 2, 5, 1, 2)

        # 隐藏公式输入框初始状态
        self.obj_formula_edit.setVisible(False)
        self.obj_formula_validate_btn.setVisible(False)
        self.obj_formula_error_label.setVisible(False)

        # 连接类型变化事件
        self.obj_type_combo.currentTextChanged.connect(self._on_obj_type_changed)

        form_layout.addWidget(QLabel("权重:"), 1, 3)
        self.obj_weight_edit = QDoubleSpinBox()
        self.obj_weight_edit.setRange(0.1, 100)
        self.obj_weight_edit.setDecimals(1)
        self.obj_weight_edit.setValue(1.0)
        form_layout.addWidget(self.obj_weight_edit, 1, 4)

        form_group.setLayout(form_layout)
        layout.addWidget(form_group)

        # 按钮
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("添加")
        add_btn.setProperty('primary', 'true')
        add_btn.clicked.connect(self.add_objective)
        btn_layout.addWidget(add_btn)

        update_btn = QPushButton("修改选中")
        update_btn.clicked.connect(self.update_objective)
        btn_layout.addWidget(update_btn)

        delete_btn = QPushButton("删除选中")
        delete_btn.clicked.connect(self.delete_objective)
        btn_layout.addWidget(delete_btn)

        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(lambda: self.obj_table.setRowCount(0))
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        return widget

    def _on_obj_type_changed(self, obj_type: str):
        """当目标类型变化时"""
        is_s_param = (obj_type == 'S参数')
        self.obj_freq_edit.setVisible(True)  # 频率设置始终显示
        self.obj_formula_edit.setVisible(is_s_param)
        self.obj_formula_validate_btn.setVisible(is_s_param)
        self.obj_formula_error_label.setVisible(False)
        
        # S参数类型时，强制设置默认公式和频率
        if is_s_param:
            self.obj_formula_edit.setText('dB(S(1,1))')
            if not self.obj_freq_edit.text() or self.obj_freq_edit.text() == '5.1-7.2':
                self.obj_freq_edit.setText('5.6-6.2')
        
        # 提示文本
        if is_s_param:
            self.obj_freq_edit.setPlaceholderText("频段范围，如 5.6-6.2")
        else:
            self.obj_freq_edit.setPlaceholderText("5.6-6.2 或 5.9")

    def validate_objective_formula(self):
        """验证公式语法"""
        formula = self.obj_formula_edit.text().strip()
        if not formula:
            self.obj_formula_error_label.setText("请输入公式")
            self.obj_formula_error_label.setStyleSheet("color: #e74c3c; font-size: 12px;")
            self.obj_formula_error_label.setVisible(True)
            return False
        
        valid, errors = FormulaValidator(formula).validate()
        if valid:
            self.obj_formula_error_label.setText("✓ 公式正确")
            self.obj_formula_error_label.setStyleSheet("color: #27ae60; font-size: 12px;")
            self.obj_formula_error_label.setVisible(True)
            return True
        else:
            self.obj_formula_error_label.setText(f"错误: {errors[0]}")
            self.obj_formula_error_label.setStyleSheet("color: #e74c3c; font-size: 12px;")
            self.obj_formula_error_label.setVisible(True)
            return False

    def add_objective(self):
        name = self.obj_name_edit.text().strip()
        if not name:
            self._show_message("警告", "请输入目标名称", "warning")
            return
        
        obj_type = self.obj_type_combo.currentText()
        
        # 如果是S参数类型，验证公式
        if obj_type == 'S参数':
            formula = self.obj_formula_edit.text().strip()
            if not formula:
                self._show_message("警告", "请输入公式", "warning")
                return
            valid, errors = FormulaValidator(formula).validate()
            if not valid:
                self.obj_formula_error_label.setText(f"错误: {errors[0]}")
                self.obj_formula_error_label.setStyleSheet("color: #e74c3c; font-size: 12px;")
                self.obj_formula_error_label.setVisible(True)
                self._show_message("警告", f"公式错误: {errors[0]}", "warning")
                return
            # 尝试自动修正
            suggestion = FormulaValidator.suggest_correction(formula)
            if suggestion and suggestion != formula:
                formula = suggestion
                self.obj_formula_edit.setText(formula)
                self._show_message("提示", f"公式已修正为: {formula}", "info")
        
        row = self.obj_table.rowCount()
        self.obj_table.insertRow(row)
        self.obj_table.setItem(row, 0, QTableWidgetItem(name))
        self.obj_table.setItem(row, 1, QTableWidgetItem(obj_type))
        # 第2列：频段（对于所有类型都是频率范围）
        self.obj_table.setItem(row, 2, QTableWidgetItem(self.obj_freq_edit.text()))
        # 第3列：公式（仅S参数类型使用）
        self.obj_table.setItem(row, 3, QTableWidgetItem(self.obj_formula_edit.text() if obj_type == 'S参数' else ''))
        self.obj_table.setItem(row, 4, QTableWidgetItem(str(self.obj_goal_edit.value())))
        self.obj_table.setItem(row, 5, QTableWidgetItem(self.obj_target_combo.currentText()))
        self.obj_table.setItem(row, 6, QTableWidgetItem(str(self.obj_weight_edit.value())))
        self.obj_name_edit.clear()
        self.obj_formula_error_label.setVisible(False)

    def _on_obj_table_selected(self, row, _col):
        """选中目标行时，将行数据回填到编辑表单"""
        if self._loading_ui:
            return
        item_name = self.obj_table.item(row, 0)
        item_type = self.obj_table.item(row, 1)
        item_freq = self.obj_table.item(row, 2)
        item_formula = self.obj_table.item(row, 3)
        item_goal = self.obj_table.item(row, 4)
        item_target = self.obj_table.item(row, 5)
        item_weight = self.obj_table.item(row, 6)
        if item_name:
            self.obj_name_edit.setText(item_name.text())
        if item_type:
            gui_type = item_type.text()
            idx = self.obj_type_combo.findText(gui_type)
            if idx >= 0:
                self.obj_type_combo.setCurrentIndex(idx)
            # 触发类型变化事件以更新公式输入框的显示
            self._on_obj_type_changed(gui_type)
        if item_freq:
            self.obj_freq_edit.setText(item_freq.text())
        if item_formula:
            self.obj_formula_edit.setText(item_formula.text())
        else:
            self.obj_formula_edit.setText('dB(S(1,1))')
        if item_goal:
            try:
                self.obj_goal_edit.setValue(float(item_goal.text()))
            except ValueError:
                pass
        if item_target:
            idx = self.obj_target_combo.findText(item_target.text())
            if idx >= 0:
                self.obj_target_combo.setCurrentIndex(idx)
        if item_weight:
            try:
                self.obj_weight_edit.setValue(float(item_weight.text()))
            except ValueError:
                pass

    def update_objective(self):
        row = self.obj_table.currentRow()
        if row < 0:
            self._show_message("警告", "请先选中要修改的行", "warning")
            return
        obj_type = self.obj_type_combo.currentText()
        self.obj_table.setItem(row, 0, QTableWidgetItem(self.obj_name_edit.text()))
        self.obj_table.setItem(row, 1, QTableWidgetItem(obj_type))
        self.obj_table.setItem(row, 2, QTableWidgetItem(self.obj_freq_edit.text()))
        self.obj_table.setItem(row, 3, QTableWidgetItem(self.obj_formula_edit.text() if obj_type == 'S参数' else ''))
        self.obj_table.setItem(row, 4, QTableWidgetItem(str(self.obj_goal_edit.value())))
        self.obj_table.setItem(row, 5, QTableWidgetItem(self.obj_target_combo.currentText()))
        self.obj_table.setItem(row, 6, QTableWidgetItem(str(self.obj_weight_edit.value())))

    def delete_objective(self):
        row = self.obj_table.currentRow()
        if row >= 0:
            self.obj_table.removeRow(row)

    # ==================== 算法页 ====================
    def create_algorithm_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # 创建滚动区域
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: #1a1a2e;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #4a4a6c;
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: #5a5a7c;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        
        # 滚动区域的内容widget
        scroll_content = QWidget()
        layout = QVBoxLayout(scroll_content)
        layout.setSpacing(15)
        layout.setContentsMargins(15, 15, 15, 15)

        # 算法选择
        algo_group = QGroupBox("算法选择")
        algo_layout = QGridLayout()
        algo_layout.addWidget(QLabel("优化算法:"), 0, 0)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['mobo', 'mopso', 'nsga2', 'surrogate', 'robust', 'adaptive'])
        self.algorithm_combo.setCurrentText('mobo')
        algo_layout.addWidget(self.algorithm_combo, 0, 1)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # 基础参数
        basic_group = QGroupBox("基础参数")
        basic_layout = QGridLayout()
        basic_layout.addWidget(QLabel("种群/初始样本:"), 0, 0)
        self.population_spin = QSpinBox()
        self.population_spin.setRange(1, 1000)
        self.population_spin.setValue(30)
        basic_layout.addWidget(self.population_spin, 0, 1)

        basic_layout.addWidget(QLabel("迭代次数:"), 0, 2)
        self.generations_spin = QSpinBox()
        self.generations_spin.setRange(1, 1000)
        self.generations_spin.setValue(30)
        basic_layout.addWidget(self.generations_spin, 0, 3)
        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        # ==================== 代理模型配置（动态） ====================
        surrogate_group = QGroupBox("代理模型")
        surrogate_main_layout = QVBoxLayout()
        
        # 启用开关
        self.surrogate_check = QCheckBox("启用代理模型加速")
        surrogate_main_layout.addWidget(self.surrogate_check)
        
        # 模型类型选择
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("模型类型:"))
        self.surrogate_type_combo = QComboBox()
        # 使用 SURROGATE_MODELS 中的显示名称
        for model_key, model_info in SURROGATE_MODELS.items():
            self.surrogate_type_combo.addItem(model_info['display_name'], model_key)
        self.surrogate_type_combo.setCurrentIndex(3)  # 默认 gpflow_svgp
        self.surrogate_type_combo.currentIndexChanged.connect(self._on_surrogate_type_changed)
        type_layout.addWidget(self.surrogate_type_combo)
        type_layout.addStretch()
        surrogate_main_layout.addLayout(type_layout)
        
        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("background-color: #3a3a5c;")
        surrogate_main_layout.addWidget(line)
        
        # 模型描述区域
        self.model_desc_label = QLabel()
        self.model_desc_label.setStyleSheet("color: #a0a0b0; padding: 5px;")
        self.model_desc_label.setWordWrap(True)
        surrogate_main_layout.addWidget(self.model_desc_label)
        
        # 通用参数区域
        common_group = QGroupBox("通用参数")
        common_layout = QFormLayout()
        
        # 最小训练样本
        self.min_samples_spin = QSpinBox()
        self.min_samples_spin.setRange(3, 100)
        self.min_samples_spin.setValue(5)
        self.min_samples_spin.setToolTip(COMMON_PARAMS[0]['tooltip'])
        common_layout.addRow("最小训练样本:", self.min_samples_spin)
        
        # 不确定性阈值
        self.uncertainty_spin = QDoubleSpinBox()
        self.uncertainty_spin.setRange(0.1, 2.0)
        self.uncertainty_spin.setDecimals(2)
        self.uncertainty_spin.setValue(0.5)
        self.uncertainty_spin.setToolTip(COMMON_PARAMS[1]['tooltip'])
        common_layout.addRow("不确定性阈值:", self.uncertainty_spin)
        
        common_group.setLayout(common_layout)
        surrogate_main_layout.addWidget(common_group)
        
        # 专属参数区域（动态）
        self.model_params_group = QGroupBox("模型专属参数")
        self.model_params_layout = QFormLayout()
        self.model_params_group.setLayout(self.model_params_layout)
        surrogate_main_layout.addWidget(self.model_params_group)
        
        # 存储动态控件的字典
        self.surrogate_param_controls = {}
        
        surrogate_group.setLayout(surrogate_main_layout)
        layout.addWidget(surrogate_group)
        
        # 初始化显示
        self._on_surrogate_type_changed(self.surrogate_type_combo.currentIndex())

        # 早停
        early_group = QGroupBox("早停配置")
        early_layout = QHBoxLayout()
        self.earlystop_check = QCheckBox("启用早停")
        self.earlystop_check.setChecked(True)
        early_layout.addWidget(self.earlystop_check)
        early_layout.addWidget(QLabel("达标解数量:"))
        self.earlystop_count_spin = QSpinBox()
        self.earlystop_count_spin.setRange(1, 100)
        self.earlystop_count_spin.setValue(5)
        early_layout.addWidget(self.earlystop_count_spin)
        early_layout.addStretch()
        early_group.setLayout(early_layout)
        layout.addWidget(early_group)
        
        # 可视化配置
        viz_group = QGroupBox("可视化配置")
        viz_layout = QHBoxLayout()
        viz_layout.addWidget(QLabel("绘图间隔:"))
        self.plot_interval_spin = QSpinBox()
        self.plot_interval_spin.setRange(1, 100)
        self.plot_interval_spin.setValue(5)
        self.plot_interval_spin.setToolTip("每隔多少次迭代生成一张中间图")
        viz_layout.addWidget(self.plot_interval_spin)
        viz_layout.addWidget(QLabel("次迭代"))
        viz_layout.addWidget(QLabel("|"))
        viz_layout.addWidget(QLabel("对比图窗口:"))
        self.surrogate_recent_window_spin = QSpinBox()
        self.surrogate_recent_window_spin.setRange(1, 50)
        self.surrogate_recent_window_spin.setValue(5)
        self.surrogate_recent_window_spin.setToolTip("对比图展示最近几次评估")
        viz_layout.addWidget(self.surrogate_recent_window_spin)
        viz_layout.addWidget(QLabel("次"))
        viz_layout.addStretch()
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # 说明
        info_group = QGroupBox("算法说明")
        info_layout = QVBoxLayout()
        info_text = QLabel(
            "<b>MOBO</b>: 贝叶斯优化，仿真次数最少，智能选择测试点<br><br>"
            "<b>MOPSO</b>: 粒子群算法，收敛快，推荐中等问题<br><br>"
            "<b>NSGA2</b>: 遗传算法，最稳健，仿真次数最多<br><br>"
            "<b>代理模型</b>: incremental=轻量增量 | <b>gpflow_svgp</b>=复杂场景推荐"
        )
        info_text.setStyleSheet("color: #c0c0c0; line-height: 1.5;")
        info_layout.addWidget(info_text)
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)

        layout.addStretch()
        
        # 设置滚动区域的内容
        scroll_area.setWidget(scroll_content)
        main_layout.addWidget(scroll_area)
        
        return widget

    # ==================== 代理模型动态配置方法 ====================
    def _on_surrogate_type_changed(self, index):
        """模型类型切换时的回调"""
        model_key = self.surrogate_type_combo.currentData()
        self._update_model_desc(model_key)
        self._update_model_params_ui(model_key)

    def _update_model_desc(self, model_key):
        """更新模型描述"""
        model_info = SURROGATE_MODELS.get(model_key, {})
        desc = model_info.get('description', '')
        is_incremental = model_info.get('is_incremental', False)
        
        if is_incremental:
            desc += "\n\n✓ 增量学习模型，每次真实仿真后自动更新"
        
        self.model_desc_label.setText(desc)

    def _update_model_params_ui(self, model_key):
        """根据模型类型动态更新专属参数UI"""
        # 清除现有控件
        while self.model_params_layout.count():
            item = self.model_params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.surrogate_param_controls.clear()
        
        # 获取模型配置定义
        model_info = SURROGATE_MODELS.get(model_key, {})
        params = model_info.get('params', [])
        
        if not params:
            # 无专属参数时显示提示
            label = QLabel("此模型使用默认配置，无额外参数")
            label.setStyleSheet("color: #a0a0b0;")
            self.model_params_layout.addRow(label)
            return
        
        # 创建各参数的控件
        for param in params:
            control = self._create_param_control(param)
            self.surrogate_param_controls[param['key']] = control
            
            # 添加标签和单位
            label_text = param['label']
            if param.get('unit'):
                label_text += f" ({param['unit']})"
            
            self.model_params_layout.addRow(label_text + ":", control)

    def _create_param_control(self, param):
        """根据参数定义创建控件"""
        param_type = param.get('type', 'int')
        
        if param_type == 'int':
            control = QSpinBox()
            control.setRange(param.get('min', 0), param.get('max', 1000))
            control.setValue(param.get('default', 0))
            control.setMinimumWidth(100)
        
        elif param_type == 'float':
            control = QDoubleSpinBox()
            control.setRange(param.get('min', 0), param.get('max', 1))
            control.setDecimals(param.get('decimals', 2))
            control.setValue(param.get('default', 0.5))
            control.setMinimumWidth(100)
        
        elif param_type == 'combo':
            control = QComboBox()
            for opt in param.get('options', []):
                control.addItem(opt['label'], opt['value'])
            control.setMinimumWidth(150)
        
        else:
            control = QLineEdit(str(param.get('default', '')))
        
        # 设置tooltip
        if param.get('tooltip'):
            control.setToolTip(param['tooltip'])
        
        return control

    def _get_surrogate_model_key(self):
        """获取当前选中的模型类型键"""
        return self.surrogate_type_combo.currentData()

    def _set_surrogate_type_by_key(self, model_key):
        """通过模型键设置下拉框选中项"""
        idx = self.surrogate_type_combo.findData(model_key)
        if idx >= 0:
            self.surrogate_type_combo.setCurrentIndex(idx)

    def _get_model_params(self):
        """获取当前模型专属参数值"""
        params = {}
        for key, control in self.surrogate_param_controls.items():
            if isinstance(control, QSpinBox):
                params[key] = control.value()
            elif isinstance(control, QDoubleSpinBox):
                params[key] = control.value()
            elif isinstance(control, QComboBox):
                params[key] = control.currentData()
            elif isinstance(control, QLineEdit):
                params[key] = control.text()
        return params

    def _set_model_params(self, params: dict):
        """设置模型专属参数值"""
        for key, value in params.items():
            if key not in self.surrogate_param_controls:
                continue
            control = self.surrogate_param_controls[key]
            if isinstance(control, QSpinBox):
                control.setValue(int(value))
            elif isinstance(control, QDoubleSpinBox):
                control.setValue(float(value))
            elif isinstance(control, QComboBox):
                idx = control.findData(value)
                if idx >= 0:
                    control.setCurrentIndex(idx)
            elif isinstance(control, QLineEdit):
                control.setText(str(value))

    # ==================== 运行页 ====================
    def create_run_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)

        # 预估
        est_group = QGroupBox("预估信息")
        est_layout = QHBoxLayout()
        self.estimate_label = QLabel("请先配置参数")
        self.estimate_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #4a9eff;")
        est_layout.addWidget(self.estimate_label)
        est_layout.addStretch()
        est_group.setLayout(est_layout)
        layout.addWidget(est_group)

        # 历史数据状态
        history_group = QGroupBox("历史评估数据")
        history_layout = QHBoxLayout()
        self.history_status_label = QLabel("未导入历史数据")
        self.history_status_label.setStyleSheet("font-size: 13px; color: #a0a0b0;")
        history_layout.addWidget(self.history_status_label)
        history_layout.addStretch()
        history_group.setLayout(history_layout)
        layout.addWidget(history_group)

        # 控制按钮
        control_group = QGroupBox("控制")
        control_layout = QHBoxLayout()

        # 自检按钮
        self.check_btn = QPushButton("🔍 自检项目")
        self.check_btn.setProperty('class', 'secondary')
        self.check_btn.clicked.connect(self.run_check)
        control_layout.addWidget(self.check_btn)

        # 自检采样
        control_layout.addWidget(QLabel("采样数:"))
        self.check_samples_spin = QSpinBox()
        self.check_samples_spin.setRange(5, 100)
        self.check_samples_spin.setValue(10)
        self.check_samples_spin.setFixedWidth(60)
        control_layout.addWidget(self.check_samples_spin)

        control_layout.addSpacing(20)

        # 开始/停止
        self.start_btn = QPushButton("▶ 开始优化")
        self.start_btn.setProperty('primary', 'true')
        self.start_btn.setMinimumHeight(45)
        self.start_btn.clicked.connect(self.start_optimization)
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■ 停止")
        self.stop_btn.setProperty('danger', 'true')
        self.stop_btn.setMinimumHeight(45)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_optimization)
        control_layout.addWidget(self.stop_btn)

        control_layout.addSpacing(20)

        # 导入/导出/历史
        import_btn = QPushButton("📥 导入配置")
        import_btn.clicked.connect(self.import_config)
        control_layout.addWidget(import_btn)

        save_btn = QPushButton("💾 保存配置")
        save_btn.clicked.connect(self.save_config)
        control_layout.addWidget(save_btn)

        history_btn = QPushButton("📜 导入历史")
        history_btn.clicked.connect(self.import_history)
        control_layout.addWidget(history_btn)

        self.clear_history_btn = QPushButton("✖ 清除历史")
        self.clear_history_btn.setToolTip("清除已导入的历史评估数据路径")
        self.clear_history_btn.clicked.connect(self.clear_history)
        control_layout.addWidget(self.clear_history_btn)

        open_btn = QPushButton("📂 结果")
        open_btn.clicked.connect(self.open_results)
        control_layout.addWidget(open_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 进度
        progress_group = QGroupBox("优化进度")
        progress_layout = QVBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(30)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("就绪")
        self.progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.progress_label.setStyleSheet("font-weight: bold; color: #ffffff;")
        progress_layout.addWidget(self.progress_label)
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # 日志
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(280)
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # 绑定更新预估
        self.population_spin.valueChanged.connect(self.update_estimate)
        self.generations_spin.valueChanged.connect(self.update_estimate)
        self.algorithm_combo.currentTextChanged.connect(self.update_estimate)
        self.update_estimate()

        return widget

    def update_estimate(self):
        algo = self.algorithm_combo.currentText()
        pop = self.population_spin.value()
        gen = self.generations_spin.value()

        if algo == 'mobo':
            est = pop + gen
            time_est = est * 3 // 60
            text = f"预估: {pop}(初始) + {gen}(迭代) = {est} 次 | 约 {time_est} 小时"
        else:
            est = pop * gen
            time_est = est * 3 // 60
            text = f"预估: {pop} × {gen} = {est} 次 | 约 {time_est} 小时"

        self.estimate_label.setText(text)

    # ==================== 核心功能 ====================
    def run_check(self):
        """运行自检"""
        if not self.project_path_edit.text():
            self._show_message("警告", "请先选择HFSS项目", "warning")
            return

        self.save_config()
        self.log_text.clear()
        self.log("开始自检...")
        self.progress_bar.setValue(0)
        self.progress_label.setText("自检中...")
        self.status_indicator.setText("● 自检中")
        self.status_indicator.setStyleSheet(f"color: {Colors.WARNING}; font-size: 14px; font-weight: bold;")

        n_samples = self.check_samples_spin.value()
        self.check_thread = CheckThread(self.config, n_samples)
        self.check_thread.log_signal.connect(self.log)
        self.check_thread.progress_signal.connect(lambda m: self.progress_label.setText(m))
        self.check_thread.finished_signal.connect(self.check_finished)
        self.check_thread.start()

    def check_finished(self, results, report):
        """自检完成"""
        self.log("\n" + report)
        summary = results.get('summary', {})
        status = summary.get('status', 'UNKNOWN')

        if status == 'OK':
            self.status_indicator.setText("● 自检通过")
            self.status_indicator.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 14px; font-weight: bold;")
            self.progress_label.setText("✅ 所有检查通过，可以开始优化")
        elif status == 'WARNING':
            self.status_indicator.setText("● 有警告")
            self.status_indicator.setStyleSheet(f"color: {Colors.WARNING}; font-size: 14px; font-weight: bold;")
            self.progress_label.setText(f"⚠️ 有 {summary.get('warnings', 0)} 个警告")
        else:
            self.status_indicator.setText("● 自检失败")
            self.status_indicator.setStyleSheet(f"color: {Colors.DANGER}; font-size: 14px; font-weight: bold;")
            self.progress_label.setText(f"❌ 发现 {summary.get('errors', 0)} 个错误")

    def start_optimization(self):
        """开始优化"""
        if self.is_running:
            return

        if not self.project_path_edit.text():
            self._show_message("警告", "请先选择HFSS项目", "warning")
            return

        if self.var_table.rowCount() == 0:
            self._show_message("警告", "请先添加变量", "warning")
            return

        if self.obj_table.rowCount() == 0:
            self._show_message("警告", "请先添加目标", "warning")
            return

        # 自动保存配置
        self.save_config()
        self.log_text.clear()
        self.log("=" * 50)
        self.log("配置已保存，优化开始...")
        self.log("=" * 50)

        self.is_running = True
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_indicator.setText("● 运行中")
        self.status_indicator.setStyleSheet(f"color: {Colors.DANGER}; font-size: 14px; font-weight: bold;")
        self.progress_bar.setValue(0)
        self.progress_label.setText("启动中...")

        self.optimization_thread = OptimizationThread(self.config)
        self.optimization_thread.log_signal.connect(self.log)
        self.optimization_thread.progress_signal.connect(self.update_progress)
        self.optimization_thread.finished_signal.connect(self.optimization_finished)
        self.optimization_thread.start()

    def update_progress(self, current, total):
        """更新进度条"""
        if total > 0:
            percent = int(current / total * 100)
            self.progress_bar.setValue(percent)
            self.progress_label.setText(f"评估进度: {current}/{total} ({percent}%)")

    def stop_optimization(self):
        """停止优化"""
        if self.optimization_thread:
            self.log("正在停止...")
            self.optimization_thread.stop()
            self.is_running = False

    def optimization_finished(self, success, message):
        """优化完成"""
        self.is_running = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        if success:
            self.status_indicator.setText("● 完成")
            self.status_indicator.setStyleSheet(f"color: {Colors.SUCCESS}; font-size: 14px; font-weight: bold;")
            self.progress_label.setText("✅ 优化完成!")
            self._show_message("完成", message, "success")
        else:
            self.status_indicator.setText("● 已停止")
            self.status_indicator.setStyleSheet(f"color: {Colors.WARNING}; font-size: 14px; font-weight: bold;")
            self.progress_label.setText("已停止")

    def open_results(self):
        """打开结果目录"""
        results_dir = str(PROJECT_ROOT / "optim_results")
        os.makedirs(results_dir, exist_ok=True)
        if sys.platform == 'win32':
            subprocess.run(['explorer', results_dir])

    def log(self, message):
        """写日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        # 自动滚动到底部
        cursor = self.log_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.log_text.setTextCursor(cursor)

    # ==================== 配置 ====================
    def load_config(self):
        """加载配置"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            'hfss': {'project_path': '', 'design_name': 'HFSSDesign1', 'setup_name': 'Setup1', 'sweep_name': 'Sweep'},
            'variables': [],
            'objectives': [],
            'algorithm': {'algorithm': 'mobo', 'population_size': 30, 'n_generations': 30, 'surrogate_type': 'gpflow_svgp'}
        }

    def load_data_to_ui(self):
        """加载数据到UI"""
        self._loading_ui = True
        cfg = self.config

        hfss = cfg.get('hfss', {})
        self.project_path_edit.setText(hfss.get('project_path', ''))
        self.design_name_edit.setText(hfss.get('design_name', 'HFSSDesign1'))
        self.setup_name_edit.setText(hfss.get('setup_name', 'Setup1'))
        self.sweep_name_edit.setText(hfss.get('sweep_name', 'Sweep'))

        for var in cfg.get('variables', []):
            row = self.var_table.rowCount()
            self.var_table.insertRow(row)
            self.var_table.setItem(row, 0, QTableWidgetItem(var.get('name', '')))
            self.var_table.setItem(row, 1, QTableWidgetItem(str(var.get('bounds', [0, 1])[0])))
            self.var_table.setItem(row, 2, QTableWidgetItem(str(var.get('bounds', [0, 1])[1])))
            self.var_table.setItem(row, 3, QTableWidgetItem(var.get('unit', 'mm')))

        for obj in cfg.get('objectives', []):
            row = self.obj_table.rowCount()
            self.obj_table.insertRow(row)
            # 内部类型到GUI类型的映射
            internal_to_gui = {
                'formula': 'S参数',
                'gain': 'Gain',
                'peak_gain': 'peakGain',
                's_db': 'S参数',
                's_mag': 'S参数',
                's_phase': 'S参数'
            }
            internal_type = obj.get('type', 'formula')
            gui_type = internal_to_gui.get(internal_type, 'S参数')
            self.obj_table.setItem(row, 0, QTableWidgetItem(obj.get('name', '')))
            self.obj_table.setItem(row, 1, QTableWidgetItem(gui_type))
            
            # 频段/频率
            freq = obj.get('freq', obj.get('freq_range', ['', '']))
            if isinstance(freq, list):
                freq_text = f"{freq[0]}-{freq[1]}" if freq[0] else ''
            else:
                freq_text = str(freq) if freq else ''
            
            # 公式
            formula_text = obj.get('formula', 'dB(S(1,1))') if internal_type == 'formula' else ''
            
            self.obj_table.setItem(row, 2, QTableWidgetItem(freq_text))
            self.obj_table.setItem(row, 3, QTableWidgetItem(formula_text))
            self.obj_table.setItem(row, 4, QTableWidgetItem(str(obj.get('goal', ''))))
            self.obj_table.setItem(row, 5, QTableWidgetItem(obj.get('target', 'minimize')))
            self.obj_table.setItem(row, 6, QTableWidgetItem(str(obj.get('weight', 1.0))))

        algo = cfg.get('algorithm', {})
        self.algorithm_combo.setCurrentText(algo.get('algorithm', 'mobo'))
        self.population_spin.setValue(algo.get('population_size', 30))
        self.generations_spin.setValue(algo.get('n_generations', 30))
        
        # 加载代理模型配置（兼容新旧格式）
        surrogate_type = algo.get('surrogate_type', 'gpflow_svgp')
        self._set_surrogate_type_by_key(surrogate_type)
        
        self.surrogate_check.setChecked(algo.get('use_surrogate', False))
        
        # 加载代理模型参数
        surrogate_config = algo.get('surrogate_config', {})
        
        # 通用参数（兼容旧格式）
        min_samples = surrogate_config.get('min_samples', algo.get('surrogate_min_samples', 5))
        self.min_samples_spin.setValue(min_samples)
        
        uncertainty = surrogate_config.get('uncertainty_threshold', algo.get('surrogate_threshold', 0.5))
        self.uncertainty_spin.setValue(uncertainty)
        
        # 模型专属参数
        model_params = surrogate_config.get('model_params', {})
        if model_params:
            self._set_model_params(model_params)
        
        # 早停配置
        self.earlystop_check.setChecked(algo.get('stop_when_goal_met', True))
        self.earlystop_count_spin.setValue(algo.get('n_solutions_to_stop', 5))
        
        # 可视化配置
        viz = cfg.get('visualization', {})
        self.plot_interval_spin.setValue(viz.get('plot_interval', 5))
        self.surrogate_recent_window_spin.setValue(viz.get('surrogate_recent_window', 5))

        # 更新历史数据状态
        self._update_history_status()
        
        self._loading_ui = False

    def _update_history_status(self):
        """更新历史数据导入状态显示"""
        eval_path = self.config.get('algorithm', {}).get('load_evaluations', '')
        if eval_path:
            display_path = Path(eval_path).name if len(eval_path) > 60 else eval_path
            # 检查是否是本地文件
            local_eval = PROJECT_ROOT / "evaluations.jsonl"
            if local_eval.exists():
                try:
                    count = sum(1 for line in open(local_eval, 'r', encoding='utf-8') if line.strip())
                    self.history_status_label.setText(f"本地 evaluations.jsonl ({count} 条记录)")
                    self.history_status_label.setStyleSheet("font-size: 13px; color: #2ecc71;")
                except Exception:
                    self.history_status_label.setText(f"本地 evaluations.jsonl (无法读取)")
                    self.history_status_label.setStyleSheet("font-size: 13px; color: #e67e22;")
            else:
                self.history_status_label.setText("未导入历史数据")
                self.history_status_label.setStyleSheet("font-size: 13px; color: #a0a0b0;")
                self.config.get('algorithm', {}).pop('load_evaluations', None)
                self._save_config_quiet()
        else:
            self.history_status_label.setText("未导入历史数据")
            self.history_status_label.setStyleSheet("font-size: 13px; color: #a0a0b0;")

    def clear_history(self):
        """清除已导入的历史评估数据（包括本地文件和配置路径）"""
        eval_path = self.config.get('algorithm', {}).get('load_evaluations', '')
        if not eval_path:
            self._show_message("提示", "当前没有关联的历史数据", "info")
            return
        
        reply = QMessageBox.question(
            self, "确认清除",
            f"确定要清除历史评估数据吗?\n\n本地文件: {eval_path}\n\n(清除后需要重新导入才能使用)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # 删除本地 evaluations.jsonl
            local_eval = PROJECT_ROOT / "evaluations.jsonl"
            if local_eval.exists():
                try:
                    local_eval.unlink()
                except Exception:
                    pass
            
            self.config.get('algorithm', {}).pop('load_evaluations', None)
            self._save_config_quiet()
            self._update_history_status()
            self.log("历史评估数据已清除")

    def _show_message(self, title, text, msg_type='info'):
        """显示自定义样式消息框"""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(text)
        
        if msg_type == 'info':
            msg_box.setIcon(QMessageBox.Icon.Information)
        elif msg_type == 'warning':
            msg_box.setIcon(QMessageBox.Icon.Warning)
        elif msg_type == 'error':
            msg_box.setIcon(QMessageBox.Icon.Critical)
        elif msg_type == 'success':
            msg_box.setIcon(QMessageBox.Icon.Information)
        
        msg_box.setStyleSheet("""
            QMessageBox {
                background-color: #1e1e3f;
                color: #ffffff;
            }
            QMessageBox QLabel {
                color: #ffffff;
                font-size: 12px;
            }
            QPushButton {
                background-color: #2980b9;
                color: #ffffff;
                border: none;
                border-radius: 4px;
                padding: 6px 20px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4a9eff;
            }
        """)
        msg_box.exec()

    def import_config(self):
        """导入配置"""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入配置", "", "JSON Files (*.json);;All Files (*)"
        )
        if path:
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                
                # 基本格式验证
                if not isinstance(loaded, dict):
                    self._show_message("错误", "配置文件格式错误：需要 JSON 对象", "error")
                    return
                
                # 保留 load_evaluations 路径（如果已有）
                old_eval_path = self.config.get('algorithm', {}).get('load_evaluations')
                
                self.config = loaded
                
                # 如果之前有导入历史数据但新配置没有，保留旧路径
                if old_eval_path and not self.config.get('algorithm', {}).get('load_evaluations'):
                    self.config.setdefault('algorithm', {})['load_evaluations'] = old_eval_path
                    self.log(f"[INFO] 已保留历史数据路径: {old_eval_path}")
                
                self.load_data_to_ui()
                
                # 检查历史数据
                eval_path = self.config.get('algorithm', {}).get('load_evaluations', '')
                history_info = f"\n已关联历史数据: {eval_path}" if eval_path else ""
                
                self.log(f"配置已导入: {path}{history_info}")
                self._show_message("成功", f"配置导入成功!{history_info}", "success")
            except json.JSONDecodeError as e:
                self._show_message("错误", f"JSON 解析失败: {e}", "error")
            except Exception as e:
                self._show_message("错误", f"导入失败: {e}", "error")

    def import_history(self):
        """导入历史评估数据：直接复制到本地 evaluations.jsonl，后续优化在其基础上追加"""
        path, _ = QFileDialog.getOpenFileName(
            self, "导入历史评估数据", "", "JSONL Files (*.jsonl);;JSON Files (*.json);;All Files (*)"
        )
        if not path:
            return
        
        source_file = Path(path)
        if not source_file.exists():
            self._show_message("错误", "文件不存在!", "error")
            return
        
        try:
            # 解析并验证历史数据
            records = []
            with open(source_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError as e:
                            self._show_message("错误", f"第 {line_num} 行 JSON 解析失败: {e}", "error")
                            return
                        # 验证必要字段
                        if 'parameters' not in record or 'objectives' not in record:
                            self._show_message("错误", f"第 {line_num} 行缺少必要字段 (parameters/objectives)", "error")
                            return
                        records.append(record)
            
            if not records:
                self._show_message("警告", "文件中没有有效的评估数据", "warning")
                return
            
            # 统计目标名称
            obj_names = set()
            for rec in records:
                objs = rec.get('objectives', {})
                if isinstance(objs, dict):
                    obj_names.update(objs.keys())
                elif isinstance(objs, list):
                    for i in range(len(objs)):
                        obj_names.add(f"obj_{i}")
            
            # 保存到本地 evaluations.jsonl（项目根目录下）
            eval_file = PROJECT_ROOT / "evaluations.jsonl"
            with open(eval_file, 'w', encoding='utf-8') as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            
            # 更新配置中的 load_evaluations 指向本地文件
            self.config.setdefault('algorithm', {})['load_evaluations'] = str(eval_file.resolve())
            self._save_config_quiet()
            
            self._update_history_status()
            
            self.log(f"历史评估数据已导入到本地: {eval_file}")
            self.log(f"  记录数: {len(records)}")
            self.log(f"  目标维度: {', '.join(sorted(obj_names))}")
            self.log("")
            self.log("后续优化将在此数据基础上继续扩展，无需重复导入。")
            self._show_message(
                "成功",
                f"已导入 {len(records)} 条历史评估记录!\n\n"
                f"本地文件: evaluations.jsonl\n"
                f"目标维度: {', '.join(sorted(obj_names))}\n\n"
                f"后续优化将在此基础上继续扩展。",
                "success"
            )
        except Exception as e:
            self._show_message("错误", f"导入失败: {e}", "error")

    def _save_config_quiet(self):
        """静默保存配置（不写日志、不弹提示）"""
        try:
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def save_config(self):
        """保存配置"""
        variables = []
        for row in range(self.var_table.rowCount()):
            name_item = self.var_table.item(row, 0)
            if name_item and name_item.text():
                variables.append({
                    'name': name_item.text(),
                    'bounds': [
                        float(self.var_table.item(row, 1).text()) if self.var_table.item(row, 1) else 0,
                        float(self.var_table.item(row, 2).text()) if self.var_table.item(row, 2) else 1
                    ],
                    'unit': self.var_table.item(row, 3).text() if self.var_table.item(row, 3) else 'mm'
                })

        objectives = []
        # GUI类型到内部类型的映射
        type_to_internal = {
            'S参数': 'formula',
            'Gain': 'gain',
            'peakGain': 'peak_gain'
        }
        for row in range(self.obj_table.rowCount()):
            name_item = self.obj_table.item(row, 0)
            if name_item and name_item.text():
                gui_type = self.obj_table.item(row, 1).text() if self.obj_table.item(row, 1) else 'S参数'
                internal_type = type_to_internal.get(gui_type, 'formula')
                obj = {
                    'name': name_item.text(),
                    'type': internal_type,
                    'goal': float(self.obj_table.item(row, 4).text()) if self.obj_table.item(row, 4) else 0,
                    'target': self.obj_table.item(row, 5).text() if self.obj_table.item(row, 5) else 'minimize',
                    'weight': float(self.obj_table.item(row, 6).text()) if self.obj_table.item(row, 6) else 1.0
                }
                freq_text = self.obj_table.item(row, 2).text() if self.obj_table.item(row, 2) else ''
                formula_text = self.obj_table.item(row, 3).text() if self.obj_table.item(row, 3) else ''
                
                if internal_type == 'formula':
                    # S参数类型：解析频段范围，同时保存公式
                    if '-' in freq_text:
                        parts = freq_text.split('-')
                        obj['freq_range'] = [float(parts[0]), float(parts[1])]
                    else:
                        try:
                            obj['freq'] = float(freq_text)
                        except Exception:
                            pass
                    # 公式从第3列获取
                    if formula_text:
                        obj['formula'] = formula_text
                    else:
                        obj['formula'] = 'dB(S(1,1))'
                else:
                    # 其他类型：解析频率范围或单频率
                    if '-' in freq_text:
                        parts = freq_text.split('-')
                        obj['freq_range'] = [float(parts[0]), float(parts[1])]
                    else:
                        try:
                            obj['freq'] = float(freq_text)
                        except Exception:
                            pass
                objectives.append(obj)

        config = {
            'hfss': {
                'project_path': self.project_path_edit.text(),
                'design_name': self.design_name_edit.text(),
                'setup_name': self.setup_name_edit.text(),
                'sweep_name': self.sweep_name_edit.text()
            },
            'variables': variables,
            'objectives': objectives,
            'algorithm': {
                'algorithm': self.algorithm_combo.currentText(),
                'population_size': self.population_spin.value(),
                'n_generations': self.generations_spin.value(),
                # 代理模型配置（新结构）
                'surrogate_type': self._get_surrogate_model_key(),
                'use_surrogate': self.surrogate_check.isChecked(),
                'surrogate_config': {
                    'min_samples': self.min_samples_spin.value(),
                    'uncertainty_threshold': self.uncertainty_spin.value(),
                    'model_params': self._get_model_params()
                },
                # 早停配置
                'stop_when_goal_met': self.earlystop_check.isChecked(),
                'n_solutions_to_stop': self.earlystop_count_spin.value(),
                # 历史数据路径（保留之前导入的路径）
                'load_evaluations': self.config.get('algorithm', {}).get('load_evaluations')
            },
            'run': {'output_dir': str(PROJECT_ROOT / "optim_results")},
            'visualization': {
                'plot_interval': self.plot_interval_spin.value(),
                'surrogate_recent_window': self.surrogate_recent_window_spin.value()
            }
        }

        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)


def main():
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Windows'))

    font = QFont()
    font.setFamily('Microsoft YaHei')
    font.setPointSize(10)
    app.setFont(font)

    window = HFSSOptimizerGUI()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
