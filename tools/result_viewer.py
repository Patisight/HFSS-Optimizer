#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HFSS 优化结果查看器
独立工具，用于快速检索和筛选优化结果
"""

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ResultViewer(QMainWindow):
    """优化结果查看器"""

    def __init__(self):
        super().__init__()
        self.results_dir = None
        self.config = {}
        self.all_data = []
        self.filtered_data = []
        self.var_names = []
        self.obj_names = []
        self.population_size = 20

        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("HFSS 优化结果查看器")
        self.setGeometry(100, 100, 1400, 800)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # 工具栏
        toolbar = QHBoxLayout()
        self.select_btn = QPushButton("选择结果目录")
        self.select_btn.clicked.connect(self.select_directory)
        toolbar.addWidget(self.select_btn)

        self.dir_label = QLabel("未选择目录")
        self.dir_label.setStyleSheet("color: gray;")
        toolbar.addWidget(self.dir_label)
        toolbar.addStretch()

        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.reload_data)
        self.refresh_btn.setEnabled(False)
        toolbar.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("导出")
        self.export_btn.clicked.connect(self.export_selected)
        self.export_btn.setEnabled(False)
        toolbar.addWidget(self.export_btn)

        layout.addLayout(toolbar)

        # 主内容
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧筛选
        filter_panel = self.create_filter_panel()
        splitter.addWidget(filter_panel)

        # 右侧表格
        self.table = self.create_table()
        splitter.addWidget(self.table)
        splitter.setStretchFactor(1, 4)

        layout.addWidget(splitter)

        # 状态栏
        self.status_label = QLabel("未加载数据")
        layout.addWidget(self.status_label)

    def create_filter_panel(self) -> QWidget:
        """创建筛选面板"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)

        container = QWidget()
        layout = QVBoxLayout(container)

        # 信息
        info = QGroupBox("结果信息")
        info_layout = QFormLayout()
        self.info_name = QLabel("-")
        self.info_evals = QLabel("-")
        self.info_vars = QLabel("-")
        self.info_objs = QLabel("-")
        info_layout.addRow("名称:", self.info_name)
        info_layout.addRow("评估数:", self.info_evals)
        info_layout.addRow("变量数:", self.info_vars)
        info_layout.addRow("目标数:", self.info_objs)
        info.setLayout(info_layout)
        layout.addWidget(info)

        # 筛选
        filter_grp = QGroupBox("筛选条件")
        filter_layout = QVBoxLayout()

        # 仿真编号
        iter_layout = QHBoxLayout()
        iter_layout.addWidget(QLabel("仿真编号:"))
        self.iter_min = QSpinBox()
        self.iter_min.setRange(0, 99999)
        iter_layout.addWidget(self.iter_min)
        iter_layout.addWidget(QLabel("~"))
        self.iter_max = QSpinBox()
        self.iter_max.setRange(0, 99999)
        self.iter_max.setValue(99999)
        iter_layout.addWidget(self.iter_max)
        filter_layout.addLayout(iter_layout)

        # 目标筛选
        self.obj_filters = QVBoxLayout()
        filter_layout.addLayout(self.obj_filters)

        filter_grp.setLayout(filter_layout)
        layout.addWidget(filter_grp)

        # 按钮
        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("应用")
        self.apply_btn.clicked.connect(self.apply_filter)
        self.apply_btn.setEnabled(False)
        btn_layout.addWidget(self.apply_btn)

        self.clear_btn = QPushButton("重置")
        self.clear_btn.clicked.connect(self.clear_filter)
        self.clear_btn.setEnabled(False)
        btn_layout.addWidget(self.clear_btn)
        layout.addLayout(btn_layout)

        layout.addStretch()
        scroll.setWidget(container)
        return scroll

    def create_table(self) -> QTableWidget:
        """创建表格"""
        table = QTableWidget()
        table.setAlternatingRowColors(True)
        table.setSortingEnabled(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        return table

    def select_directory(self):
        """选择目录"""
        dir_path = QFileDialog.getExistingDirectory(self, "选择结果目录", str(Path.home()))
        if dir_path:
            self.results_dir = dir_path
            self.dir_label.setText(os.path.basename(dir_path))
            self.dir_label.setStyleSheet("color: black;")
            self.load_results()

    def load_results(self):
        """加载数据"""
        if not self.results_dir:
            return

        self.status_label.setText("正在加载...")
        QApplication.processEvents()

        try:
            # 加载配置
            config_file = os.path.join(self.results_dir, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            else:
                self.config = {}

            # 获取变量名
            self.var_names = []
            if "variables" in self.config:
                for v in self.config["variables"]:
                    self.var_names.append(v["name"])

            # 获取种群大小
            self.population_size = self.config.get("algorithm", {}).get("population_size", 20)

            # 加载评估数据
            eval_files = glob.glob(os.path.join(self.results_dir, "evaluations.jsonl"))
            if not eval_files:
                eval_files = glob.glob(os.path.join(self.results_dir, "*.jsonl"))

            if not eval_files:
                QMessageBox.warning(self, "提示", "未找到评估文件")
                self.status_label.setText("未找到文件")
                return

            self.all_data = []
            self.obj_names = set()

            for eval_file in eval_files:
                with open(eval_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            self.all_data.append(record)

                            # 获取目标名
                            objectives = record.get("objectives", {})
                            for name in objectives.keys():
                                self.obj_names.add(name)
                        except json.JSONDecodeError:
                            continue

            self.obj_names = sorted(list(self.obj_names))

            if not self.all_data:
                QMessageBox.warning(self, "提示", "数据为空")
                self.status_label.setText("数据为空")
                return

            # 更新UI
            self.update_info()
            self.update_filters()
            self.filtered_data = self.all_data.copy()
            self.display_data()

            self.apply_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)
            self.refresh_btn.setEnabled(True)
            self.export_btn.setEnabled(True)

            self.status_label.setText(f"已加载 {len(self.all_data)} 条记录")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载失败:\n{str(e)}")
            self.status_label.setText("加载失败")

    def reload_data(self):
        """重新加载"""
        if self.results_dir:
            self.load_results()

    def update_info(self):
        """更新信息"""
        self.info_name.setText(os.path.basename(self.results_dir))
        self.info_evals.setText(str(len(self.all_data)))
        self.info_vars.setText(str(len(self.var_names)))
        self.info_objs.setText(str(len(self.obj_names)))

        # 更新仿真编号范围
        if self.all_data:
            eval_ids = [r.get("eval_id", 0) for r in self.all_data]
            max_id = max(eval_ids)
            self.iter_min.setRange(1, max_id)
            self.iter_max.setRange(1, max_id)
            self.iter_max.setValue(max_id)

    def update_filters(self):
        """更新筛选控件"""
        # 清除旧控件
        while self.obj_filters.count():
            item = self.obj_filters.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 添加目标筛选
        for obj_name in self.obj_names:
            grp = QGroupBox(obj_name)
            grp_layout = QFormLayout()

            combo = QComboBox()
            combo.addItems(["不过滤", "大于", "小于", "大于等于", "小于等于", "介于"])
            combo.currentTextChanged.connect(lambda t, g=grp: self._update_val2_visibility(g))
            grp_layout.addRow("条件:", combo)

            val1 = QDoubleSpinBox()
            val1.setDecimals(3)
            val1.setRange(-99999, 99999)
            grp_layout.addRow("值:", val1)

            val2 = QDoubleSpinBox()
            val2.setDecimals(3)
            val2.setRange(-99999, 99999)
            val2_label = QLabel("至")
            val2_layout = QHBoxLayout()
            val2_layout.addWidget(val2_label)
            val2_layout.addWidget(val2)
            val2_layout.addStretch()
            grp_layout.addRow(" ", val2_layout)
            val2_label.setVisible(False)
            val2.setVisible(False)

            grp.setLayout(grp_layout)
            self.obj_filters.addWidget(grp)

            # 存储引用
            grp.combo = combo
            grp.val1 = val1
            grp.val2 = val2
            grp.val2_label = val2_label
            grp.val2_layout = val2_layout
            grp.obj_name = obj_name

    def _update_val2_visibility(self, grp):
        """根据条件显示/隐藏第二个值"""
        cond = grp.combo.currentText()
        if cond == "介于":
            grp.val2_label.setVisible(True)
            grp.val2.setVisible(True)
        else:
            grp.val2_label.setVisible(False)
            grp.val2.setVisible(False)

    def apply_filter(self):
        """应用筛选"""
        filtered = []

        iter_min = self.iter_min.value()
        iter_max = self.iter_max.value()

        for record in self.all_data:
            # 检查仿真编号
            eval_id = record.get("eval_id", 0)
            if eval_id < iter_min or eval_id > iter_max:
                continue

            # 检查目标
            objectives = record.get("objectives", {})

            valid = True
            for i in range(self.obj_filters.count()):
                grp = self.obj_filters.itemAt(i).widget()
                if not isinstance(grp, QGroupBox):
                    continue

                cond = grp.combo.currentText()
                if cond == "不过滤":
                    continue

                obj_name = grp.obj_name
                if obj_name not in objectives:
                    valid = False
                    break

                value = objectives[obj_name].get("value", 0)
                v1 = grp.val1.value()
                v2 = grp.val2.value()

                if cond == "大于" and not (value > v1):
                    valid = False
                    break
                elif cond == "小于" and not (value < v1):
                    valid = False
                    break
                elif cond == "大于等于" and not (value >= v1):
                    valid = False
                    break
                elif cond == "小于等于" and not (value <= v1):
                    valid = False
                    break
                elif cond == "介于" and not (v1 <= value <= v2):
                    valid = False
                    break

            if valid:
                filtered.append(record)

        self.filtered_data = filtered
        self.display_data()
        self.status_label.setText(f"筛选: {len(self.filtered_data)} / {len(self.all_data)} 条")

    def clear_filter(self):
        """重置筛选"""
        self.iter_max.setValue(self.iter_max.maximum())

        for i in range(self.obj_filters.count()):
            grp = self.obj_filters.itemAt(i).widget()
            if isinstance(grp, QGroupBox):
                grp.combo.setCurrentIndex(0)
                grp.val1.setValue(0)
                grp.val2.setValue(0)

        self.filtered_data = self.all_data.copy()
        self.display_data()
        self.status_label.setText(f"已加载 {len(self.all_data)} 条记录")

    def display_data(self):
        """显示数据"""
        self.table.clear()

        if not self.filtered_data:
            self.table.setRowCount(0)
            self.table.setColumnCount(0)
            return

        # 表头: # | 参数1 | 参数2 | ... | 目标1 | 目标2 | ...
        headers = ["#"] + self.var_names + self.obj_names
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)

        self.table.setRowCount(len(self.filtered_data))

        for row, record in enumerate(self.filtered_data):
            # 仿真编号
            self.table.setItem(row, 0, QTableWidgetItem(str(record.get("eval_id", "-"))))

            # 参数值
            params = record.get("parameters", [])
            for col, var_name in enumerate(self.var_names, 1):
                if col - 1 < len(params):
                    val = params[col - 1]
                    self.table.setItem(row, col, QTableWidgetItem(f"{val:.4f}"))
                else:
                    self.table.setItem(row, col, QTableWidgetItem("-"))

            # 目标值
            objectives = record.get("objectives", {})
            for col, obj_name in enumerate(self.obj_names, 1 + len(self.var_names)):
                if obj_name in objectives:
                    val = objectives[obj_name].get("value", 0)
                    self.table.setItem(row, col, QTableWidgetItem(f"{val:.4f}"))
                else:
                    self.table.setItem(row, col, QTableWidgetItem("-"))

        self.table.resizeColumnsToContents()

        # 按仿真编号排序
        self.table.sortByColumn(0, Qt.SortOrder.AscendingOrder)

    def export_selected(self):
        """导出选中"""
        selected = set(item.row() for item in self.table.selectedItems())
        if not selected:
            QMessageBox.information(self, "提示", "请选择要导出的行")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "导出", f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "JSON (*.json)"
        )
        if not path:
            return

        try:
            data = [self.filtered_data[row] for row in selected]
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "成功", f"已导出 {len(data)} 条")
        except Exception as e:
            QMessageBox.critical(self, "错误", str(e))


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = ResultViewer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
