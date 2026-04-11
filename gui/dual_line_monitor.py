"""
双线状态监控GUI模块

提供实时监控界面，显示：
1. 优化线状态（迭代次数、真实/代理评估次数、模型版本）
2. 训练线状态（样本数、模型版本、模型质量）
3. 模型质量趋势图
4. 热替换日志
"""

import json
import os

# 添加项目路径
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import scrolledtext, ttk
from typing import Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from core.shared_memory import SharedMemoryManager


class DualLineMonitorFrame(ttk.Frame):
    """
    双线状态监控框架

    显示优化线和训练线的实时状态。
    """

    def __init__(self, parent, shared_dir: str = "./shared_data"):
        """
        初始化监控框架

        Args:
            parent: 父窗口
            shared_dir: 共享数据目录
        """
        super().__init__(parent)

        self.shared_dir = shared_dir
        self.shared_memory = SharedMemoryManager(shared_dir)

        # 状态更新线程
        self._update_thread = None
        self._running = False

        # 数据历史（用于绘图）
        self.quality_history = []
        self.swap_history = []

        # 创建UI
        self._create_ui()

        # 启动更新
        self.start_update()

    def _create_ui(self):
        """创建UI组件"""
        # 主布局：上下两部分
        # 上部：状态面板
        # 下部：图表和日志

        # ===== 上部：状态面板 =====
        status_frame = ttk.LabelFrame(self, text="双线状态", padding=10)
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        # 优化线状态
        optimizer_frame = ttk.LabelFrame(status_frame, text="优化线", padding=5)
        optimizer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.optimizer_status_var = tk.StringVar(value="状态: 空闲")
        self.optimizer_iteration_var = tk.StringVar(value="迭代: 0")
        self.optimizer_real_evals_var = tk.StringVar(value="真实评估: 0")
        self.optimizer_surrogate_evals_var = tk.StringVar(value="代理评估: 0")
        self.optimizer_model_version_var = tk.StringVar(value="模型版本: 0")

        ttk.Label(optimizer_frame, textvariable=self.optimizer_status_var, font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(optimizer_frame, textvariable=self.optimizer_iteration_var).pack(anchor=tk.W)
        ttk.Label(optimizer_frame, textvariable=self.optimizer_real_evals_var).pack(anchor=tk.W)
        ttk.Label(optimizer_frame, textvariable=self.optimizer_surrogate_evals_var).pack(anchor=tk.W)
        ttk.Label(optimizer_frame, textvariable=self.optimizer_model_version_var).pack(anchor=tk.W)

        # 训练线状态
        trainer_frame = ttk.LabelFrame(status_frame, text="训练线", padding=5)
        trainer_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.trainer_status_var = tk.StringVar(value="状态: 空闲")
        self.trainer_samples_var = tk.StringVar(value="样本数: 0")
        self.trainer_model_version_var = tk.StringVar(value="模型版本: 0")
        self.trainer_model_quality_var = tk.StringVar(value="模型质量: N/A")

        ttk.Label(trainer_frame, textvariable=self.trainer_status_var, font=("Arial", 10, "bold")).pack(anchor=tk.W)
        ttk.Label(trainer_frame, textvariable=self.trainer_samples_var).pack(anchor=tk.W)
        ttk.Label(trainer_frame, textvariable=self.trainer_model_version_var).pack(anchor=tk.W)
        ttk.Label(trainer_frame, textvariable=self.trainer_model_quality_var).pack(anchor=tk.W)

        # ===== 下部：图表和日志 =====
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧：模型质量趋势图
        chart_frame = ttk.LabelFrame(bottom_frame, text="模型质量趋势", padding=5)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("训练次数")
        self.ax.set_ylabel("R²")
        self.ax.set_title("模型质量趋势")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 右侧：热替换日志
        log_frame = ttk.LabelFrame(bottom_frame, text="热替换日志", padding=5)
        log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, width=40, height=15, font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 清除按钮
        ttk.Button(log_frame, text="清除日志", command=self._clear_log).pack(pady=5)

    def start_update(self):
        """启动状态更新"""
        if self._running:
            return

        self._running = True
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

        self._log("监控已启动")

    def stop_update(self):
        """停止状态更新"""
        self._running = False
        if self._update_thread is not None:
            self._update_thread.join(timeout=2.0)

        self._log("监控已停止")

    def _update_loop(self):
        """状态更新循环"""
        last_optimizer_version = 0
        last_trainer_version = 0

        while self._running:
            try:
                # 获取状态
                status = self.shared_memory.get_full_status()

                optimizer_status = status.get("optimizer", {})
                trainer_status = status.get("trainer", {})

                # 更新优化线状态
                self._update_optimizer_status(optimizer_status)

                # 更新训练线状态
                self._update_trainer_status(trainer_status)

                # 检查模型版本变化（热替换）
                current_optimizer_version = optimizer_status.get("model_version", 0)
                current_trainer_version = trainer_status.get("model_version", 0)

                if current_optimizer_version > last_optimizer_version:
                    self._log(f"✅ 模型热替换: v{last_optimizer_version} -> v{current_optimizer_version}")
                    last_optimizer_version = current_optimizer_version

                if current_trainer_version > last_trainer_version:
                    # 更新质量历史
                    model_quality = trainer_status.get("model_quality", {})
                    if model_quality and model_quality.get("r2") is not None:
                        self.quality_history.append(
                            {
                                "version": current_trainer_version,
                                "r2": model_quality.get("r2"),
                                "mae": model_quality.get("mae"),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                        # 更新图表
                        self._update_chart()

                    last_trainer_version = current_trainer_version

            except Exception as e:
                self._log(f"[ERROR] 更新失败: {e}")

            time.sleep(1.0)

    def _update_optimizer_status(self, status: Dict):
        """更新优化线状态显示"""
        status_text = status.get("status", "unknown")

        # 状态颜色映射
        status_map = {"idle": "空闲", "running": "运行中", "paused": "已暂停", "stopped": "已停止"}

        self.optimizer_status_var.set(f"状态: {status_map.get(status_text, status_text)}")
        self.optimizer_iteration_var.set(f"迭代: {status.get('iteration', 0)}")
        self.optimizer_real_evals_var.set(f"真实评估: {status.get('n_real_evals', 0)}")
        self.optimizer_surrogate_evals_var.set(f"代理评估: {status.get('n_surrogate_evals', 0)}")
        self.optimizer_model_version_var.set(f"模型版本: {status.get('model_version', 0)}")

    def _update_trainer_status(self, status: Dict):
        """更新训练线状态显示"""
        status_text = status.get("status", "unknown")

        # 状态颜色映射
        status_map = {
            "idle": "空闲",
            "waiting_data": "等待数据",
            "collecting": "收集数据",
            "training": "训练中",
            "model_ready": "模型就绪",
            "error": "错误",
        }

        self.trainer_status_var.set(f"状态: {status_map.get(status_text, status_text)}")
        self.trainer_samples_var.set(f"样本数: {status.get('n_samples', 0)}")
        self.trainer_model_version_var.set(f"模型版本: {status.get('model_version', 0)}")

        # 模型质量
        model_quality = status.get("model_quality", {})
        if model_quality and model_quality.get("r2") is not None:
            r2 = model_quality.get("r2", 0)
            mae = model_quality.get("mae", 0)
            self.trainer_model_quality_var.set(f"模型质量: R²={r2:.3f}, MAE={mae:.3f}")
        else:
            self.trainer_model_quality_var.set("模型质量: N/A")

    def _update_chart(self):
        """更新模型质量趋势图"""
        if not self.quality_history:
            return

        # 清空图表
        self.ax.clear()

        # 提取数据
        versions = [h["version"] for h in self.quality_history]
        r2_values = [h["r2"] for h in self.quality_history]

        # 绘制
        self.ax.plot(versions, r2_values, "b-o", linewidth=2, markersize=6)
        self.ax.set_xlabel("模型版本")
        self.ax.set_ylabel("R²")
        self.ax.set_title("模型质量趋势")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_ylim([0, 1])

        # 刷新画布
        self.canvas.draw()

    def _log(self, message: str):
        """添加日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)

    def _clear_log(self):
        """清除日志"""
        self.log_text.delete(1.0, tk.END)
        self._log("日志已清除")

    def destroy(self):
        """销毁时清理"""
        self.stop_update()
        super().destroy()


class DualLineMonitorWindow(tk.Tk):
    """
    独立的双线监控窗口

    用于测试和调试。
    """

    def __init__(self, shared_dir: str = "./shared_data"):
        """初始化窗口"""
        super().__init__()

        self.title("双线状态监控")
        self.geometry("1000x600")

        # 创建监控框架
        self.monitor_frame = DualLineMonitorFrame(self, shared_dir)
        self.monitor_frame.pack(fill=tk.BOTH, expand=True)

        # 关闭时清理
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """关闭窗口"""
        self.monitor_frame.destroy()
        self.destroy()


def main():
    """测试入口"""
    import argparse

    parser = argparse.ArgumentParser(description="Dual-Line Monitor")
    parser.add_argument("--shared-dir", type=str, default="./shared_data", help="Shared data directory")
    args = parser.parse_args()

    app = DualLineMonitorWindow(args.shared_dir)
    app.mainloop()


if __name__ == "__main__":
    main()
