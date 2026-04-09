#!/usr/bin/env python
"""
HFSS 天线优化程序 - 图形界面启动器

支持两种UI:
- PyQt6 (推荐): python launch_gui.py --ui pyqt6
- Tkinter: python launch_gui.py --ui tkinter

默认使用 PyQt6
"""

import os
import sys
import argparse
import subprocess
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def check_dependencies():
    """检查依赖"""
    missing = []
    
    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    return missing


def launch_pyqt6():
    """启动PyQt6版本"""
    missing = check_dependencies()
    if missing:
        print(f"[ERROR] Missing dependencies: {', '.join(missing)}")
        print("Run: pip install " + " ".join(missing))
        sys.exit(1)
    
    from gui_pyqt6 import main
    main()


def launch_tkinter():
    """启动Tkinter版本"""
    from gui import OptimizerGUI

    def fixed_open_results(self):
        results_dir = os.path.join(PROJECT_ROOT, "optim_results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        try:
            if sys.platform == 'win32':
                subprocess.run(['explorer', results_dir])
            elif sys.platform == 'darwin':
                subprocess.run(['open', results_dir])
            else:
                subprocess.run(['xdg-open', results_dir])
        except Exception as e:
            import tkinter.messagebox as messagebox
            messagebox.showerror("错误", f"无法打开目录: {e}")

    def fixed_edit_config(self):
        CONFIG_FILE = os.path.join(PROJECT_ROOT, "user_config.json")
        DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "default_config.py")
        if os.path.exists(CONFIG_FILE):
            config_path = CONFIG_FILE
        else:
            config_path = DEFAULT_CONFIG
        if not os.path.exists(config_path):
            import tkinter.messagebox as messagebox
            messagebox.showwarning("提示", "配置文件不存在")
            return
        try:
            if sys.platform == 'win32':
                subprocess.Popen(['notepad', config_path])
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', '-a', 'TextEdit', config_path])
            else:
                subprocess.Popen(['xdg-open', config_path])
        except Exception as e:
            import tkinter.messagebox as messagebox
            messagebox.showerror("错误", f"无法打开配置文件: {e}")

    OptimizerGUI.open_results = fixed_open_results
    OptimizerGUI.edit_config = fixed_edit_config

    app = OptimizerGUI()
    app.run()


def main():
    try:
        parser = argparse.ArgumentParser(description='HFSS天线优化程序启动器')
        parser.add_argument('--ui', choices=['pyqt6', 'tkinter'], default='pyqt6',
                            help='选择UI类型 (默认: pyqt6)')
        args = parser.parse_args()

        if args.ui == 'pyqt6':
            launch_pyqt6()
        else:
            launch_tkinter()
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()
