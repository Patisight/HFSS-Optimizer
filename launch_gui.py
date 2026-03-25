#!/usr/bin/env python
"""
HFSS 天线优化程序 - 图形界面启动器
解决编辑配置文件无法打开的问题
"""
import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入主GUI
from gui import OptimizerGUI

# 修复后的方法
def fixed_open_results(self):
    """打开结果目录 - 修复版"""
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
    """编辑配置文件 - 修复版"""
    CONFIG_FILE = os.path.join(PROJECT_ROOT, "user_config.json")
    DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "config", "default_config.py")
    
    # 优先打开用户配置文件
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
            # 使用记事本打开
            subprocess.Popen(['notepad', config_path])
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', '-a', 'TextEdit', config_path])
        else:
            subprocess.Popen(['xdg-open', config_path])
    except Exception as e:
        import tkinter.messagebox as messagebox
        messagebox.showerror("错误", f"无法打开配置文件: {e}")

# 替换原始方法
OptimizerGUI.open_results = fixed_open_results
OptimizerGUI.edit_config = fixed_edit_config

if __name__ == "__main__":
    app = OptimizerGUI()
    app.run()