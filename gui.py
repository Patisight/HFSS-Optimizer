#!/usr/bin/env python
"""
HFSS 天线优化程序 - 图形界面 v3
支持滚动、可视化编辑变量和目标
"""
import os
import sys
import json
import threading
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from datetime import datetime
from typing import Dict

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 用户配置文件路径
CONFIG_FILE = os.path.join(PROJECT_ROOT, "user_config.json")


class ScrollableFrame(ttk.Frame):
    """可滚动的 Frame"""
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        
        # 创建 Canvas
        self.canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0)
        
        # 创建滚动条
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        
        # 可滚动的内部 Frame
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # 当内部 Frame 大小改变时更新滚动区域
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        # 在 Canvas 中创建窗口
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # 配置 Canvas 滚动
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 布局
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        
        # 鼠标滚轮支持
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind('<Enter>', self._bind_mousewheel)
        self.canvas.bind('<Leave>', self._unbind_mousewheel)
    
    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
    
    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")


class OptimizerGUI:
    """优化程序图形界面"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HFSS 天线优化程序 v3")
        self.root.geometry("1000x750")
        
        # 加载默认配置
        self.config = self.load_config()
        
        # 运行状态
        self.running = False
        self.process = None
        
        # 变量和目标列表
        self.variables_data = []
        self.objectives_data = []
        
        # 创建界面
        self.create_widgets()
        
        # 加载数据到界面
        self.load_data_to_ui()
    
    def load_config(self):
        """加载配置"""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        try:
            from config import get_default_config
            return get_default_config()
        except:
            return {
                'hfss': {'project_path': '', 'design_name': '', 'setup_name': 'Setup1', 'sweep_name': 'Sweep'},
                'variables': [],
                'objectives': [],
                'algorithm': {'population_size': 30, 'n_generations': 15, 'initial_samples': 80}
            }
    
    def save_config(self):
        """保存配置"""
        algo = self.algorithm_var.get()
        
        # 基础算法配置
        algo_config = {
            'algorithm': algo,
            'population_size': self.population_var.get(),
            'n_generations': self.generations_var.get(),
            'use_surrogate': self.use_surrogate_var.get(),
            'surrogate_type': self.surrogate_var.get(),
        }
        
        # 根据算法添加特定参数
        if algo == "mobo":
            algo_config['initial_samples'] = self.population_var.get()
            algo_config['n_iterations'] = self.generations_var.get()
            algo_config['acquisition'] = self.acquisition_var.get()
        elif algo == "mopso":
            algo_config['population_size'] = self.population_var.get()
            algo_config['n_generations'] = self.generations_var.get()
            algo_config['inertia_weight'] = self.inertia_var.get()
            algo_config['c1'] = self.c1_var.get()
            algo_config['c2'] = self.c1_var.get()  # 通常 c1 = c2
            # 代理模型参数
            algo_config['use_surrogate'] = self.use_surrogate_var.get()
            algo_config['surrogate_type'] = self.surrogate_var.get()
            algo_config['surrogate_min_samples'] = self.surrogate_min_samples_var.get()
            algo_config['surrogate_threshold'] = self.surrogate_threshold_var.get()
            # 加载历史评估数据
            load_eval = self.load_eval_var.get().strip()
            if load_eval:
                algo_config['load_evaluations'] = load_eval
        elif algo == "nsga2":
            algo_config['population_size'] = self.population_var.get()
            algo_config['n_generations'] = self.generations_var.get()
            algo_config['mutation_prob'] = self.mutation_var.get()
            algo_config['crossover_prob'] = self.crossover_var.get()
        
        config = {
            'hfss': {
                'project_path': self.project_path_var.get(),
                'design_name': self.design_name_var.get(),
                'setup_name': self.setup_name_var.get(),
                'sweep_name': self.sweep_name_var.get(),
            },
            'variables': self.variables_data,
            'objectives': self.objectives_data,
            'algorithm': algo_config,
            'visualization': {
                'plot_interval': self.plot_interval_var.get()
            },
            'run': {'output_dir': os.path.join(PROJECT_ROOT, "optim_results")}
        }
        
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        self.log("配置已保存")
    
    def _select_eval_file(self):
        """选择历史评估文件"""
        file_path = filedialog.askopenfilename(
            title="选择历史评估数据文件",
            filetypes=[("JSON Lines 文件", "*.jsonl"), ("所有文件", "*.*")],
            initialdir=os.path.join(PROJECT_ROOT, "optim_results")
        )
        if file_path:
            self.load_eval_var.set(file_path)
            self.log(f"已选择历史数据: {os.path.basename(file_path)}")
    
    def create_widgets(self):
        """创建界面组件"""
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 项目配置页
        project_frame = ttk.Frame(notebook, padding="5")
        notebook.add(project_frame, text="项目配置")
        self.create_project_tab(project_frame)
        
        # 变量配置页
        variables_frame = ttk.Frame(notebook, padding="5")
        notebook.add(variables_frame, text="优化变量")
        self.create_variables_tab(variables_frame)
        
        # 目标配置页
        objectives_frame = ttk.Frame(notebook, padding="5")
        notebook.add(objectives_frame, text="优化目标")
        self.create_objectives_tab(objectives_frame)
        
        # 算法配置页（可滚动）
        algo_container = ttk.Frame(notebook, padding="5")
        notebook.add(algo_container, text="算法配置")
        self.create_algorithm_tab(algo_container)
        
        # 运行页
        run_frame = ttk.Frame(notebook, padding="5")
        notebook.add(run_frame, text="运行")
        self.create_run_tab(run_frame)
    
    # ==================== 项目配置页 ====================
    def create_project_tab(self, parent):
        hfss_frame = ttk.LabelFrame(parent, text="HFSS 项目配置", padding="10")
        hfss_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(hfss_frame, text="项目路径:").grid(row=0, column=0, sticky=tk.W, pady=3)
        self.project_path_var = tk.StringVar(value=self.config['hfss'].get('project_path', ''))
        ttk.Entry(hfss_frame, textvariable=self.project_path_var, width=60).grid(row=0, column=1, pady=3, padx=5)
        ttk.Button(hfss_frame, text="浏览...", command=self.browse_project).grid(row=0, column=2, pady=3)
        
        ttk.Label(hfss_frame, text="设计名称:").grid(row=1, column=0, sticky=tk.W, pady=3)
        self.design_name_var = tk.StringVar(value=self.config['hfss'].get('design_name', ''))
        ttk.Entry(hfss_frame, textvariable=self.design_name_var, width=30).grid(row=1, column=1, sticky=tk.W, pady=3, padx=5)
        
        ttk.Label(hfss_frame, text="Setup:").grid(row=2, column=0, sticky=tk.W, pady=3)
        self.setup_name_var = tk.StringVar(value=self.config['hfss'].get('setup_name', 'Setup1'))
        ttk.Entry(hfss_frame, textvariable=self.setup_name_var, width=20).grid(row=2, column=1, sticky=tk.W, pady=3, padx=5)
        
        ttk.Label(hfss_frame, text="Sweep:").grid(row=3, column=0, sticky=tk.W, pady=3)
        self.sweep_name_var = tk.StringVar(value=self.config['hfss'].get('sweep_name', 'Sweep'))
        ttk.Entry(hfss_frame, textvariable=self.sweep_name_var, width=20).grid(row=3, column=1, sticky=tk.W, pady=3, padx=5)
        
        # 按钮区域
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=5)
        
        # 导出/导入配置
        ttk.Separator(btn_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(btn_frame, text="导出配置", command=self.export_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="导入配置", command=self.import_config).pack(side=tk.LEFT, padx=5)
        
        # 配置说明
        info_frame = ttk.LabelFrame(parent, text="配置说明", padding="5")
        info_frame.pack(fill=tk.X, pady=5)
        
        info_text = """配置文件保存位置: user_config.json

导出配置: 将当前配置保存为 JSON 文件，可备份或分享给他人
导入配置: 从 JSON 文件加载配置，快速恢复设置

配置内容包括:
• HFSS 项目路径
• 优化变量（名称、范围、单位）
• 优化目标（类型、频率、目标值）
• 算法参数"""
        
        ttk.Label(info_frame, text=info_text, justify=tk.LEFT, foreground="gray").pack(anchor=tk.W)
    
    def export_config(self):
        """导出配置到文件"""
        # 先保存当前配置
        self.save_config()
        
        # 选择保存位置
        file_path = filedialog.asksaveasfilename(
            title="导出配置",
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
            initialfile="hfss_optimizer_config.json"
        )
        
        if file_path:
            try:
                import shutil
                shutil.copy(CONFIG_FILE, file_path)
                messagebox.showinfo("成功", f"配置已导出到:\n{file_path}")
                self.log(f"配置已导出: {file_path}")
            except Exception as e:
                messagebox.showerror("错误", f"导出失败: {e}")
    
    def import_config(self):
        """从文件导入配置"""
        file_path = filedialog.askopenfilename(
            title="导入配置",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    new_config = json.load(f)
                
                # 验证配置
                if 'variables' not in new_config or 'objectives' not in new_config:
                    messagebox.showwarning("警告", "配置文件格式不正确")
                    return
                
                # 备份当前配置
                import shutil
                backup_path = CONFIG_FILE + ".backup"
                if os.path.exists(CONFIG_FILE):
                    shutil.copy(CONFIG_FILE, backup_path)
                
                # 保存新配置
                with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                    json.dump(new_config, f, indent=2, ensure_ascii=False)
                
                # 重新加载配置
                self.config = new_config
                
                # 清空并重新加载界面
                for item in self.var_tree.get_children():
                    self.var_tree.delete(item)
                for item in self.obj_tree.get_children():
                    self.obj_tree.delete(item)
                
                self.load_data_to_ui()
                
                # 加载 HFSS 配置
                hfss = new_config.get('hfss', {})
                self.project_path_var.set(hfss.get('project_path', ''))
                self.design_name_var.set(hfss.get('design_name', ''))
                self.setup_name_var.set(hfss.get('setup_name', 'Setup1'))
                self.sweep_name_var.set(hfss.get('sweep_name', 'Sweep'))
                
                messagebox.showinfo("成功", f"配置已导入:\n{file_path}")
                self.log(f"配置已导入: {file_path}")
                
            except Exception as e:
                messagebox.showerror("错误", f"导入失败: {e}")
    
    # ==================== 变量配置页 ====================
    def create_variables_tab(self, parent):
        ttk.Label(parent, text="定义优化变量（变量名必须与 HFSS 项目中一致）", foreground="gray").pack(anchor=tk.W)
        
        # 变量列表
        columns = ('name', 'min', 'max', 'unit')
        self.var_tree = ttk.Treeview(parent, columns=columns, show='headings', height=8)
        for col, text, width in [('name', '变量名', 120), ('min', '最小值', 80), ('max', '最大值', 80), ('unit', '单位', 60)]:
            self.var_tree.heading(col, text=text)
            self.var_tree.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.var_tree.yview)
        self.var_tree.configure(yscrollcommand=scrollbar.set)
        self.var_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 编辑框
        edit_frame = ttk.LabelFrame(parent, text="添加/修改变量", padding="5")
        edit_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(edit_frame, text="名称:").grid(row=0, column=0, padx=2)
        self.var_name = ttk.Entry(edit_frame, width=10)
        self.var_name.grid(row=0, column=1, padx=2)
        
        ttk.Label(edit_frame, text="最小:").grid(row=0, column=2, padx=2)
        self.var_min = ttk.Entry(edit_frame, width=8)
        self.var_min.grid(row=0, column=3, padx=2)
        
        ttk.Label(edit_frame, text="最大:").grid(row=0, column=4, padx=2)
        self.var_max = ttk.Entry(edit_frame, width=8)
        self.var_max.grid(row=0, column=5, padx=2)
        
        ttk.Label(edit_frame, text="单位:").grid(row=0, column=6, padx=2)
        self.var_unit = ttk.Combobox(edit_frame, values=['mm', 'GHz', 'nH', 'pF', 'deg', ''], width=6)
        self.var_unit.set('mm')
        self.var_unit.grid(row=0, column=7, padx=2)
        
        btn_frame = ttk.Frame(edit_frame)
        btn_frame.grid(row=1, column=0, columnspan=8, pady=5)
        ttk.Button(btn_frame, text="添加", command=self.add_variable).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="修改", command=self.update_variable).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="删除", command=self.delete_variable).pack(side=tk.LEFT, padx=2)
        
        self.var_tree.bind('<<TreeviewSelect>>', self.on_variable_select)
    
    def add_variable(self):
        name, min_v, max_v, unit = self.var_name.get().strip(), self.var_min.get().strip(), self.var_max.get().strip(), self.var_unit.get()
        if name and min_v and max_v:
            try:
                self.var_tree.insert('', tk.END, values=(name, float(min_v), float(max_v), unit))
                self.update_variables_data()
                self.var_name.delete(0, tk.END)
                self.var_min.delete(0, tk.END)
                self.var_max.delete(0, tk.END)
            except ValueError:
                messagebox.showwarning("警告", "数值格式错误")
    
    def update_variable(self):
        selected = self.var_tree.selection()
        if selected:
            name, min_v, max_v, unit = self.var_name.get().strip(), self.var_min.get().strip(), self.var_max.get().strip(), self.var_unit.get()
            try:
                self.var_tree.item(selected[0], values=(name, float(min_v), float(max_v), unit))
                self.update_variables_data()
            except ValueError:
                messagebox.showwarning("警告", "数值格式错误")
    
    def delete_variable(self):
        for item in self.var_tree.selection():
            self.var_tree.delete(item)
        self.update_variables_data()
    
    def on_variable_select(self, event):
        if self.var_tree.selection():
            values = self.var_tree.item(self.var_tree.selection()[0], 'values')
            self.var_name.delete(0, tk.END); self.var_name.insert(0, values[0])
            self.var_min.delete(0, tk.END); self.var_min.insert(0, values[1])
            self.var_max.delete(0, tk.END); self.var_max.insert(0, values[2])
            self.var_unit.set(values[3])
    
    def update_variables_data(self):
        self.variables_data = []
        for item in self.var_tree.get_children():
            v = self.var_tree.item(item, 'values')
            self.variables_data.append({'name': v[0], 'bounds': (float(v[1]), float(v[2])), 'unit': v[3]})
    
    # ==================== 目标配置页 ====================
    def create_objectives_tab(self, parent):
        ttk.Label(parent, text="定义优化目标", foreground="gray").pack(anchor=tk.W)
        
        columns = ('name', 'type', 'freq', 'goal', 'target', 'weight')
        self.obj_tree = ttk.Treeview(parent, columns=columns, show='headings', height=6)
        for col, text, width in [('name', '名称', 80), ('type', '类型', 90), ('freq', '频率', 100), ('goal', '目标值', 70), ('target', '方向', 70), ('weight', '权重', 60)]:
            self.obj_tree.heading(col, text=text)
            self.obj_tree.column(col, width=width)
        
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.obj_tree.yview)
        self.obj_tree.configure(yscrollcommand=scrollbar.set)
        self.obj_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        edit_frame = ttk.LabelFrame(parent, text="添加/修改目标", padding="5")
        edit_frame.pack(fill=tk.X, pady=5)
        
        row1 = ttk.Frame(edit_frame)
        row1.pack(fill=tk.X, pady=2)
        ttk.Label(row1, text="名称:").pack(side=tk.LEFT)
        self.obj_name = ttk.Entry(row1, width=10)
        self.obj_name.pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="类型:").pack(side=tk.LEFT, padx=(10,0))
        self.obj_type = ttk.Combobox(row1, values=['s_db', 'peak_gain'], width=10, state='readonly')
        self.obj_type.set('s_db')
        self.obj_type.pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="目标值:").pack(side=tk.LEFT, padx=(10,0))
        self.obj_goal = ttk.Entry(row1, width=8)
        self.obj_goal.pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="方向:").pack(side=tk.LEFT, padx=(10,0))
        self.obj_target = ttk.Combobox(row1, values=['minimize', 'maximize'], width=10, state='readonly')
        self.obj_target.set('minimize')
        self.obj_target.pack(side=tk.LEFT, padx=2)
        ttk.Label(row1, text="权重:").pack(side=tk.LEFT, padx=(10,0))
        self.obj_weight = ttk.Entry(row1, width=6)
        self.obj_weight.insert(0, '1.0')
        self.obj_weight.pack(side=tk.LEFT, padx=2)
        
        row2 = ttk.Frame(edit_frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Label(row2, text="频率(GHz):").pack(side=tk.LEFT)
        self.obj_freq = ttk.Entry(row2, width=10)
        self.obj_freq.pack(side=tk.LEFT, padx=2)
        ttk.Label(row2, text="(范围用 5.1-7.2 格式，权重越大越重要)", foreground="gray").pack(side=tk.LEFT, padx=5)
        
        btn_frame = ttk.Frame(edit_frame)
        btn_frame.pack(pady=5)
        ttk.Button(btn_frame, text="添加", command=self.add_objective).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="修改", command=self.update_objective).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="删除", command=self.delete_objective).pack(side=tk.LEFT, padx=2)
        
        self.obj_tree.bind('<<TreeviewSelect>>', self.on_objective_select)
    
    def add_objective(self):
        name = self.obj_name.get().strip()
        obj_type = self.obj_type.get()
        goal = self.obj_goal.get().strip()
        target = self.obj_target.get()
        freq = self.obj_freq.get().strip()
        weight = self.obj_weight.get().strip() or '1.0'
        
        if name and goal:
            type_names = {'s_db': 'S参数(dB)', 'peak_gain': '峰值增益'}
            self.obj_tree.insert('', tk.END, values=(name, type_names.get(obj_type, obj_type), freq, goal, target, weight))
            self.update_objectives_data()
            self.obj_name.delete(0, tk.END)
            self.obj_goal.delete(0, tk.END)
            self.obj_freq.delete(0, tk.END)
            self.obj_weight.delete(0, tk.END)
            self.obj_weight.insert(0, '1.0')
    
    def update_objective(self):
        selected = self.obj_tree.selection()
        if selected:
            name = self.obj_name.get().strip()
            obj_type = self.obj_type.get()
            goal = self.obj_goal.get().strip()
            target = self.obj_target.get()
            freq = self.obj_freq.get().strip()
            weight = self.obj_weight.get().strip() or '1.0'
            type_names = {'s_db': 'S参数(dB)', 'peak_gain': '峰值增益'}
            self.obj_tree.item(selected[0], values=(name, type_names.get(obj_type, obj_type), freq, goal, target, weight))
            self.update_objectives_data()
    
    def delete_objective(self):
        for item in self.obj_tree.selection():
            self.obj_tree.delete(item)
        self.update_objectives_data()
    
    def on_objective_select(self, event):
        if self.obj_tree.selection():
            v = self.obj_tree.item(self.obj_tree.selection()[0], 'values')
            self.obj_name.delete(0, tk.END); self.obj_name.insert(0, v[0])
            type_map = {'S参数(dB)': 's_db', '峰值增益': 'peak_gain'}
            self.obj_type.set(type_map.get(v[1], 's_db'))
            self.obj_goal.delete(0, tk.END); self.obj_goal.insert(0, v[3])
            self.obj_target.set(v[4])
            self.obj_freq.delete(0, tk.END); self.obj_freq.insert(0, v[2].replace(' GHz', ''))
            self.obj_weight.delete(0, tk.END); self.obj_weight.insert(0, v[5] if len(v) > 5 else '1.0')
    
    def update_objectives_data(self):
        self.objectives_data = []
        type_map = {'S参数(dB)': 's_db', '峰值增益': 'peak_gain'}
        for item in self.obj_tree.get_children():
            v = self.obj_tree.item(item, 'values')
            obj = {'type': type_map.get(v[1], 's_db'), 'name': v[0], 'goal': float(v[3]), 'target': v[4]}
            # 添加权重
            if len(v) > 5:
                try:
                    obj['weight'] = float(v[5])
                except:
                    obj['weight'] = 1.0
            else:
                obj['weight'] = 1.0
            freq_str = v[2].replace(' GHz', '')
            if '-' in freq_str:
                parts = freq_str.split('-')
                obj['freq_range'] = (float(parts[0]), float(parts[1]))
            elif freq_str:
                obj['freq'] = float(freq_str)
            if obj['type'] == 's_db':
                obj['port'] = (1, 1)
                obj['constraint'] = 'max'
            self.objectives_data.append(obj)
    
    # ==================== 算法配置页 ====================
    def create_algorithm_tab(self, parent):
        # 创建滚动容器
        scroll_frame = ScrollableFrame(parent)
        scroll_frame.pack(fill=tk.BOTH, expand=True)
        
        inner = scroll_frame.scrollable_frame
        
        # 推荐配置
        rec_frame = ttk.LabelFrame(inner, text="天线优化推荐", padding="8")
        rec_frame.pack(fill=tk.X, pady=5, padx=5)
        
        rec_text = """问题特点: 多维(13变量)、非凸、可能不连续、仿真昂贵

推荐算法: MOBO (贝叶斯优化) - 仿真次数最少，智能选择测试点
推荐代理模型: GP (高斯过程) - 提供不确定性估计，指导下一步测哪里"""
        ttk.Label(rec_frame, text=rec_text, justify=tk.LEFT).pack(anchor=tk.W)
        
        # 算法配置
        cfg_frame = ttk.LabelFrame(inner, text="算法配置", padding="8")
        cfg_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # 算法类型
        ttk.Label(cfg_frame, text="算法类型:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.algorithm_var = tk.StringVar(value="mobo")
        algo_combo = ttk.Combobox(cfg_frame, textvariable=self.algorithm_var, 
                                   values=["mobo", "mopso", "nsga2"], width=18, state="readonly")
        algo_combo.grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        algo_combo.bind('<<ComboboxSelected>>', self.on_algorithm_change)
        self.algo_desc_label = ttk.Label(cfg_frame, text="贝叶斯优化，仿真次数最少", foreground="blue")
        self.algo_desc_label.grid(row=0, column=2, sticky=tk.W)
        
        # ===== 公共参数区 =====
        common_frame = ttk.LabelFrame(cfg_frame, text="基本参数", padding="5")
        common_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        # 种群/样本大小
        ttk.Label(common_frame, text="种群/样本:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.population_var = tk.IntVar(value=50)
        ttk.Spinbox(common_frame, from_=20, to=200, textvariable=self.population_var, width=8).grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        self.pop_desc_label = ttk.Label(common_frame, text="MOBO:初始样本数, 其他:种群大小", foreground="gray")
        self.pop_desc_label.grid(row=0, column=2, sticky=tk.W)
        
        # 迭代次数
        ttk.Label(common_frame, text="迭代次数:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.generations_var = tk.IntVar(value=30)
        ttk.Spinbox(common_frame, from_=10, to=100, textvariable=self.generations_var, width=8).grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)
        self.gen_desc_label = ttk.Label(common_frame, text="优化迭代次数，建议 20-50", foreground="gray")
        self.gen_desc_label.grid(row=1, column=2, sticky=tk.W)
        
        # 图表生成频率
        ttk.Label(common_frame, text="图表频率:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.plot_interval_var = tk.IntVar(value=5)
        ttk.Spinbox(common_frame, from_=1, to=20, textvariable=self.plot_interval_var, width=8).grid(row=2, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(common_frame, text="每几次迭代生成一张图表，1=每次都生成", foreground="gray").grid(row=2, column=2, sticky=tk.W)
        
        # ===== 代理模型区（可选）=====
        self.surrogate_frame = ttk.LabelFrame(cfg_frame, text="代理模型（可选，MOPSO专用）", padding="5")
        self.surrogate_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        # 启用代理模型
        self.use_surrogate_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.surrogate_frame, text="启用代理模型辅助", 
                        variable=self.use_surrogate_var).grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=2)
        
        # 代理模型类型
        ttk.Label(self.surrogate_frame, text="模型类型:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.surrogate_var = tk.StringVar(value="gp")
        surr_combo = ttk.Combobox(self.surrogate_frame, textvariable=self.surrogate_var, 
                                   values=["gp", "rf"], width=8, state="readonly")
        surr_combo.grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(self.surrogate_frame, text="gp=高斯过程(平滑), rf=随机森林(不连续)", foreground="gray").grid(row=1, column=2, columnspan=2, sticky=tk.W)
        
        # 最少样本数
        ttk.Label(self.surrogate_frame, text="最少样本:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.surrogate_min_samples_var = tk.IntVar(value=5)
        ttk.Spinbox(self.surrogate_frame, from_=3, to=20, textvariable=self.surrogate_min_samples_var, width=8).grid(row=2, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(self.surrogate_frame, text="范围 3~20，训练代理模型需要的最少真实仿真次数", foreground="gray").grid(row=2, column=2, columnspan=2, sticky=tk.W)
        
        # 不确定性阈值
        ttk.Label(self.surrogate_frame, text="不确定性阈值:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.surrogate_threshold_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(self.surrogate_frame, from_=0.1, to=2.0, increment=0.1, textvariable=self.surrogate_threshold_var, width=8, format="%.1f").grid(row=3, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(self.surrogate_frame, text="范围 0.1~2.0，值越大越容易使用代理模型", foreground="gray").grid(row=3, column=2, columnspan=2, sticky=tk.W)
        
        # 加载历史评估数据
        ttk.Label(self.surrogate_frame, text="加载历史数据:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.load_eval_var = tk.StringVar(value="")
        load_entry = ttk.Entry(self.surrogate_frame, textvariable=self.load_eval_var, width=30)
        load_entry.grid(row=4, column=1, columnspan=2, sticky=tk.W, pady=2, padx=5)
        ttk.Button(self.surrogate_frame, text="选择...", command=self._select_eval_file, width=8).grid(row=4, column=3, sticky=tk.W, pady=2)
        ttk.Label(self.surrogate_frame, text="导入之前的 evaluations.jsonl 继续优化", foreground="gray").grid(row=5, column=1, columnspan=3, sticky=tk.W)
        
        # 说明
        desc_label = ttk.Label(self.surrogate_frame, 
            text="提示：代理模型需要足够的真实仿真样本才能发挥作用。加载历史数据可以复用之前的仿真结果。", 
            foreground="blue", justify=tk.LEFT)
        desc_label.grid(row=6, column=0, columnspan=4, sticky=tk.W, pady=5)
        
        # ===== MOBO 高级参数 =====
        self.mobo_advanced_frame = ttk.LabelFrame(cfg_frame, text="MOBO 高级参数", padding="5")
        self.mobo_advanced_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        ttk.Label(self.mobo_advanced_frame, text="采集函数:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.acquisition_var = tk.StringVar(value="ehvi")
        ttk.Combobox(self.mobo_advanced_frame, textvariable=self.acquisition_var, 
                     values=["ehvi", "ei", "ucb"], width=10, state="readonly").grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(self.mobo_advanced_frame, text="ehvi=多目标, ei=单目标, ucb=探索型", foreground="gray").grid(row=0, column=2, sticky=tk.W)
        
        # ===== MOPSO 参数 =====
        self.mopso_frame = ttk.LabelFrame(cfg_frame, text="MOPSO 参数", padding="5")
        self.mopso_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        ttk.Label(self.mopso_frame, text="惯性权重:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.inertia_var = tk.DoubleVar(value=0.5)
        ttk.Spinbox(self.mopso_frame, from_=0.2, to=0.9, increment=0.1, textvariable=self.inertia_var, width=8, format="%.1f").grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(self.mopso_frame, text="范围 0.2~0.9，大则全局探索，小则局部精细", foreground="gray").grid(row=0, column=2, sticky=tk.W)
        
        ttk.Label(self.mopso_frame, text="学习因子:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.c1_var = tk.DoubleVar(value=1.5)
        ttk.Spinbox(self.mopso_frame, from_=0.5, to=3.0, increment=0.1, textvariable=self.c1_var, width=8, format="%.1f").grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(self.mopso_frame, text="范围 0.5~3.0，粒子向最优解学习的强度", foreground="gray").grid(row=1, column=2, sticky=tk.W)
        
        # ===== NSGA2 参数 =====
        self.nsga2_frame = ttk.LabelFrame(cfg_frame, text="NSGA2 参数", padding="5")
        self.nsga2_frame.grid(row=3, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        ttk.Label(self.nsga2_frame, text="变异概率:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.mutation_var = tk.DoubleVar(value=0.15)
        ttk.Spinbox(self.nsga2_frame, from_=0.05, to=0.5, increment=0.01, textvariable=self.mutation_var, width=8, format="%.2f").grid(row=0, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(self.nsga2_frame, text="建议 0.1-0.2", foreground="gray").grid(row=0, column=2, sticky=tk.W)
        
        ttk.Label(self.nsga2_frame, text="交叉概率:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.crossover_var = tk.DoubleVar(value=0.9)
        ttk.Spinbox(self.nsga2_frame, from_=0.5, to=1.0, increment=0.05, textvariable=self.crossover_var, width=8, format="%.2f").grid(row=1, column=1, sticky=tk.W, pady=2, padx=5)
        ttk.Label(self.nsga2_frame, text="建议 0.8-0.95", foreground="gray").grid(row=1, column=2, sticky=tk.W)
        
        ttk.Label(self.nsga2_frame, text="⚠️ NSGA2 不使用代理模型，仿真次数 = 种群 × 迭代", foreground="orange").grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 预估仿真次数
        self.est_eval_label = ttk.Label(cfg_frame, text="", foreground="green", font=('Arial', 11, 'bold'))
        self.est_eval_label.grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # 绑定参数变化
        self.population_var.trace_add('write', lambda *args: self.update_estimated_evaluations())
        self.generations_var.trace_add('write', lambda *args: self.update_estimated_evaluations())
        
        # 算法对比
        cmp_frame = ttk.LabelFrame(inner, text="算法对比", padding="8")
        cmp_frame.pack(fill=tk.X, pady=5, padx=5)
        
        cmp_text = """算法      仿真次数    特点                    推荐场景
───────────────────────────────────────────────────────────
MOBO      最少        智能选择测试点          HFSS仿真慢，想少跑几次
MOPSO     中等        粒子群协作，收敛快      想要快一点的收敛
NSGA2     最多        遗传算法，全面搜索      不赶时间，要全面

代理模型:
• gp (高斯过程): 平滑连续函数，提供不确定性估计，MOBO必须
• rf (随机森林): 不连续函数，天线结构突变时推荐"""
        
        ttk.Label(cmp_frame, text=cmp_text, justify=tk.LEFT, font=('Consolas', 9)).pack(anchor=tk.W)
        
        ttk.Button(inner, text="保存配置", command=self.save_config).pack(pady=10)
    
    def on_algorithm_change(self, event=None):
        """算法类型改变时切换参数面板"""
        algo = self.algorithm_var.get()
        
        # 隐藏所有高级参数面板
        self.mobo_advanced_frame.grid_remove()
        self.mopso_frame.grid_remove()
        self.nsga2_frame.grid_remove()
        self.surrogate_frame.grid_remove()
        
        # 更新描述和显示对应面板
        if algo == "mobo":
            self.algo_desc_label.config(text="贝叶斯优化，仿真次数最少")
            self.pop_desc_label.config(text="MOBO:初始样本数, 其他:种群大小")
            self.gen_desc_label.config(text="优化迭代次数，建议 20-50")
            self.mobo_advanced_frame.grid()
            # MOBO 内置代理模型，不显示选项
        elif algo == "mopso":
            self.algo_desc_label.config(text="多目标粒子群，收敛快")
            self.pop_desc_label.config(text="粒子群大小，建议 30-100")
            self.gen_desc_label.config(text="迭代代数，建议 20-50")
            self.mopso_frame.grid()
            # MOPSO 代理模型可选（可减少仿真次数）
            self.surrogate_frame.grid()
        elif algo == "nsga2":
            self.algo_desc_label.config(text="遗传算法，最稳健但仿真多")
            self.pop_desc_label.config(text="种群大小，建议 50-100")
            self.gen_desc_label.config(text="迭代代数，建议 30-50")
            self.nsga2_frame.grid()
            # NSGA2 不用代理模型
        
        self.update_estimated_evaluations()
    
    def update_estimated_evaluations(self):
        """更新预估仿真次数"""
        algo = self.algorithm_var.get()
        pop = self.population_var.get()
        gen = self.generations_var.get()
        
        if algo == "mobo":
            total = pop + gen
            self.est_eval_label.config(text=f"预估仿真次数: {pop}(初始) + {gen}(迭代) = {total} 次  |  约 {total*3//60} 小时")
        elif algo == "mopso":
            total = pop * gen
            self.est_eval_label.config(text=f"预估仿真次数: {pop} × {gen} = {total} 次  |  约 {total*3//60} 小时")
        elif algo == "nsga2":
            total = pop * gen
            self.est_eval_label.config(text=f"预估仿真次数: {pop} × {gen} = {total} 次  |  约 {total*3//60} 小时")
    
    # ==================== 运行页 ====================
    
    # ==================== 运行页 ====================
    def create_run_tab(self, parent):
        control_frame = ttk.Frame(parent)
        control_frame.pack(fill=tk.X, pady=5)
        
        # 自检按钮
        self.check_button = ttk.Button(control_frame, text="自检项目", command=self.run_check)
        self.check_button.pack(side=tk.LEFT, padx=5)
        
        # 自检采样数量
        ttk.Label(control_frame, text="采样数:").pack(side=tk.LEFT, padx=(10, 2))
        self.check_samples_var = tk.IntVar(value=10)
        samples_spin = ttk.Spinbox(control_frame, from_=5, to=100, width=5, textvariable=self.check_samples_var)
        samples_spin.pack(side=tk.LEFT)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.run_button = ttk.Button(control_frame, text="开始优化", command=self.start_optimization)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(control_frame, text="停止", command=self.stop_optimization, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(control_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        ttk.Button(control_frame, text="保存配置", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="打开结果", command=self.open_results).pack(side=tk.LEFT, padx=5)
        
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=2)
        ttk.Label(status_frame, text="状态:").pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(status_frame, textvariable=self.status_var, foreground="blue").pack(side=tk.LEFT, padx=5)
        
        # 进度信息
        progress_frame = ttk.LabelFrame(parent, text="优化进度", padding="5")
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_info = ttk.Label(progress_frame, text="等待开始...")
        self.progress_info.pack(anchor=tk.W)
        
        # 创建带图表的区域
        chart_frame = ttk.LabelFrame(parent, text="运行日志", padding="5")
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(chart_frame, height=15, state=tk.DISABLED)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def run_check(self):
        """运行项目自检"""
        # 获取采样数量
        try:
            n_samples = self.check_samples_var.get()
        except:
            n_samples = 10
        
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        self.status_var.set("自检中...")
        self.progress_info.config(text=f"正在检查项目配置 (采样 {n_samples} 组)...")
        
        # 在线程中运行
        thread = threading.Thread(target=self._run_check_thread, args=(n_samples,))
        thread.daemon = True
        thread.start()
    
    def _run_check_thread(self, n_samples: int):
        """自检线程"""
        try:
            from checker import ProjectChecker
            
            def progress_callback(current, total, message):
                self.root.after(0, lambda: self.progress_info.config(text=f"[{current}/{total}] {message}"))
                self.log(message)
            
            checker = ProjectChecker(self.config, n_samples=n_samples)
            results = checker.run_all_checks(progress_callback)
            
            # 显示报告
            report = checker.get_report_text()
            
            self.root.after(0, lambda: self._show_check_results(results, report))
            
        except Exception as e:
            self.root.after(0, lambda: self.log(f"自检异常: {e}"))
            self.root.after(0, lambda: self.status_var.set("自检失败"))
    
    def _show_check_results(self, results: Dict, report: str):
        """显示自检结果"""
        self.log("\n" + report)
        
        summary = results.get('summary', {})
        status = summary.get('status', 'UNKNOWN')
        
        if status == 'OK':
            self.status_var.set("自检通过")
            self.progress_info.config(text="✅ 所有检查通过，可以开始优化")
            self.log("\n[OK] 所有检查通过！")
        elif status == 'OK_WITH_WARNINGS':
            self.status_var.set("自检通过(有警告)")
            self.progress_info.config(text=f"⚠️ 有 {summary.get('warnings', 0)} 个警告，建议检查")
            self.log(f"\n[WARNING] 有 {summary.get('warnings', 0)} 个警告项")
        elif status == 'WARNING':
            self.status_var.set("自检发现问题")
            self.progress_info.config(text=f"⚠️ 发现 {summary.get('warnings', 0)} 个问题，建议检查变量范围")
            self.log(f"\n[WARNING] 发现较多问题，建议检查变量范围")
        else:  # ERROR
            self.status_var.set("自检失败")
            self.progress_info.config(text=f"❌ 发现 {summary.get('errors', 0)} 个错误，必须修复")
            self.log(f"\n[ERROR] 发现 {summary.get('errors', 0)} 个错误，必须修复后再优化")
    
    def load_data_to_ui(self):
        # 加载变量
        for var in self.config.get('variables', []):
            self.var_tree.insert('', tk.END, values=(var['name'], var['bounds'][0], var['bounds'][1], var.get('unit', 'mm')))
        self.update_variables_data()
        
        # 加载目标
        type_names = {'s_db': 'S参数(dB)', 'peak_gain': '峰值增益'}
        for obj in self.config.get('objectives', []):
            freq_str = ""
            if 'freq_range' in obj:
                freq_str = f"{obj['freq_range'][0]}-{obj['freq_range'][1]} GHz"
            elif 'freq' in obj:
                freq_str = f"{obj['freq']} GHz"
            weight = obj.get('weight', 1.0)
            self.obj_tree.insert('', tk.END, values=(obj.get('name', ''), type_names.get(obj['type'], obj['type']), freq_str, obj.get('goal', ''), obj.get('target', 'minimize'), weight))
        self.update_objectives_data()
        
        # 加载算法配置
        algo_config = self.config.get('algorithm', {})
        algo = algo_config.get('algorithm', 'mobo')
        self.algorithm_var.set(algo)
        self.population_var.set(algo_config.get('population_size', algo_config.get('initial_samples', 50)))
        self.generations_var.set(algo_config.get('n_generations', algo_config.get('n_iterations', 30)))
        self.surrogate_var.set(algo_config.get('surrogate_type', 'gp'))
        self.use_surrogate_var.set(algo_config.get('use_surrogate', False))
        self.surrogate_min_samples_var.set(algo_config.get('surrogate_min_samples', 5))
        self.surrogate_threshold_var.set(algo_config.get('surrogate_threshold', 1.0))
        self.load_eval_var.set(algo_config.get('load_evaluations', ''))
        
        # 加载可视化配置
        viz_config = self.config.get('visualization', {})
        self.plot_interval_var.set(viz_config.get('plot_interval', 5))
        
        # 算法特定参数
        if algo == "mobo":
            self.acquisition_var.set(algo_config.get('acquisition', 'ehvi'))
        elif algo == "mopso":
            self.inertia_var.set(algo_config.get('inertia_weight', 0.5))
            self.c1_var.set(algo_config.get('c1', 1.5))
        elif algo == "nsga2":
            self.mutation_var.set(algo_config.get('mutation_prob', 0.15))
            self.crossover_var.set(algo_config.get('crossover_prob', 0.9))
        
        # 更新界面显示
        self.on_algorithm_change()
    
    def browse_project(self):
        filename = filedialog.askopenfilename(title="选择 HFSS 项目", filetypes=[("HFSS Project", "*.aedt"), ("All Files", "*.*")])
        if filename:
            self.project_path_var.set(filename)
    
    def log(self, message):
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
    
    def start_optimization(self):
        if not self.project_path_var.get():
            messagebox.showerror("错误", "请选择 HFSS 项目")
            return
        if not self.variables_data:
            messagebox.showerror("错误", "请添加变量")
            return
        if not self.objectives_data:
            messagebox.showerror("错误", "请添加目标")
            return
        
        self.save_config()
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.running = True
        self.status_var.set("运行中...")
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        thread = threading.Thread(target=self.run_optimization)
        thread.daemon = True
        thread.start()
    
    def run_optimization(self):
        try:
            import subprocess
            cmd = [sys.executable, os.path.join(PROJECT_ROOT, "run.py"), "--config", CONFIG_FILE]
            self.log("启动优化...")
            self.log(f"变量: {len(self.variables_data)}, 目标: {len(self.objectives_data)}")
            
            self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace', cwd=PROJECT_ROOT)
            
            for line in iter(self.process.stdout.readline, ''):
                if not self.running:
                    break
                if line.strip():
                    self.log(line.strip())
            
            self.process.wait()
            if self.running:
                self.log("完成!")
                self.status_var.set("完成")
            else:
                self.log("已停止")
                self.status_var.set("已停止")
        except Exception as e:
            self.log(f"错误: {e}")
            self.status_var.set("错误")
        finally:
            self.root.after(0, self.optimization_finished)
    
    def optimization_finished(self):
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.running = False
        self.process = None
    
    def stop_optimization(self):
        if self.process:
            self.running = False
            self.process.terminate()
            self.log("停止中...")
    
    def open_results(self):
        import subprocess
        results_dir = os.path.join(PROJECT_ROOT, "optim_results")
        os.makedirs(results_dir, exist_ok=True)
        if sys.platform == 'win32':
            subprocess.run(['explorer', results_dir])
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    OptimizerGUI().run()
