"""
优化可视化模块 - 实时展示优化进度

可视化内容：
1. 帕累托前沿 (Pareto Front)
2. 收敛曲线 (Convergence)
3. 目标空间探索过程
4. 参数分布
5. 代理模型预测 vs 真实值
"""
import os
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# 尝试导入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False
    print("[WARN] matplotlib not available, visualization disabled")


class OptimizationVisualizer:
    """
    优化可视化器 - 实时生成图表
    """
    
    def __init__(self, output_dir: str, objectives: List[Dict], variables: List[Dict] = None, plot_interval: int = 5):
        self.output_dir = output_dir
        self.objectives = objectives
        self.variables = variables or []
        self.n_objectives = len(objectives)
        self.plot_interval = plot_interval  # 图表生成频率
        
        # 数据存储
        self.evaluations = []  # 所有评估点
        self.pareto_history = []  # 帕累托前沿历史
        self.hypervolume_history = []  # 超体积历史
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 图片保存路径
        self.figures_dir = os.path.join(output_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def update(self, iteration: int, params: np.ndarray, objectives: np.ndarray, 
               pareto_params: np.ndarray = None, pareto_objectives: np.ndarray = None):
        """
        更新数据并生成图表
        
        Args:
            iteration: 当前迭代次数
            params: 当前参数
            objectives: 当前目标值
            pareto_params: 帕累托参数集
            pareto_objectives: 帕累托目标集
        """
        # 存储数据 - 确保 objectives 是一维列表
        if hasattr(objectives, 'flatten'):
            obj_list = objectives.flatten().tolist()
        elif hasattr(objectives, 'tolist'):
            obj_list = objectives.tolist()
        elif isinstance(objectives, list):
            obj_list = list(objectives)
        else:
            obj_list = [objectives]
        
        # 确保是一维的
        if isinstance(obj_list[0], list):
            obj_list = obj_list[0] if len(obj_list) == 1 else [item for sublist in obj_list for item in sublist]
        
        self.evaluations.append({
            'iteration': iteration,
            'params': params.tolist() if hasattr(params, 'tolist') else list(params),
            'objectives': obj_list
        })
        
        # 更新帕累托前沿
        if pareto_objectives is not None and len(pareto_objectives) > 0:
            self.pareto_history.append(pareto_objectives.copy())
            
            # 计算超体积
            hv = self._compute_hypervolume(pareto_objectives)
            self.hypervolume_history.append(hv)
        
        # 根据设置生成图表
        if iteration % self.plot_interval == 0 or iteration <= self.plot_interval:
            self._generate_figures(iteration, pareto_params, pareto_objectives)
    
    def _compute_hypervolume(self, pareto_front: np.ndarray) -> float:
        """计算超体积指标"""
        if len(pareto_front) == 0:
            return 0.0
        
        # 简化：使用 2D 超体积（假设最小化）
        if pareto_front.shape[1] == 2:
            # 参考点：最差点
            ref_point = pareto_front.max(axis=0) * 1.1
            
            # 计算 2D 超体积
            hv = 0.0
            sorted_front = pareto_front[pareto_front[:, 0].argsort()]
            
            for i in range(len(sorted_front)):
                if i == 0:
                    hv += (ref_point[0] - sorted_front[i, 0]) * (ref_point[1] - sorted_front[i, 1])
                else:
                    hv += (sorted_front[i-1, 0] - sorted_front[i, 0]) * (ref_point[1] - sorted_front[i, 1])
            
            return hv
        
        return 0.0
    
    def _generate_figures(self, iteration: int, pareto_params: np.ndarray, pareto_objectives: np.ndarray):
        """生成可视化图表"""
        if not MPL_AVAILABLE:
            return
        
        try:
            # 1. 帕累托前沿
            if pareto_objectives is not None and len(pareto_objectives) > 0:
                self._plot_pareto_front(iteration, pareto_objectives)
            
            # 2. 收敛曲线
            if len(self.hypervolume_history) > 0:
                self._plot_convergence(iteration)
            
            # 3. 目标空间探索
            self._plot_objective_space(iteration)
            
        except Exception as e:
            print(f"[WARN] Figure generation failed: {e}")
    
    def _plot_pareto_front(self, iteration: int, pareto_objectives: np.ndarray):
        """绘制帕累托前沿"""
        if self.n_objectives == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 获取目标类型（用于判断是否需要取反显示）
            obj_targets = [obj.get('target', 'minimize') for obj in self.objectives]
            
            # 所有评估点 - 兼容列表格式
            all_objs_list = []
            for e in self.evaluations:
                obj = e['objectives']
                if isinstance(obj, list):
                    all_objs_list.append(obj)
                elif isinstance(obj, dict):
                    all_objs_list.append(list(obj.values()))
                else:
                    all_objs_list.append([obj])
            
            if len(all_objs_list) == 0:
                plt.close(fig)
                return
            
            all_objs = np.array(all_objs_list)
            
            # 检查维度
            if all_objs.ndim != 2 or all_objs.shape[1] != 2:
                print(f"[WARN] Cannot plot pareto front: all_objs shape = {all_objs.shape}")
                plt.close(fig)
                return
            
            # 转换为实际值显示（maximize 目标取反）
            all_objs_display = all_objs.copy()
            for i, target in enumerate(obj_targets):
                if target == 'maximize' and i < all_objs_display.shape[1]:
                    all_objs_display[:, i] = -all_objs_display[:, i]
            
            ax.scatter(all_objs_display[:, 0], all_objs_display[:, 1], c='lightgray', s=30, alpha=0.5, label='All evaluations')
            
            # 帕累托前沿
            if pareto_objectives is not None and len(pareto_objectives) > 0:
                # 转换为实际值显示
                pareto_display = pareto_objectives.copy()
                for i, target in enumerate(obj_targets):
                    if target == 'maximize' and i < pareto_display.shape[1]:
                        pareto_display[:, i] = -pareto_display[:, i]
                
                ax.scatter(pareto_display[:, 0], pareto_display[:, 1], c='red', s=50, label='Pareto front')
                
                # 连接帕累托点
                sorted_idx = np.argsort(pareto_display[:, 0])
                ax.plot(pareto_display[sorted_idx, 0], pareto_display[sorted_idx, 1], 'r--', alpha=0.5)
                sorted_idx = np.argsort(pareto_objectives[:, 0])
                ax.plot(pareto_objectives[sorted_idx, 0], pareto_objectives[sorted_idx, 1], 'r--', alpha=0.5)
            
            obj_names = [obj.get('name', f'Obj{i}') for i, obj in enumerate(self.objectives)]
            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])
            ax.set_title(f'Pareto Front (Iteration {iteration})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f'pareto_front_iter{iteration}.png'), dpi=150)
            plt.close(fig)
        
        elif self.n_objectives == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 所有评估点 - 兼容列表格式
            all_objs = np.array([e['objectives'] if isinstance(e['objectives'], list) 
                                 else list(e['objectives'].values()) if isinstance(e['objectives'], dict)
                                 else [e['objectives']] 
                                 for e in self.evaluations])
            ax.scatter(all_objs[:, 0], all_objs[:, 1], all_objs[:, 2], c='lightgray', s=30, alpha=0.5)
            
            # 帕累托前沿
            ax.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], pareto_objectives[:, 2], 
                      c='red', s=50, label='Pareto front')
            
            obj_names = [obj.get('name', f'Obj{i}') for i, obj in enumerate(self.objectives)]
            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])
            ax.set_zlabel(obj_names[2])
            ax.set_title(f'Pareto Front (Iteration {iteration})')
            ax.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f'pareto_front_iter{iteration}.png'), dpi=150)
            plt.close(fig)
    
    def _plot_convergence(self, iteration: int):
        """绘制收敛曲线"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = list(range(len(self.hypervolume_history)))
        ax.plot(iterations, self.hypervolume_history, 'b-', linewidth=2, label='Hypervolume')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Hypervolume')
        ax.set_title(f'Convergence (Iteration {iteration})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.figures_dir, f'convergence_iter{iteration}.png'), dpi=150)
        plt.close(fig)
    
    def _plot_objective_space(self, iteration: int):
        """绘制目标空间探索过程"""
        if self.n_objectives >= 1:
            fig, axes = plt.subplots(1, max(self.n_objectives, 1), figsize=(5*max(self.n_objectives, 1), 4))
            if self.n_objectives == 1:
                axes = [axes]
            
            # 获取目标类型
            obj_targets = [obj.get('target', 'minimize') for obj in self.objectives]
            
            for i, ax in enumerate(axes):
                # 兼容列表格式
                obj_values = []
                for e in self.evaluations:
                    obj = e['objectives']
                    if isinstance(obj, list) and len(obj) > i:
                        val = obj[i]
                        # 对于 maximize 目标，取反显示实际值
                        if i < len(obj_targets) and obj_targets[i] == 'maximize':
                            val = -val
                        obj_values.append(val)
                    else:
                        obj_values.append(0)
                
                iterations = [e['iteration'] for e in self.evaluations]
                
                if len(iterations) > 0 and len(obj_values) > 0:
                    ax.scatter(iterations, obj_values, c=range(len(obj_values)), cmap='viridis', s=30)
                ax.set_xlabel('Iteration')
                obj_name = self.objectives[i].get('name', f'Objective {i+1}') if i < len(self.objectives) else f'Objective {i+1}'
                ax.set_ylabel(obj_name)
                ax.set_title(f'{obj_name} Evolution')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f'objectives_iter{iteration}.png'), dpi=150)
            plt.close(fig)
    
    def generate_final_report(self, pareto_solutions: List[Dict], stats: Dict):
        """生成最终报告"""
        if not MPL_AVAILABLE:
            return
        
        # 最终帕累托前沿 - 兼容两种格式
        try:
            pareto_objectives = np.array([
                s['objectives'] if isinstance(s.get('objectives'), list) 
                else list(s['objectives'].values()) if isinstance(s.get('objectives'), dict)
                else [s.get('objectives', 0)]
                for s in pareto_solutions if s.get('objectives') is not None
            ])
            pareto_params = np.array([
                s['parameters'] if 'parameters' in s else s.get('params', [])
                for s in pareto_solutions
            ])
            
            if len(pareto_objectives) > 0:
                self._generate_figures(9999, pareto_params, pareto_objectives)
        except Exception as e:
            print(f"[WARN] Failed to generate figures: {e}")
        
        # 保存数据
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_evaluations': stats.get('n_evaluations', 0),
            'pareto_size': len(pareto_solutions),
            'best_objectives': stats.get('best_objectives', []),
            'pareto_solutions': pareto_solutions[:10],  # 只保存前10个
        }
        
        with open(os.path.join(self.output_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成文本报告
        self._generate_text_report(pareto_solutions, stats)
    
    def _generate_text_report(self, pareto_solutions: List[Dict], stats: Dict):
        """生成文本报告"""
        # 获取目标配置
        obj_names = [obj.get('name', f'Obj{i}') for i, obj in enumerate(self.objectives)]
        obj_targets = [obj.get('target', 'minimize') for obj in self.objectives]
        obj_weights = [obj.get('weight', 1.0) for obj in self.objectives]
        
        # 归一化权重
        total_weight = sum(obj_weights)
        obj_weights_norm = [w / total_weight for w in obj_weights]
        
        report_lines = [
            "=" * 70,
            "OPTIMIZATION REPORT",
            "=" * 70,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total evaluations: {stats.get('n_evaluations', 0)}",
            f"Pareto solutions: {len(pareto_solutions)}",
            "",
            "=" * 70,
            "OBJECTIVE WEIGHTS",
            "=" * 70,
        ]
        
        for name, weight in zip(obj_names, obj_weights_norm):
            report_lines.append(f"  {name}: {weight:.2%}")
        
        report_lines.extend([
            "",
            "=" * 70,
            "PARETO OPTIMAL SOLUTIONS",
            "=" * 70,
        ])
        
        # 计算每个解的加权得分并排序（满分100分）
        scored_solutions = []
        
        # 获取目标值（goal）
        obj_goals = [obj.get('goal', None) for obj in self.objectives]
        
        for sol in pareto_solutions:
            objectives = sol.get('objectives', {})
            if isinstance(objectives, dict):
                obj_values = list(objectives.values())
            elif isinstance(objectives, list):
                obj_values = objectives
            else:
                obj_values = [objectives]
            
            # 计算每个目标的得分（0-100分）
            obj_scores = []
            for j, val in enumerate(obj_values):
                if j >= len(obj_targets):
                    break
                target = obj_targets[j]
                goal = obj_goals[j] if j < len(obj_goals) else None
                weight = obj_weights_norm[j] if j < len(obj_weights_norm) else 1.0
                
                # 转换为实际值
                if target == 'maximize':
                    actual = -val if val < 0 else val
                else:
                    actual = val
                
                # 计算得分（越接近目标越高，满分100）
                if goal is None:
                    score = 50  # 没有目标值，默认50分
                elif target == 'minimize':
                    # minimize: 值越小越好
                    if actual <= goal:
                        score = 100
                    else:
                        excess_ratio = abs((actual - goal) / max(abs(goal), 1))
                        score = max(0, 100 - excess_ratio * 40)
                else:
                    # maximize: 值越大越好
                    if actual >= goal:
                        score = 100
                    else:
                        deficit_ratio = abs((goal - actual) / max(abs(goal), 1))
                        score = max(0, 100 - deficit_ratio * 40)
                
                obj_scores.append(score * weight)
            
            # 加权平均得分（满分100）
            total_score = sum(obj_scores) / sum(obj_weights_norm) if sum(obj_weights_norm) > 0 else 0
            scored_solutions.append((total_score, sol))
        
        # 按得分排序（越高越好）
        scored_solutions.sort(key=lambda x: -x[0])  # 降序
        
        for rank, (score, sol) in enumerate(scored_solutions[:5], 1):
            objectives = sol.get('objectives', {})
            params = sol.get('parameters', [])
            
            if rank == 1:
                report_lines.append(f"\n★ BEST SOLUTION (rank {rank}, score: {score:.1f}/100):")
            else:
                report_lines.append(f"\nSolution {rank} (score: {score:.1f}/100):")
            
            report_lines.append("  Objectives:")
            if isinstance(objectives, dict):
                for name, val in objectives.items():
                    # 如果是 maximize 目标（存储为负值），显示正值
                    idx = obj_names.index(name) if name in obj_names else -1
                    if idx >= 0 and obj_targets[idx] == 'maximize':
                        display_val = -val if val < 0 else val
                    else:
                        display_val = val
                    
                    # 计算该目标的得分
                    goal = obj_goals[idx] if idx >= 0 and idx < len(obj_goals) else None
                    if goal is not None:
                        target = obj_targets[idx] if idx >= 0 else 'minimize'
                        if target == 'minimize':
                            if display_val <= goal:
                                obj_score = 100
                            else:
                                excess_ratio = abs((display_val - goal) / max(abs(goal), 1))
                                obj_score = max(0, 100 - excess_ratio * 40)
                        else:
                            if display_val >= goal:
                                obj_score = 100
                            else:
                                deficit_ratio = abs((goal - display_val) / max(abs(goal), 1))
                                obj_score = max(0, 100 - deficit_ratio * 40)
                        report_lines.append(f"    {name}: {display_val:.4f} (goal: {goal}, score: {obj_score:.0f}/100)")
                    else:
                        report_lines.append(f"    {name}: {display_val:.4f}")
            else:
                # 列表格式
                for j, val in enumerate(objectives):
                    if j < len(obj_names):
                        name = obj_names[j]
                        target = obj_targets[j] if j < len(obj_targets) else 'minimize'
                        goal = obj_goals[j] if j < len(obj_goals) else None
                        
                        # 转换为实际值
                        if target == 'maximize':
                            display_val = -val if val < 0 else val
                        else:
                            display_val = val
                        
                        # 计算得分
                        if goal is not None:
                            if target == 'minimize':
                                if display_val <= goal:
                                    obj_score = 100
                                else:
                                    excess_ratio = abs((display_val - goal) / max(abs(goal), 1))
                                    obj_score = max(0, 100 - excess_ratio * 40)
                            else:
                                if display_val >= goal:
                                    obj_score = 100
                                else:
                                    deficit_ratio = abs((goal - display_val) / max(abs(goal), 1))
                                    obj_score = max(0, 100 - deficit_ratio * 40)
                            report_lines.append(f"    {name}: {display_val:.4f} (goal: {goal}, score: {obj_score:.0f}/100)")
                        else:
                            report_lines.append(f"    {name}: {display_val:.4f}")
            
            report_lines.append("  Parameters:")
            # 从 self.variables 获取变量名
            var_names = [v.get('name', f'Var{j}') for j, v in enumerate(self.variables)] if self.variables else [f'Var{j}' for j in range(len(params))]
            
            for j, val in enumerate(params):
                name = var_names[j] if j < len(var_names) else f'Var{j}'
                report_lines.append(f"    {name}: {val:.4f}")
        
        report_lines.append("\n" + "=" * 70)
        
        with open(os.path.join(self.output_dir, 'report.txt'), 'w') as f:
            f.write('\n'.join(report_lines))


class RealTimePlotter:
    """
    实时绘图器 - 用于 GUI 实时更新
    """
    
    def __init__(self):
        self.iterations = []
        self.objective_values = {i: [] for i in range(10)}  # 支持最多10个目标
        self.pareto_fronts = []
    
    def add_point(self, iteration: int, objectives: np.ndarray):
        """添加一个评估点"""
        self.iterations.append(iteration)
        for i, obj_val in enumerate(objectives):
            self.objective_values[i].append(obj_val)
    
    def update_pareto(self, pareto_objectives: np.ndarray):
        """更新帕累托前沿"""
        self.pareto_fronts.append(pareto_objectives.copy())
    
    def get_convergence_data(self, obj_idx: int = 0) -> tuple:
        """获取收敛数据"""
        return self.iterations, self.objective_values[obj_idx]
    
    def get_pareto_data(self) -> Optional[np.ndarray]:
        """获取当前帕累托前沿"""
        if self.pareto_fronts:
            return self.pareto_fronts[-1]
        return None
