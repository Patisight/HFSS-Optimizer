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
    
    def __init__(self, output_dir: str, objectives: List[Dict], variables: List[Dict] = None, 
                 plot_interval: int = 5, surrogate_recent_window: int = 5):
        self.output_dir = output_dir
        self.objectives = objectives
        self.variables = variables or []
        self.n_objectives = len(objectives)
        self.plot_interval = plot_interval  # 图表生成频率
        self.surrogate_recent_window = surrogate_recent_window  # 代理对比图展示最近几次评估
        
        # 数据存储
        self.evaluations = []  # 所有评估点
        self.pareto_history = []  # 帕累托前沿历史
        self.hypervolume_history = []  # 超体积历史
        
        # 分离存储：真实仿真点 vs 纯代理预测点
        self.real_evaluations = []  # 真实仿真点（包含真实目标值）
        self.surrogate_only_evaluations = []  # 纯代理预测点（只有预测值，无真实值）
        
        # 代理模型对比数据：真实仿真时同时记录代理预测值（用于对比图）
        self.surrogate_comparison_data = []  # [{iteration, real_objectives, pred_objectives}]
    
    def _normalize_objectives(self, obj):
        """统一处理目标值格式，返回一维列表"""
        if obj is None:
            return [0.0] * self.n_objectives
        if isinstance(obj, dict):
            return list(obj.values())
        elif isinstance(obj, (list, np.ndarray)):
            result = list(obj.flatten()) if hasattr(obj, 'flatten') else list(obj)
            while result and isinstance(result[0], (list, np.ndarray)):
                result = [item for sublist in result for item in sublist]
            try:
                return [float(x) for x in result]
            except (ValueError, TypeError) as e:
                print(f"[WARN] Failed to convert objectives to float: {result}, error: {e}")
                return [0.0] * self.n_objectives
        else:
            try:
                return [float(obj)]
            except (ValueError, TypeError):
                return [0.0] * self.n_objectives
    
    def update(self, iteration: int, params: np.ndarray, objectives: np.ndarray, 
               pareto_params: np.ndarray = None, pareto_objectives: np.ndarray = None,
               surrogate_preds: np.ndarray = None, is_surrogate_prediction: bool = False):
        """
        更新数据并生成图表
        
        Args:
            iteration: 当前迭代次数
            params: 当前参数
            objectives: 当前目标值
            pareto_params: 帕累托参数集
            pareto_objectives: 帕累托目标集
            surrogate_preds: 代理模型预测值（与objectives同维度，可选）
            is_surrogate_prediction: 是否为纯代理预测（无对应真实值）
        """
        # 创建输出目录（延迟创建，确保 output_dir 已就绪）
        os.makedirs(self.output_dir, exist_ok=True)
        self.figures_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(self.figures_dir, exist_ok=True)
        self.surrogate_figures_dir = os.path.join(self.output_dir, 'surrogate_figures')
        os.makedirs(self.surrogate_figures_dir, exist_ok=True)
        
        # 统一处理 objectives 格式
        obj_list = self._normalize_objectives(objectives)
        
        # 获取 params 列表格式（防御性处理）
        if hasattr(params, 'tolist'):
            param_list = params.tolist()
        elif isinstance(params, dict):
            param_list = list(params.values())
        elif isinstance(params, (list, tuple)):
            param_list = list(params)
        else:
            param_list = [float(params)] if not isinstance(params, str) else [params]
        # 确保 param_list 是一维列表
        while param_list and isinstance(param_list[0], (list, np.ndarray)):
            param_list = [item for sublist in param_list for item in sublist]
        
        # 存储到对应数据结构
        eval_entry = {
            'iteration': iteration,
            'params': param_list,
            'objectives': obj_list,
            'is_predicted': is_surrogate_prediction  # 保存标志以便后续识别
        }
        self.evaluations.append(eval_entry)
        
        if is_surrogate_prediction:
            # 纯代理预测点（只有预测值）
            self.surrogate_only_evaluations.append(eval_entry)
        else:
            # 真实仿真点
            self.real_evaluations.append(eval_entry)
            
            # 如果真实仿真时同时有代理预测值（之前存的），记录对比数据
            if surrogate_preds is not None:
                pred_list = self._normalize_objectives(surrogate_preds)
                self.surrogate_comparison_data.append({
                    'iteration': iteration,
                    'real_objectives': obj_list,
                    'pred_objectives': pred_list
                })
        
        # 自动计算帕累托前沿（如果外部没有传入）
        if pareto_objectives is None:
            pareto_objectives = self._compute_pareto_from_evaluations()
        
        # 更新帕累托前沿
        if pareto_objectives is not None and len(pareto_objectives) > 0:
            self.pareto_history.append(pareto_objectives.copy())
            
            # 计算超体积
            hv = self._compute_hypervolume(pareto_objectives)
            self.hypervolume_history.append(hv)
        
        # 根据设置生成图表
        if iteration % self.plot_interval == 0 or iteration <= self.plot_interval:
            self._generate_figures(iteration, pareto_params, pareto_objectives)
    
    def load_historical_evaluations(self, eval_path: str):
        """
        加载历史评估数据到可视化器
        
        Args:
            eval_path: evaluations.jsonl 文件路径
        """
        import json as _json
        if not os.path.isfile(eval_path):
            return
        
        count = 0
        with open(eval_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):
                if line.strip():
                    try:
                        record = _json.loads(line)
                        params = record.get('parameters', record.get('params', []))
                        objectives = record.get('objectives', {})
                        
                        # 转换为列表格式
                        if isinstance(params, dict):
                            param_list = list(params.values())
                        elif isinstance(params, (list, np.ndarray)):
                            param_list = list(params)
                        else:
                            continue
                        
                        if isinstance(objectives, dict):
                            obj_list = list(objectives.values())
                        elif isinstance(objectives, (list, np.ndarray)):
                            obj_list = list(objectives)
                        else:
                            continue
                        
                        self.evaluations.append({
                            'iteration': idx,
                            'params': param_list,
                            'objectives': obj_list
                        })
                        count += 1
                    except Exception:
                        continue
        
        if count > 0:
            print(f"[Visualizer] Loaded {count} historical evaluations for plotting")
    
    def _compute_pareto_from_evaluations(self) -> Optional[np.ndarray]:
        """从已有评估数据中自动计算帕累托前沿"""
        if len(self.evaluations) < 2:
            return None
        
        # 收集所有目标值
        obj_arrays = []
        for e in self.evaluations:
            obj = e['objectives']
            if isinstance(obj, list) and len(obj) >= 2:
                obj_arrays.append(np.array(obj, dtype=float))
        
        if len(obj_arrays) < 2:
            return None
        
        objectives_matrix = np.array(obj_arrays)
        
        # 简单的非支配排序
        n = len(objectives_matrix)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(i + 1, n):
                if is_dominated[j]:
                    continue
                # 检查 i 是否支配 j
                if np.all(objectives_matrix[i] <= objectives_matrix[j]) and np.any(objectives_matrix[i] < objectives_matrix[j]):
                    is_dominated[j] = True
                # 检查 j 是否支配 i
                elif np.all(objectives_matrix[j] <= objectives_matrix[i]) and np.any(objectives_matrix[j] < objectives_matrix[i]):
                    is_dominated[i] = True
                    break
        
        pareto_mask = ~is_dominated
        if np.sum(pareto_mask) == 0:
            return None
        
        return objectives_matrix[pareto_mask]
    
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

            # 4. 代理模型预测 vs 真实值对比
            if len(self.surrogate_comparison_data) > 0:
                self._plot_surrogate_comparison(iteration)
            
        except Exception as e:
            print(f"[WARN] Figure generation failed: {e}")
    
    def _plot_pareto_front(self, iteration: int, pareto_objectives: np.ndarray):
        """绘制帕累托前沿
        
        标记说明：
        - 蓝色圆点 (Blue circle): 真实仿真点
        - 红色叉号 (Red X): 纯代理预测点
        - 绿色三角 (Green triangle): Pareto 前沿解
        """
        if self.n_objectives == 2:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # 获取目标类型（用于判断是否需要取反显示）
            obj_targets = [obj.get('target', 'minimize') for obj in self.objectives]
            
            # 分离真实仿真点和纯代理预测点的目标值
            real_objs_list = [e['objectives'] for e in self.real_evaluations]
            surrogate_objs_list = [e['objectives'] for e in self.surrogate_only_evaluations]
            
            # 转换为数组并应用最大化目标取反
            def apply_target_transform(obj_list, obj_targets):
                if len(obj_list) == 0:
                    return np.array([]).reshape(0, 2)
                arr = np.array(obj_list)
                if arr.ndim != 2:
                    return np.array([]).reshape(0, 2)
                for i, target in enumerate(obj_targets):
                    if target == 'maximize' and i < arr.shape[1]:
                        arr[:, i] = -arr[:, i]
                return arr
            
            real_objs = apply_target_transform(real_objs_list, obj_targets)
            surrogate_objs = apply_target_transform(surrogate_objs_list, obj_targets)
            
            if len(real_objs) == 0 and len(surrogate_objs) == 0:
                plt.close(fig)
                return
            
            # 绘制真实仿真点（蓝色圆点）
            if len(real_objs) > 0:
                ax.scatter(real_objs[:, 0], real_objs[:, 1], c='blue', s=50, marker='o', 
                          alpha=0.7, label=f'Real ({len(real_objs)})')
            
            # 绘制纯代理预测点（红色叉号）
            if len(surrogate_objs) > 0:
                ax.scatter(surrogate_objs[:, 0], surrogate_objs[:, 1], c='red', s=50, marker='x', 
                          alpha=0.7, label=f'Surrogate ({len(surrogate_objs)})')
            
            # 帕累托前沿（绿色三角）
            if pareto_objectives is not None and len(pareto_objectives) > 0:
                pareto_display = pareto_objectives.copy()
                for i, target in enumerate(obj_targets):
                    if target == 'maximize' and i < pareto_display.shape[1]:
                        pareto_display[:, i] = -pareto_display[:, i]
                
                ax.scatter(pareto_display[:, 0], pareto_display[:, 1], c='green', s=100, marker='^', 
                          alpha=0.9, label=f'Pareto Front ({len(pareto_display)})', zorder=5)
                
                # 连接帕累托点
                sorted_idx = np.argsort(pareto_display[:, 0])
                ax.plot(pareto_display[sorted_idx, 0], pareto_display[sorted_idx, 1], 'g--', alpha=0.5, linewidth=2)
            
            obj_names = [obj.get('name', f'Obj{i}') for i, obj in enumerate(self.objectives)]
            ax.set_xlabel(obj_names[0])
            ax.set_ylabel(obj_names[1])
            ax.set_title(f'Pareto Front (Iteration {iteration})')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.figures_dir, f'pareto_front_iter{iteration}.png'), dpi=150)
            plt.close(fig)
        
        elif self.n_objectives == 3:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 分离真实仿真点和纯代理预测点
            real_objs_list = [e['objectives'] for e in self.real_evaluations]
            surrogate_objs_list = [e['objectives'] for e in self.surrogate_only_evaluations]
            
            real_objs = np.array(real_objs_list) if len(real_objs_list) > 0 else np.array([]).reshape(0, 3)
            surrogate_objs = np.array(surrogate_objs_list) if len(surrogate_objs_list) > 0 else np.array([]).reshape(0, 3)
            
            # 绘制真实仿真点（蓝色圆点）
            if len(real_objs) > 0:
                ax.scatter(real_objs[:, 0], real_objs[:, 1], real_objs[:, 2], 
                          c='blue', s=50, marker='o', alpha=0.7, label=f'Real ({len(real_objs)})')
            
            # 绘制纯代理预测点（红色叉号）
            if len(surrogate_objs) > 0:
                ax.scatter(surrogate_objs[:, 0], surrogate_objs[:, 1], surrogate_objs[:, 2], 
                          c='red', s=50, marker='x', alpha=0.7, label=f'Surrogate ({len(surrogate_objs)})')
            
            # 帕累托前沿（绿色三角）
            if pareto_objectives is not None and len(pareto_objectives) > 0:
                ax.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], pareto_objectives[:, 2], 
                          c='green', s=100, marker='^', alpha=0.9, label=f'Pareto Front ({len(pareto_objectives)})')
            
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
            plt.savefig(os.path.join(self.figures_dir, 'objectives.png'), dpi=150)
            plt.close(fig)

    def _plot_surrogate_comparison(self, iteration: int):
        """绘制代理模型预测 vs 真实仿真值对比图
        
        数据来源：self.surrogate_comparison_data
        每条记录包含同一 iteration 的真实仿真值和代理模型预测值，
        在真实仿真之前先用代理模型预测，然后用真实仿真验证。
        
        中间图（iteration < 9999）：只展示最近 self.surrogate_recent_window 次评估
        最终报告（iteration >= 9999）：展示所有累积数据
        """
        if self.n_objectives < 1 or len(self.surrogate_comparison_data) == 0:
            return

        try:
            # 确定使用哪些数据：中间图用最近N次，最终报告用全部
            if iteration >= 9999:
                # 最终报告：使用所有累积数据
                comp_data = self.surrogate_comparison_data
                filename = f'surrogate_comparison_iter{iteration}.png'
            else:
                # 中间图：只使用最近 N 次评估
                comp_data = self.surrogate_comparison_data[-self.surrogate_recent_window:]
                filename = 'surrogate_comparison.png'
            
            if len(comp_data) < 1:
                return

            fig, axes = plt.subplots(1, min(self.n_objectives, 3), figsize=(6 * min(self.n_objectives, 3), 5))
            if self.n_objectives == 1:
                axes = [axes]
            elif self.n_objectives > 3:
                axes = axes.flatten()

            obj_targets = [obj.get('target', 'minimize') for obj in self.objectives]

            for i, ax in enumerate(axes):
                if i >= self.n_objectives:
                    ax.set_visible(False)
                    continue

                iters = []
                real_vals = []
                pred_vals = []

                for d in comp_data:
                    real_obj = d['real_objectives']
                    pred_obj = d['pred_objectives']

                    if isinstance(real_obj, list) and len(real_obj) > i:
                        rv = real_obj[i]
                    else:
                        rv = None

                    if isinstance(pred_obj, list) and len(pred_obj) > i:
                        pv = pred_obj[i]
                    else:
                        pv = None

                    if rv is not None and pv is not None:
                        iters.append(d['iteration'])
                        if obj_targets[i] == 'maximize':
                            rv = -rv
                            pv = -pv
                        real_vals.append(rv)
                        pred_vals.append(pv)

                if len(iters) < 1:
                    ax.set_visible(False)
                    continue

                ax.plot(iters, real_vals, 'b-o', label='Real Simulation', markersize=4)
                if len(iters) >= 2:
                    ax.plot(iters, pred_vals, 'r--s', label='Surrogate Prediction', markersize=4)

                obj_name = self.objectives[i].get('name', f'Objective {i+1}') if i < len(self.objectives) else f'Objective {i+1}'
                ax.set_xlabel('Iteration')
                ax.set_ylabel(obj_name)
                ax.set_title(f'{obj_name}: Real vs Surrogate')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.surrogate_figures_dir, filename), dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"[WARN] Surrogate comparison plot failed: {e}")
    
    def _save_surrogate_comparison_data(self):
        """保存代理模型预测与真实值对比数据（JSON 和 CSV）"""
        if len(self.surrogate_comparison_data) == 0:
            return
        
        try:
            obj_names = [obj.get('name', f'Obj{i}') for i, obj in enumerate(self.objectives)]
            obj_targets = [obj.get('target', 'minimize') for obj in self.objectives]
            
            processed_data = []
            for d in self.surrogate_comparison_data:
                real_obj = d['real_objectives']
                pred_obj = d['pred_objectives']
                
                if isinstance(real_obj, list):
                    real_list = real_obj
                elif isinstance(real_obj, dict):
                    real_list = list(real_obj.values())
                else:
                    continue
                
                if isinstance(pred_obj, list):
                    pred_list = pred_obj
                elif isinstance(pred_obj, dict):
                    pred_list = list(pred_obj.values())
                else:
                    continue
                
                # 处理 maximize 目标的取反
                for i, target in enumerate(obj_targets):
                    if target == 'maximize' and i < len(real_list):
                        real_list[i] = -real_list[i] if real_list[i] < 0 else real_list[i]
                        pred_list[i] = -pred_list[i] if pred_list[i] < 0 else pred_list[i]
                
                # 计算误差
                abs_error = [abs(r - p) for r, p in zip(real_list, pred_list)]
                rel_error = []
                for r, e in zip(real_list, abs_error):
                    if abs(r) > 1e-8:
                        rel_error.append(e / abs(r) * 100)
                    else:
                        rel_error.append(0.0)
                
                processed_data.append({
                    'iteration': d['iteration'],
                    'real_objectives': real_list,
                    'pred_objectives': pred_list,
                    'absolute_error': abs_error,
                    'relative_error_percent': rel_error
                })
            
            # 保存 JSON
            json_data = {
                'timestamp': datetime.now().isoformat(),
                'n_records': len(processed_data),
                'objectives_names': obj_names,
                'data': processed_data
            }
            json_path = os.path.join(self.output_dir, 'surrogate_comparison_data.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            print(f"[OK] Saved surrogate comparison data (JSON): {json_path}")
            
            # 保存 CSV
            csv_path = os.path.join(self.output_dir, 'surrogate_comparison_data.csv')
            with open(csv_path, 'w', encoding='utf-8') as f:
                # CSV 表头
                header_parts = ['iteration']
                for name in obj_names:
                    header_parts.extend([f'real_{name}', f'pred_{name}', f'abs_err_{name}'])
                f.write(','.join(header_parts) + '\n')
                
                # CSV 数据行
                for d in processed_data:
                    row_parts = [str(d['iteration'])]
                    for i in range(len(obj_names)):
                        if i < len(d['real_objectives']):
                            row_parts.append(f"{d['real_objectives'][i]:.6f}")
                            row_parts.append(f"{d['pred_objectives'][i]:.6f}")
                            row_parts.append(f"{d['absolute_error'][i]:.6f}")
                        else:
                            row_parts.extend(['', '', ''])
                    f.write(','.join(row_parts) + '\n')
            
            print(f"[OK] Saved surrogate comparison data (CSV): {csv_path}")
            
        except Exception as e:
            print(f"[WARN] Failed to save surrogate comparison data: {e}")
    
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
        
        # 分离真实和预测解
        real_solutions = [s for s in pareto_solutions if not s.get('is_predicted', False)]
        pred_solutions = [s for s in pareto_solutions if s.get('is_predicted', False)]
        
        # 保存数据
        report = {
            'timestamp': datetime.now().isoformat(),
            'n_evaluations': stats.get('n_evaluations', 0),
            'pareto_size': len(pareto_solutions),
            'n_real_solutions': len(real_solutions),
            'n_predicted_solutions': len(pred_solutions),
            'best_objectives': stats.get('best_objectives', []),
            'real_solutions': real_solutions[:5],  # 真实解最多5个
            'predicted_solutions': pred_solutions[:5],  # 预测解最多5个
        }
        
        with open(os.path.join(self.output_dir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成文本报告
        self._generate_text_report(pareto_solutions, stats)
        
        # 保存代理模型对比数据（JSON 和 CSV）
        self._save_surrogate_comparison_data()
    
    def _generate_text_report(self, pareto_solutions: List[Dict], stats: Dict):
        """生成文本报告"""
        # 获取目标配置
        obj_names = [obj.get('name', f'Obj{i}') for i, obj in enumerate(self.objectives)]
        obj_targets = [obj.get('target', 'minimize') for obj in self.objectives]
        obj_weights = [obj.get('weight', 1.0) for i, obj in enumerate(self.objectives)]
        
        # 归一化权重
        total_weight = sum(obj_weights)
        obj_weights_norm = [w / total_weight for w in obj_weights]
        
        # 分离真实和预测解
        real_solutions = [s for s in pareto_solutions if not s.get('is_predicted', False)]
        pred_solutions = [s for s in pareto_solutions if s.get('is_predicted', False)]
        
        report_lines = [
            "=" * 70,
            "OPTIMIZATION REPORT",
            "=" * 70,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total evaluations: {stats.get('n_evaluations', 0)}",
            f"Pareto solutions: {len(pareto_solutions)}",
            f"Real solutions: {len(real_solutions)}",
            f"Predicted solutions: {len(pred_solutions)}",
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
        
        # 获取目标值（goal）
        obj_goals = [obj.get('goal', None) for obj in self.objectives]
        
        def calculate_score(sol):
            """计算单个解的得分"""
            objectives = sol.get('objectives', {})
            if isinstance(objectives, dict):
                obj_values = list(objectives.values())
            elif isinstance(objectives, list):
                obj_values = objectives
            else:
                obj_values = [objectives]
            
            obj_scores = []
            for j, val in enumerate(obj_values):
                if j >= len(obj_targets):
                    break
                target = obj_targets[j]
                goal = obj_goals[j] if j < len(obj_goals) else None
                weight = obj_weights_norm[j] if j < len(obj_weights_norm) else 1.0
                
                if target == 'maximize':
                    actual = -val if val < 0 else val
                else:
                    actual = val
                
                if goal is None:
                    score = 50
                elif target == 'minimize':
                    if actual <= goal:
                        score = 100
                    else:
                        excess_ratio = abs((actual - goal) / max(abs(goal), 1))
                        score = max(0, 100 - excess_ratio * 40)
                else:
                    if actual >= goal:
                        score = 100
                    else:
                        deficit_ratio = abs((goal - actual) / max(abs(goal), 1))
                        score = max(0, 100 - deficit_ratio * 40)
                
                obj_scores.append(score * weight)
            
            total_score = sum(obj_scores) / sum(obj_weights_norm) if sum(obj_weights_norm) > 0 else 0
            return total_score
        
        def append_solution_lines(report_lines, sol, rank, marker=""):
            """向报告追加单个解的信息"""
            objectives = sol.get('objectives', {})
            params = sol.get('parameters', [])
            
            if marker:
                report_lines.append(f"\nSolution {rank} (score: {calculate_score(sol):.1f}/100){marker}:")
            elif rank == 1:
                report_lines.append(f"\n★ BEST SOLUTION (rank {rank}, score: {calculate_score(sol):.1f}/100):")
            else:
                report_lines.append(f"\nSolution {rank} (score: {calculate_score(sol):.1f}/100):")
            
            report_lines.append("  Objectives:")
            if isinstance(objectives, dict):
                for name, val in objectives.items():
                    idx = obj_names.index(name) if name in obj_names else -1
                    if idx >= 0 and obj_targets[idx] == 'maximize':
                        display_val = -val if val < 0 else val
                    else:
                        display_val = val
                    
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
                for j, val in enumerate(objectives):
                    if j < len(obj_names):
                        name = obj_names[j]
                        target = obj_targets[j] if j < len(obj_targets) else 'minimize'
                        goal = obj_goals[j] if j < len(obj_goals) else None
                        
                        if target == 'maximize':
                            display_val = -val if val < 0 else val
                        else:
                            display_val = val
                        
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
            var_names = [v.get('name', f'Var{j}') for j, v in enumerate(self.variables)] if self.variables else [f'Var{j}' for j in range(len(params))]
            
            for j, val in enumerate(params):
                name = var_names[j] if j < len(var_names) else f'Var{j}'
                report_lines.append(f"    {name}: {val:.4f}")
        
        # 计算真实解得分并排序
        scored_real = [(calculate_score(s), s) for s in real_solutions]
        scored_real.sort(key=lambda x: -x[0])
        
        # 打印真实仿真解
        report_lines.append("")
        report_lines.append(f"--- REAL SIMULATION SOLUTIONS ({len(real_solutions)} solutions) ---")
        for rank, (score, sol) in enumerate(scored_real[:5], 1):
            append_solution_lines(report_lines, sol, rank)
        
        # 计算预测解得分并排序
        scored_pred = [(calculate_score(s), s) for s in pred_solutions]
        scored_pred.sort(key=lambda x: -x[0])
        
        # 打印预测解
        if pred_solutions:
            report_lines.append("")
            report_lines.append(f"--- PREDICTED SOLUTIONS ({len(pred_solutions)} solutions) ---")
            report_lines.append("    [NOTE: Predicted values - NOT yet verified by real simulation]")
            for rank, (score, sol) in enumerate(scored_pred[:5], 1):
                append_solution_lines(report_lines, sol, len(real_solutions) + rank, " [PREDICTED]")
        
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
