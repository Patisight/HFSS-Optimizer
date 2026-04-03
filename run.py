#!/usr/bin/env python
"""
HFSS 天线优化程序 - 主入口
支持多种优化算法，统一配置管理

使用方法:
    python run.py --algorithm nsga2
    python run.py --algorithm surrogate
    python run.py --config my_config.py
"""
import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# 导入模块
from core import HFSSController, ObjectiveEvaluator, format_results
from utils import OptimizationVisualizer
from config import get_default_config, validate_config


def setup_logging(output_dir: str) -> str:
    """设置日志"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # 同时输出到文件和控制台
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def clear_old_results(config: dict):
    """清除旧结果"""
    import shutil
    
    project_path = config['hfss']['project_path']
    results_dir = project_path.replace('.aedt', '.aedtresults')
    
    if os.path.exists(results_dir):
        print(f"[INFO] Clearing old results: {results_dir}")
        try:
            shutil.rmtree(results_dir)
            print("[OK] Old results cleared")
        except Exception as e:
            print(f"[WARN] Could not clear: {e}")


def load_evaluations_to_file(source_path: str, output_dir: str) -> int:
    """
    加载历史评估数据到输出目录的 evaluations.jsonl
    
    如果输出目录中已有 evaluations.jsonl，则跳过（避免重复加载）。
    
    Args:
        source_path: 历史数据文件路径 (jsonl 格式)
        output_dir: 本次运行输出目录
        
    Returns:
        成功加载的记录数
    """
    import shutil
    
    eval_file = os.path.join(output_dir, 'evaluations.jsonl')
    
    # 如果目标文件已存在，说明已经在之前的运行中加载过，跳过
    if os.path.isfile(eval_file):
        existing_count = sum(1 for line in open(eval_file, 'r', encoding='utf-8') if line.strip())
        print(f"[OK] evaluations.jsonl already exists with {existing_count} records, skipping load")
        return existing_count
    
    count = 0
    
    with open(source_path, 'r', encoding='utf-8') as src:
        with open(eval_file, 'w', encoding='utf-8') as dst:
            for line in src:
                if line.strip():
                    record = json.loads(line)
                    # 验证必要字段
                    if 'parameters' in record and 'objectives' in record:
                        dst.write(line)
                        count += 1
    
    return count


def run_optimization(config: dict, algorithm: str = 'surrogate'):
    """
    运行优化
    
    Args:
        config: 配置字典
        algorithm: 算法类型 ('nsga2' 或 'surrogate')
    """
    # 验证配置
    if not validate_config(config):
        raise ValueError("Invalid configuration")
    
    # 设置输出目录
    output_dir = config['run'].get('output_dir', './optim_results')
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"{algorithm}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    
    # 设置日志
    log_file = setup_logging(run_dir)
    print(f"[OK] Log file: {log_file}")
    
    # 保存配置
    config_file = os.path.join(run_dir, 'config.json')
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[OK] Config saved: {config_file}")
    
    # 清除旧结果
    if config['run'].get('clear_old_results', False):
        clear_old_results(config)
    
    # 连接 HFSS
    print("\n" + "=" * 60)
    print("HFSS ANTENNA OPTIMIZATION")
    print(f"Algorithm: {algorithm}")
    print("=" * 60)
    
    hfss = HFSSController(
        config['hfss']['project_path'],
        config['hfss']['design_name'],
        config['hfss'].get('setup_name', 'Setup1'),
        config['hfss'].get('sweep_name', 'Sweep'),
    )
    
    if not hfss.connect():
        raise RuntimeError("Failed to connect to HFSS")
    
    try:
        # 创建评估器（传入输出目录以保存每次仿真数据）
        evaluator = ObjectiveEvaluator(config['objectives'], hfss, output_dir=run_dir)
        
        # 加载历史评估数据（用于预训练代理模型或继续优化）
        load_eval_path = config.get('algorithm', {}).get('load_evaluations')
        if load_eval_path and os.path.isfile(load_eval_path):
            try:
                loaded = load_evaluations_to_file(load_eval_path, run_dir)
                print(f"[OK] Loaded {loaded} historical evaluations from: {load_eval_path}")
                print(f"[OK] Copied to: {os.path.join(run_dir, 'evaluations.jsonl')}")
            except Exception as e:
                print(f"[WARN] Failed to load evaluations: {e}")
        
        # 创建优化器
        algo_config = {**config['algorithm'], **config}
        
        algorithm = config['algorithm'].get('algorithm', 'mobo')
        
        # 计算总评估次数（用于进度条）
        pop_size = config['algorithm'].get('population_size', 30)
        n_gen = config['algorithm'].get('n_generations', 30)
        n_init = config['algorithm'].get('n_initial_samples', pop_size)
        
        if algorithm == 'mobo':
            total_evals = n_init + config['algorithm'].get('n_iterations', 50)
        else:
            total_evals = pop_size + pop_size * n_gen
        
        print(f"[PROGRESS] TOTAL:{total_evals}")
        
        if algorithm == 'mobo':
            from algorithms import MultiObjectiveBayesianOptimizer
            optimizer = MultiObjectiveBayesianOptimizer(algo_config)
        elif algorithm == 'mopso':
            from algorithms import MOPSO
            optimizer = MOPSO(algo_config)
        elif algorithm == 'robust':
            from algorithms import RobustSurrogateOptimizer
            optimizer = RobustSurrogateOptimizer(algo_config)
        elif algorithm == 'adaptive':
            from algorithms import AdaptiveOptimizer
            optimizer = AdaptiveOptimizer(algo_config)
        elif algorithm == 'surrogate':
            from algorithms import SurrogateAssistedNSGA2
            optimizer = SurrogateAssistedNSGA2(algo_config)
        else:
            from algorithms import NSGA2
            optimizer = NSGA2(algo_config)
        
        # 创建可视化器
        viz_config = config.get('visualization', {})
        plot_interval = viz_config.get('plot_interval', 5)
        surrogate_recent_window = viz_config.get('surrogate_recent_window', 5)
        visualizer = OptimizationVisualizer(
            run_dir, config['objectives'], config.get('variables', []), 
            plot_interval, surrogate_recent_window
        )
        
        # 加载历史评估数据到可视化器（用于图表展示）
        load_eval_path = config.get('algorithm', {}).get('load_evaluations')
        if load_eval_path and os.path.isfile(load_eval_path):
            visualizer.load_historical_evaluations(load_eval_path)
        
        # 定义回调函数 - 用于迭代过程中更新可视化
        iteration_count = [0]  # 用列表包装以便在闭包中修改
        
        def progress_callback(current, total, params, objectives, phase, surrogate_preds=None, is_surrogate=False):
            """迭代回调：更新进度和可视化"""
            iteration_count[0] += 1
            
            # 输出进度信息（便于GUI解析）
            print(f"[PROGRESS] {iteration_count[0]}")
            
            # 确保 objectives 是一维列表（防御性处理）
            try:
                if hasattr(objectives, 'flatten'):
                    obj_list = objectives.flatten().tolist()
                elif hasattr(objectives, 'tolist'):
                    obj_list = objectives.tolist()
                elif isinstance(objectives, dict):
                    obj_list = list(objectives.values())
                elif isinstance(objectives, list):
                    obj_list = list(objectives)  # 确保是列表而非生成器
                else:
                    obj_list = [float(objectives)] if not isinstance(objectives, str) else [objectives]
                # 验证 obj_list 包含的是数值
                obj_list = [float(x) if not isinstance(x, (int, float)) else x for x in obj_list]
            except (ValueError, TypeError, AttributeError) as e:
                print(f"[WARN] Failed to normalize objectives ({type(objectives)}): {e}")
                obj_list = [0.0]  # 默认值
            
            # 确保 surrogate_preds 是一维列表（防御性处理）
            surrogate_list = None
            if surrogate_preds is not None:
                try:
                    if hasattr(surrogate_preds, 'flatten'):
                        surrogate_list = surrogate_preds.flatten().tolist()
                    elif hasattr(surrogate_preds, 'tolist'):
                        surrogate_list = surrogate_preds.tolist()
                    elif isinstance(surrogate_preds, dict):
                        surrogate_list = list(surrogate_preds.values())
                    elif isinstance(surrogate_preds, (list, tuple)):
                        surrogate_list = list(surrogate_preds)
                    else:
                        surrogate_list = [float(surrogate_preds)] if not isinstance(surrogate_preds, str) else [surrogate_preds]
                    # 验证 surrogate_list 包含的是数值
                    surrogate_list = [float(x) if not isinstance(x, (int, float)) else x for x in surrogate_list]
                except (ValueError, TypeError, AttributeError) as e:
                    print(f"[WARN] Failed to normalize surrogate_preds ({type(surrogate_preds)}): {e}")
                    surrogate_list = None
            
            # 更新可视化
            try:
                visualizer.update(iteration_count[0], params, obj_list, surrogate_preds=surrogate_list, is_surrogate_prediction=is_surrogate)
            except Exception as e:
                print(f"[WARN] Visualization update failed: {e}")
        
        # 运行优化（传递回调）
        start_time = time.time()
        pareto_params = optimizer.run(evaluator, callback=progress_callback)
        elapsed = time.time() - start_time
        
        # 获取统计信息
        stats = optimizer.get_statistics()
        stats['elapsed_time'] = f"{elapsed:.1f}s"
        
        # 生成报告 (generate_final_report 会自动处理可视化)
        visualizer.generate_final_report(pareto_params, stats)
        
        # 保存结果
        results_file = os.path.join(run_dir, 'results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(pareto_params, f, indent=2, ensure_ascii=False)
        
        # 打印摘要
        print("\n" + "=" * 60)
        print("OPTIMIZATION COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed:.1f}s")
        for key, val in stats.items():
            print(f"  {key}: {val}")
        
        print(f"\nPareto front: {len(pareto_params)} solutions")
        print(f"Results saved to: {run_dir}")
        
        # 分离真实仿真解和预测解
        real_solutions = [s for s in pareto_params if not s.get('is_predicted', False)]
        pred_solutions = [s for s in pareto_params if s.get('is_predicted', False)]
        
        obj_names = [obj.get('name', f'Obj{i}') for i, obj in enumerate(config.get('objectives', []))]
        obj_targets = [obj.get('target', 'minimize') for obj in config.get('objectives', [])]
        var_names = [var.get('name', f'Var{i}') for i, var in enumerate(config.get('variables', []))]
        
        def print_solution(sol, idx, marker=""):
            """打印单个解"""
            print(f"\n  Solution {idx}{marker}:")
            params = sol.get('parameters', sol.get('params', []))
            objectives = sol.get('objectives', {})
            
            if isinstance(params, list):
                print(f"    Parameters:")
                for j, val in enumerate(params):
                    name = var_names[j] if j < len(var_names) else f'Var{j}'
                    print(f"      {name}: {val:.4f}")
            else:
                print(f"    Params: {params}")
            
            if isinstance(objectives, list):
                print(f"    Objectives:")
                for j, val in enumerate(objectives):
                    name = obj_names[j] if j < len(obj_names) else f'Obj{j}'
                    target = obj_targets[j] if j < len(obj_targets) else 'minimize'
                    actual_val = -val if target == 'maximize' else val
                    print(f"      {name}: {actual_val:.4f}")
            elif isinstance(objectives, dict):
                for name, res in objectives.items():
                    if isinstance(res, dict):
                        s = '[OK]' if res.get('goal_met') else '[FAIL]' if res.get('goal_met') is not None else ''
                        print(f"    {name}: {res.get('actual_value', 0):.3f} {s}")
                    else:
                        print(f"    {name}: {res:.4f}")
        
        # 打印真实仿真解
        print(f"\n--- REAL SIMULATION SOLUTIONS ({len(real_solutions)} solutions) ---")
        for i, sol in enumerate(real_solutions[:5]):
            print_solution(sol, i + 1, " [REAL]")
        
        # 打印预测解
        if pred_solutions:
            print(f"\n--- PREDICTED SOLUTIONS ({len(pred_solutions)} solutions) ---")
            print(f"    [NOTE: Predicted values - NOT yet verified by real simulation]")
            for i, sol in enumerate(pred_solutions[:5]):
                print_solution(sol, len(real_solutions) + i + 1, " [PREDICTED]")
        
        print("\n" + "=" * 60)
        
        return pareto_params
        
    finally:
        hfss.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='HFSS Antenna Optimizer')
    parser.add_argument('--algorithm', '-a', 
                        choices=['nsga2', 'surrogate', 'mopso', 'mobo', 'robust', 'adaptive'],
                        default='mopso',
                        help='Optimization algorithm (nsga2, mopso, mobo, surrogate, robust, adaptive)')
    parser.add_argument('--config', '-c',
                        type=str,
                        default=None,
                        help='Path to custom config file')
    parser.add_argument('--population', '-p',
                        type=int,
                        default=20,
                        help='Population size')
    parser.add_argument('--generations', '-g',
                        type=int,
                        default=10,
                        help='Number of generations')
    parser.add_argument('--initial-samples',
                        type=int,
                        default=50,
                        help='Initial samples for surrogate')
    
    args = parser.parse_args()
    
    # 获取配置
    if args.config:
        if args.config.endswith('.json'):
            # 加载 JSON 配置
            with open(args.config, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            # 加载 Python 配置
            import importlib.util
            spec = importlib.util.spec_from_file_location("custom_config", args.config)
            custom_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_config)
            config = custom_config.config
    else:
        config = get_default_config()
    
    # 命令行参数覆盖（仅当配置中没有设置时才使用默认值）
    algo = config.setdefault('algorithm', {})
    if 'population_size' not in algo:
        algo['population_size'] = args.population
    if 'n_generations' not in algo:
        algo['n_generations'] = args.generations
    if 'initial_samples' not in algo:
        algo['initial_samples'] = args.initial_samples
    
    # 运行优化（优先使用配置文件里的算法）
    algorithm = algo.get('algorithm', args.algorithm)
    
    try:
        run_optimization(config, algorithm)
    except KeyboardInterrupt:
        print("\n[WARN] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()