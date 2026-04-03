"""
NSGA-II 多目标优化算法
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from .base import BaseOptimizer


class NSGA2(BaseOptimizer):
    """NSGA-II 多目标优化器"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # NSGA-II 特定参数
        self.population_size = config.get('population_size', 20)
        self.n_generations = config.get('n_generations', 10)
        
        # 交叉变异参数
        self.crossover_prob = config.get('crossover_prob', 0.9)
        self.mutation_prob = config.get('mutation_prob', None)  # 默认 1/n_vars
        self.eta_c = config.get('eta_c', 20)  # SBX 分布指数
        self.eta_m = config.get('eta_m', 20)  # 多项式变异分布指数
    
    def run(self, evaluator) -> List[Dict]:
        """
        运行 NSGA-II 优化
        
        Args:
            evaluator: 目标评估器
            
        Returns:
            Pareto 前沿解列表
        """
        print("\n" + "=" * 60)
        print("NSGA-II MULTIOBJECTIVE OPTIMIZATION")
        print(f"Population: {self.population_size}, Generations: {self.n_generations}")
        print("=" * 60)
        
        # 初始化种群
        population = self._initialize_population()
        objectives = []
        all_results = []
        
        # 评估初始种群
        print("\n[Generation 0] Evaluating initial population...")
        for i, ind in enumerate(population):
            result = self._evaluate_with_cache(ind, evaluator)
            if result is not None:
                objectives.append(result[0])
                all_results.append(result[1])
                self._print_result(i + 1, result[1])
            else:
                objectives.append([1000] * len(self.objectives))
                all_results.append(None)
        
        # 主循环
        for gen in range(self.n_generations):
            print(f"\n[Generation {gen + 1}/{self.n_generations}]")
            
            # 非支配排序
            fronts = self.fast_non_dominated_sort(population, objectives)
            
            # 生成子代
            offspring = self._generate_offspring(population, objectives, fronts)
            
            # 评估子代
            offspring_objectives = []
            offspring_results = []
            
            for i, ind in enumerate(offspring):
                result = self._evaluate_with_cache(ind, evaluator)
                if result is not None:
                    offspring_objectives.append(result[0])
                    offspring_results.append(result[1])
                    self._print_result(i + 1, result[1], prefix="Offspring")
                else:
                    offspring_objectives.append([1000] * len(self.objectives))
                    offspring_results.append(None)
            
            # 合并选择
            population, objectives, all_results = self._select_next_generation(
                population, objectives, all_results,
                offspring, offspring_objectives, offspring_results
            )

            # 早停检查
            if self.should_stop_early(all_results):
                goals_count = self.count_solutions_meeting_goals(all_results)
                print(f"\n[INFO] Early stop: {goals_count} solutions meet goals (threshold: {self.n_solutions_to_stop})")
                break
            
            # 打印 Pareto 前沿
            current_fronts = self.fast_non_dominated_sort(population, objectives)
            if current_fronts and current_fronts[0]:
                print(f"  Pareto front: {len(current_fronts[0])} solutions")
        
        # 整理结果
        fronts = self.fast_non_dominated_sort(population, objectives)
        pareto_params = self._extract_pareto_params(fronts[0], population, all_results)
        
        return pareto_params
    
    def get_statistics(self) -> Dict:
        return {
            'total_evaluations': self.evaluation_count,
            'real_evaluations': self.real_evaluation_count,
            'cache_size': len(self._cache),
        }
    
    def _initialize_population(self) -> List[np.ndarray]:
        """初始化种群"""
        bounds = self.get_bounds()
        population = []
        
        for _ in range(self.population_size):
            individual = np.array([
                np.random.uniform(bounds[i, 0], bounds[i, 1])
                for i in range(len(self.variables))
            ])
            population.append(individual)
        
        return population
    
    def _evaluate_with_cache(self, params: np.ndarray, evaluator) -> Optional[Tuple]:
        """带缓存的评估"""
        # 检查缓存
        cached = self._check_cache(params)
        if cached is not None:
            return cached
        
        self.evaluation_count += 1
        self.real_evaluation_count += 1
        
        # 设置变量
        for i, var in enumerate(self.variables):
            evaluator.hfss.set_variable(var['name'], params[i], var.get('unit', 'mm'))
        
        # 运行仿真
        if not evaluator.hfss.analyze(force=True):
            print(f"[WARN] Analysis failed for params: {params}")
            return None
        
        # 评估目标
        evaluator.clear_cache()
        result = evaluator.evaluate_all(params)
        
        # 添加到缓存
        self._add_to_cache(params, result)
        
        return result
    
    def _generate_offspring(self, population, objectives, fronts) -> List[np.ndarray]:
        """生成子代"""
        offspring = []
        selected = self._selection(population, objectives, fronts)
        
        n_vars = len(self.variables)
        mut_prob = self.mutation_prob or (1.0 / n_vars)
        
        while len(offspring) < self.population_size:
            # 锦标赛选择
            idx1, idx2 = np.random.choice(len(selected), 2, replace=False)
            parent1, parent2 = selected[idx1], selected[idx2]
            
            # 交叉
            if np.random.random() < self.crossover_prob:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # 变异
            child1 = self._mutation(child1, mut_prob)
            child2 = self._mutation(child2, mut_prob)
            
            offspring.extend([child1, child2])
        
        return offspring[:self.population_size]
    
    def _selection(self, population, objectives, fronts) -> List[np.ndarray]:
        """选择操作"""
        selected = []
        
        for front in fronts:
            if len(selected) + len(front) <= self.population_size:
                selected.extend([population[i] for i in front])
            else:
                remaining = self.population_size - len(selected)
                distances = self.crowding_distance(front, objectives)
                sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                selected.extend([population[i] for i, _ in sorted_front[:remaining]])
                break
        
        return selected
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """SBX 交叉"""
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(len(parent1)):
            if np.random.random() < 0.5:
                child1[i], child2[i] = parent1[i], parent2[i]
            else:
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1.0 / (self.eta_c + 1))
                else:
                    beta = (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta_c + 1))
                
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        
        return self.clip_to_bounds(child1), self.clip_to_bounds(child2)
    
    def _mutation(self, individual: np.ndarray, prob: float) -> np.ndarray:
        """多项式变异"""
        mutated = individual.copy()
        bounds = self.get_bounds()
        
        for i in range(len(mutated)):
            if np.random.random() < prob:
                y = mutated[i]
                lower, upper = bounds[i]
                
                delta1 = (y - lower) / (upper - lower)
                delta2 = (upper - y) / (upper - lower)
                
                r = np.random.random()
                if r < 0.5:
                    xy = 1 - delta1
                    val = 2 * r + (1 - 2 * r) * (xy ** (self.eta_m + 1))
                    deltaq = val ** (1.0 / (self.eta_m + 1)) - 1
                else:
                    xy = 1 - delta2
                    val = 2 * (1 - r) + 2 * (r - 0.5) * (xy ** (self.eta_m + 1))
                    deltaq = 1 - val ** (1.0 / (self.eta_m + 1))
                
                mutated[i] = np.clip(y + deltaq * (upper - lower), lower, upper)
        
        return mutated
    
    def _select_next_generation(self, pop, obj, results, off_pop, off_obj, off_results):
        """选择下一代"""
        combined_pop = pop + off_pop
        combined_obj = obj + off_obj
        combined_results = results + off_results
        
        fronts = self.fast_non_dominated_sort(combined_pop, combined_obj)
        
        new_pop, new_obj, new_results = [], [], []
        
        for front in fronts:
            if len(new_pop) + len(front) <= self.population_size:
                for idx in front:
                    new_pop.append(combined_pop[idx])
                    new_obj.append(combined_obj[idx])
                    new_results.append(combined_results[idx])
            else:
                remaining = self.population_size - len(new_pop)
                distances = self.crowding_distance(front, combined_obj)
                sorted_front = sorted(zip(front, distances), key=lambda x: x[1], reverse=True)
                for idx, _ in sorted_front[:remaining]:
                    new_pop.append(combined_pop[idx])
                    new_obj.append(combined_obj[idx])
                    new_results.append(combined_results[idx])
                break
        
        return new_pop, new_obj, new_results
    
    def _extract_pareto_params(self, front: List[int], population, all_results) -> List[Dict]:
        """提取 Pareto 前沿参数"""
        pareto_params = []
        
        for idx in front:
            params = population[idx]
            results = all_results[idx]
            
            param_dict = {
                v['name']: round(float(params[i]), 4)
                for i, v in enumerate(self.variables)
            }
            
            obj_dict = {}
            if results:
                for name, res in results.items():
                    # 兼容字典和对象格式
                    if isinstance(res, dict):
                        goal_met = res.get('goal_met')
                        actual_val = res.get('actual_value', 0)
                        val = res.get('value', 0)
                    else:
                        goal_met = res.goal_met
                        actual_val = res.actual_value
                        val = res.value
                    
                    # 转换为 Python 原生类型
                    obj_dict[name] = {
                        'value': float(val) if val is not None else 0.0,
                        'actual_value': float(actual_val) if actual_val is not None else 0.0,
                        'goal_met': bool(goal_met) if goal_met is not None else None,
                    }
            
            pareto_params.append({
                'params': param_dict,
                'objectives': obj_dict,
            })
        
        return pareto_params
    
    def _print_result(self, idx: int, results: Dict, prefix: str = "Ind"):
        """打印结果"""
        parts = []
        for name, res in results.items():
            # 兼容字典和对象格式
            if isinstance(res, dict):
                goal_met = res.get('goal_met')
                actual = res.get('actual_value', 0)
            else:
                goal_met = res.goal_met
                actual = res.actual_value
            
            # 使用 ASCII 字符避免 GBK 编码错误
            if goal_met is True:
                s = '[OK]'
            elif goal_met is False:
                s = '[FAIL]'
            else:
                s = ''
            parts.append(f"{name}={actual:.2f}{s}")
        print(f"  {prefix} {idx}: {', '.join(parts)}")