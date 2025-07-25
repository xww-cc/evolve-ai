#!/usr/bin/env python3
"""
增强版进化算法
包含自适应参数调整、性能优化和更好的收敛性
"""

import torch
import numpy as np
import asyncio
from typing import List, Tuple, Dict, Optional
from evolution.nsga2 import evolve_population_nsga2_simple
from evolution.population import create_initial_population
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evaluators.realworld_evaluator import RealWorldEvaluator
from utils.performance_monitor import get_performance_monitor
from config.logging_setup import setup_logging
from config.global_constants import LEVEL_DESCRIPTIONS
import time

logger = setup_logging()

class AdaptiveEvolutionParameters:
    """自适应进化参数"""
    
    def __init__(self):
        self.mutation_rate = 0.1
        self.crossover_rate = 0.8
        self.selection_pressure = 2.0
        self.diversity_threshold = 0.3
        self.stagnation_threshold = 10
        self.performance_threshold = 0.8
        
        # 参数调整历史
        self.adjustment_history = []
    
    def adjust_parameters(self, 
                         current_diversity: float,
                         stagnation_count: int,
                         performance_ratio: float,
                         generation: int) -> Dict:
        """根据当前状态调整参数"""
        adjustments = {}
        
        # 根据多样性调整
        if current_diversity < self.diversity_threshold:
            self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
            self.crossover_rate = max(0.6, self.crossover_rate * 0.9)
            adjustments['mutation_rate'] = self.mutation_rate
            adjustments['crossover_rate'] = self.crossover_rate
            logger.info(f"多样性较低，增加变异率: {self.mutation_rate:.3f}")
        
        # 根据停滞调整
        if stagnation_count > self.stagnation_threshold:
            self.mutation_rate = min(0.4, self.mutation_rate * 1.5)
            self.selection_pressure = max(1.0, self.selection_pressure * 0.8)
            adjustments['mutation_rate'] = self.mutation_rate
            adjustments['selection_pressure'] = self.selection_pressure
            logger.info(f"检测到停滞，大幅增加变异率: {self.mutation_rate:.3f}")
        
        # 根据性能调整
        if performance_ratio < self.performance_threshold:
            self.crossover_rate = max(0.5, self.crossover_rate * 0.95)
            adjustments['crossover_rate'] = self.crossover_rate
            logger.info(f"性能较低，减少交叉率: {self.crossover_rate:.3f}")
        
        # 记录调整历史
        if adjustments:
            self.adjustment_history.append({
                'generation': generation,
                'diversity': current_diversity,
                'stagnation_count': stagnation_count,
                'performance_ratio': performance_ratio,
                'adjustments': adjustments.copy()
            })
        
        return adjustments

class EnhancedEvolutionEngine:
    """增强版进化引擎"""
    
    def __init__(self):
        self.symbolic_evaluator = SymbolicEvaluator()
        self.realworld_evaluator = RealWorldEvaluator()
        self.adaptive_params = AdaptiveEvolutionParameters()
        self.performance_monitor = get_performance_monitor()
        
        # 进化历史
        self.evolution_history = []
        self.best_fitness_history = []
        self.diversity_history = []
        
        # 停滞检测
        self.stagnation_count = 0
        self.last_best_fitness = 0.0
        
    async def evolve_population_enhanced(self, 
                                       population: List,
                                       num_generations: int,
                                       level: int = 0,
                                       population_size: Optional[int] = None) -> Tuple[List, List, List]:
        """增强版种群进化"""
        
        # 启动性能监控
        self.performance_monitor.start_monitoring()
        
        avg_scores = []
        best_scores = []
        diversity_scores = []
        
        current_population = population
        if population_size is None:
            population_size = len(population)
        
        # 处理边界情况
        if num_generations <= 0:
            logger.info("零代数进化，返回初始种群")
            return current_population, avg_scores, best_scores
        
        if len(population) == 0:
            logger.info("空种群，返回空结果")
            return current_population, avg_scores, best_scores
        
        logger.info(f"开始增强进化 - 种群大小: {population_size}, 代数: {num_generations}, 级别: {level}")
        
        for generation in range(num_generations):
            generation_start_time = time.time()
            
            # 评估种群
            fitness_scores = await self._evaluate_population_parallel(current_population, level)
            
            # 计算统计信息
            avg_score = np.mean([sum(score) for score in fitness_scores])
            best_score = max([sum(score) for score in fitness_scores])
            diversity = self._calculate_diversity(current_population)
            
            avg_scores.append(avg_score)
            best_scores.append(best_score)
            diversity_scores.append(diversity)
            
            # 检查停滞
            if best_score <= self.last_best_fitness:
                self.stagnation_count += 1
            else:
                self.stagnation_count = 0
                self.last_best_fitness = best_score
            
            # 自适应参数调整
            performance_ratio = best_score / (avg_score + 1e-8)
            adjustments = self.adaptive_params.adjust_parameters(
                diversity, self.stagnation_count, performance_ratio, generation
            )
            
            # 记录进化历史
            self.evolution_history.append({
                'generation': generation,
                'avg_score': avg_score,
                'best_score': best_score,
                'diversity': diversity,
                'stagnation_count': self.stagnation_count,
                'performance_ratio': performance_ratio,
                'adjustments': adjustments
            })
            
            # 性能监控
            generation_time = time.time() - generation_start_time
            evolution_speed = 1.0 / generation_time if generation_time > 0 else 0
            self.performance_monitor.record_metrics(
                evolution_speed=evolution_speed,
                evaluation_throughput=len(current_population) / generation_time if generation_time > 0 else 0
            )
            
            logger.info(f"第 {generation + 1} 代 - 平均: {avg_score:.4f}, 最佳: {best_score:.4f}, "
                       f"多样性: {diversity:.3f}, 停滞: {self.stagnation_count}")
            
            # 进化到下一代
            current_population = evolve_population_nsga2_simple(current_population, fitness_scores)
            
            # 如果停滞太久，注入多样性
            if self.stagnation_count > self.adaptive_params.stagnation_threshold * 2:
                current_population = self._inject_diversity(current_population, population_size)
                logger.info("注入多样性以打破停滞")
        
        # 生成性能报告
        performance_report = self.performance_monitor.generate_performance_report()
        logger.info(f"性能报告已生成: {performance_report}")
        
        return current_population, avg_scores, best_scores
    
    async def _evaluate_population_parallel(self, population: List, level: int) -> List[Tuple[float, float]]:
        """并行评估种群"""
        tasks = []
        for individual in population:
            task = asyncio.create_task(self._evaluate_individual(individual, level))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def _evaluate_individual(self, individual, level: int) -> Tuple[float, float]:
        """评估单个个体"""
        try:
            symbolic_score = await self.symbolic_evaluator.evaluate(individual, level=level)
            realworld_score = await self.realworld_evaluator.evaluate(individual)
            return (symbolic_score, realworld_score)
        except Exception as e:
            logger.warning(f"个体评估失败: {e}")
            return (0.1, 0.1)  # 返回最低分数
    
    def _calculate_diversity(self, population: List) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
        
        # 计算结构多样性
        structures = []
        for individual in population:
            structure = {
                'num_modules': len(individual.subnet_modules),
                'total_params': sum(p.numel() for p in individual.parameters()),
                'module_types': [getattr(module, 'module_type', 'unknown') for module in individual.subnet_modules]
            }
            structures.append(str(structure))
        
        unique_structures = len(set(structures))
        return unique_structures / len(population)
    
    def _inject_diversity(self, population: List, target_size: int) -> List:
        """注入多样性"""
        # 保留一些优秀个体
        elite_size = max(1, target_size // 4)
        elite = population[:elite_size]
        
        # 生成新的随机个体
        new_individuals = create_initial_population(target_size - elite_size)
        
        # 合并精英和新个体
        diverse_population = elite + new_individuals
        
        logger.info(f"注入多样性: 保留 {elite_size} 个精英, 生成 {len(new_individuals)} 个新个体")
        return diverse_population
    
    def get_evolution_summary(self) -> Dict:
        """获取进化摘要"""
        if not self.evolution_history:
            return {"error": "无进化历史"}
        
        latest = self.evolution_history[-1]
        first = self.evolution_history[0]
        
        improvement = latest['best_score'] - first['best_score']
        avg_diversity = np.mean([h['diversity'] for h in self.evolution_history])
        
        return {
            'total_generations': len(self.evolution_history),
            'final_best_score': latest['best_score'],
            'final_avg_score': latest['avg_score'],
            'final_diversity': latest['diversity'],
            'total_improvement': improvement,
            'average_diversity': avg_diversity,
            'stagnation_count': latest['stagnation_count'],
            'parameter_adjustments': len(self.adaptive_params.adjustment_history)
        }
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        if not self.evolution_history:
            return ["无进化数据，无法提供建议"]
        
        latest = self.evolution_history[-1]
        
        # 基于停滞的建议
        if latest['stagnation_count'] > 5:
            suggestions.append("检测到进化停滞，建议：")
            suggestions.append("  - 增加变异率")
            suggestions.append("  - 减少种群大小以提高选择压力")
            suggestions.append("  - 重新初始化部分个体")
        
        # 基于多样性的建议
        if latest['diversity'] < 0.3:
            suggestions.append("种群多样性较低，建议：")
            suggestions.append("  - 增加交叉率")
            suggestions.append("  - 引入更多随机个体")
            suggestions.append("  - 调整选择策略")
        
        # 基于性能的建议
        if latest['performance_ratio'] < 0.8:
            suggestions.append("性能提升缓慢，建议：")
            suggestions.append("  - 优化评估函数")
            suggestions.append("  - 调整适应度计算")
            suggestions.append("  - 增加精英保留比例")
        
        return suggestions

# 全局增强进化引擎实例
enhanced_evolution_engine = EnhancedEvolutionEngine()

def get_enhanced_evolution_engine() -> EnhancedEvolutionEngine:
    """获取增强进化引擎实例"""
    return enhanced_evolution_engine 