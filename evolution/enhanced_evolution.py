import asyncio
import time
import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional
from models.enhanced_reasoning_net import EnhancedReasoningNet
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

class EnhancedEvolution:
    """增强进化算法 - 真正的遗传进化"""
    
    def __init__(self, population_size: int = 10, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, elite_size: int = 2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # 进化历史
        self.evolution_history = []
        
        # 适应度缓存
        self.fitness_cache = {}
        
    async def evolve_population(self, population: List[EnhancedReasoningNet], 
                              fitness_scores: List[float]) -> List[EnhancedReasoningNet]:
        """进化种群"""
        logger.log_important("🔄 开始增强进化过程...")
        
        # 1. 选择
        selected = self._selection(population, fitness_scores)
        
        # 2. 交叉
        offspring = await self._crossover(selected)
        
        # 3. 变异
        mutated = self._mutation(offspring)
        
        # 4. 精英保留
        new_population = self._elitism(population, fitness_scores, mutated)
        
        # 5. 记录进化历史
        self._record_evolution(population, fitness_scores, new_population)
        
        return new_population
    
    def _selection(self, population: List[EnhancedReasoningNet], 
                  fitness_scores: List[float]) -> List[EnhancedReasoningNet]:
        """选择操作 - 锦标赛选择"""
        selected = []
        
        for _ in range(self.population_size):
            # 锦标赛选择
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # 选择最佳个体
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        logger.log_important(f"选择完成，选择了 {len(selected)} 个个体")
        return selected
    
    async def _crossover(self, selected: List[EnhancedReasoningNet]) -> List[EnhancedReasoningNet]:
        """交叉操作 - 参数交叉"""
        offspring = []
        
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                
                if random.random() < self.crossover_rate:
                    # 执行交叉
                    child1, child2 = self._parameter_crossover(parent1, parent2)
                    offspring.extend([child1, child2])
                else:
                    # 直接复制
                    offspring.extend([parent1, parent2])
            else:
                offspring.append(selected[i])
        
        logger.log_important(f"交叉完成，生成了 {len(offspring)} 个后代")
        return offspring
    
    def _parameter_crossover(self, parent1: EnhancedReasoningNet, 
                           parent2: EnhancedReasoningNet) -> Tuple[EnhancedReasoningNet, EnhancedReasoningNet]:
        """参数交叉"""
        # 创建子代 - 使用相同的架构参数
        child1 = type(parent1)(
            parent1.input_size,
            parent1.hidden_size,
            parent1.reasoning_layers,
            parent1.attention_heads
        )
        
        child2 = type(parent1)(  # 使用parent1的架构参数确保兼容性
            parent1.input_size,
            parent1.hidden_size,
            parent1.reasoning_layers,
            parent1.attention_heads
        )
        
        # 交叉参数 - 只对相同尺寸的参数进行交叉
        parent1_params = list(parent1.parameters())
        parent2_params = list(parent2.parameters())
        child1_params = list(child1.parameters())
        child2_params = list(child2.parameters())
        
        for i, (param1, param2, child_param1, child_param2) in enumerate(
            zip(parent1_params, parent2_params, child1_params, child2_params)
        ):
            with torch.no_grad():
                # 检查参数尺寸是否匹配
                if param1.shape == param2.shape:
                    # 随机交叉点
                    crossover_point = random.random()
                    
                    # 混合参数
                    child_param1.copy_(crossover_point * param1 + (1 - crossover_point) * param2)
                    child_param2.copy_((1 - crossover_point) * param1 + crossover_point * param2)
                else:
                    # 尺寸不匹配时，直接复制parent1的参数
                    child_param1.copy_(param1)
                    child_param2.copy_(param1)
        
        return child1, child2
    
    def _mutation(self, offspring: List[EnhancedReasoningNet]) -> List[EnhancedReasoningNet]:
        """变异操作 - 自适应变异"""
        mutated = []
        
        for individual in offspring:
            if random.random() < self.mutation_rate:
                # 执行变异
                mutated_individual = self._adaptive_mutation(individual)
                mutated.append(mutated_individual)
            else:
                mutated.append(individual)
        
        logger.log_important(f"变异完成，变异了 {len([m for m in mutated if m != offspring[mutated.index(m)]])} 个个体")
        return mutated
    
    def _adaptive_mutation(self, individual: EnhancedReasoningNet) -> EnhancedReasoningNet:
        """自适应变异"""
        # 创建变异个体
        mutated = type(individual)(
            individual.input_size,
            individual.hidden_size,
            individual.reasoning_layers,
            individual.attention_heads
        )
        
        # 复制参数
        individual_params = list(individual.parameters())
        mutated_params = list(mutated.parameters())
        
        for param, mutated_param in zip(individual_params, mutated_params):
            with torch.no_grad():
                # 自适应变异强度
                mutation_strength = 0.01 * (1 + random.random())
                
                # 高斯变异
                noise = torch.randn_like(param) * mutation_strength
                mutated_param.copy_(param + noise)
        
        return mutated
    
    def _elitism(self, population: List[EnhancedReasoningNet], 
                fitness_scores: List[float], 
                offspring: List[EnhancedReasoningNet]) -> List[EnhancedReasoningNet]:
        """精英保留"""
        # 排序种群
        sorted_indices = np.argsort(fitness_scores)[::-1]  # 降序
        
        # 保留精英
        elite = [population[i] for i in sorted_indices[:self.elite_size]]
        
        # 从后代中选择剩余个体
        remaining_size = self.population_size - self.elite_size
        selected_offspring = random.sample(offspring, min(remaining_size, len(offspring)))
        
        new_population = elite + selected_offspring
        
        logger.log_important(f"精英保留完成，保留了 {len(elite)} 个精英个体")
        return new_population
    
    def _record_evolution(self, old_population: List[EnhancedReasoningNet], 
                         fitness_scores: List[float], 
                         new_population: List[EnhancedReasoningNet]):
        """记录进化历史"""
        generation_info = {
            'generation': len(self.evolution_history) + 1,
            'timestamp': time.time(),
            'best_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'population_size': len(new_population),
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate
        }
        
        self.evolution_history.append(generation_info)
        
        logger.log_important(f"进化历史记录: 第{generation_info['generation']}代, "
                           f"最佳适应度: {generation_info['best_fitness']:.3f}, "
                           f"平均适应度: {generation_info['avg_fitness']:.3f}")
    
    def adaptive_parameters(self, generation: int, best_fitness: float, avg_fitness: float):
        """自适应参数调整"""
        # 根据进化进度调整参数
        if generation > 10:
            # 降低变异率
            self.mutation_rate = max(0.01, self.mutation_rate * 0.95)
            
            # 如果适应度停滞，增加变异
            if generation > 20 and best_fitness - avg_fitness < 0.01:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
        
        logger.log_important(f"自适应参数调整: 变异率={self.mutation_rate:.3f}, "
                           f"交叉率={self.crossover_rate:.3f}")

class MultiObjectiveEvolution:
    """多目标进化算法"""
    
    def __init__(self, population_size: int = 10):
        self.population_size = population_size
        self.evolution = EnhancedEvolution(population_size)
        
    async def evolve_multi_objective(self, population: List[EnhancedReasoningNet], 
                                   objectives: Dict[str, List[float]]) -> List[EnhancedReasoningNet]:
        """多目标进化"""
        logger.log_important("🎯 开始多目标进化...")
        
        # 计算帕累托前沿
        pareto_front = self._calculate_pareto_front(objectives)
        
        # 计算拥挤度距离
        crowding_distances = self._calculate_crowding_distance(objectives, pareto_front)
        
        # 基于拥挤度距离的选择
        selected_indices = self._tournament_selection_with_crowding(
            objectives, crowding_distances, self.population_size
        )
        
        # 选择个体
        selected_population = [population[i] for i in selected_indices]
        
        # 执行进化
        evolved_population = await self.evolution.evolve_population(
            selected_population, 
            [crowding_distances[i] for i in selected_indices]
        )
        
        return evolved_population
    
    def _calculate_pareto_front(self, objectives: Dict[str, List[float]]) -> List[int]:
        """计算帕累托前沿"""
        pareto_front = []
        num_individuals = len(list(objectives.values())[0])
        
        for i in range(num_individuals):
            dominated = False
            
            for j in range(num_individuals):
                if i != j:
                    # 检查是否被支配
                    if self._dominates(objectives, j, i):
                        dominated = True
                        break
            
            if not dominated:
                pareto_front.append(i)
        
        return pareto_front
    
    def _dominates(self, objectives: Dict[str, List[float]], i: int, j: int) -> bool:
        """检查个体i是否支配个体j"""
        at_least_one_better = False
        
        for objective_name, values in objectives.items():
            if values[i] < values[j]:  # 假设所有目标都是最小化
                return False
            elif values[i] > values[j]:
                at_least_one_better = True
        
        return at_least_one_better
    
    def _calculate_crowding_distance(self, objectives: Dict[str, List[float]], 
                                   pareto_front: List[int]) -> List[float]:
        """计算拥挤度距离"""
        num_individuals = len(list(objectives.values())[0])
        crowding_distances = [0.0] * num_individuals
        
        for objective_name, values in objectives.items():
            # 对每个目标排序
            sorted_indices = sorted(range(len(values)), key=lambda k: values[k])
            
            # 边界个体的拥挤度距离设为无穷大
            crowding_distances[sorted_indices[0]] = float('inf')
            crowding_distances[sorted_indices[-1]] = float('inf')
            
            # 计算中间个体的拥挤度距离
            for i in range(1, len(sorted_indices) - 1):
                distance = (values[sorted_indices[i + 1]] - values[sorted_indices[i - 1]]) / \
                          (max(values) - min(values) + 1e-10)
                crowding_distances[sorted_indices[i]] += distance
        
        return crowding_distances
    
    def _tournament_selection_with_crowding(self, objectives: Dict[str, List[float]], 
                                          crowding_distances: List[float], 
                                          selection_size: int) -> List[int]:
        """基于拥挤度距离的锦标赛选择"""
        selected_indices = []
        
        for _ in range(selection_size):
            # 随机选择两个个体
            idx1, idx2 = random.sample(range(len(crowding_distances)), 2)
            
            # 选择拥挤度距离更大的个体
            if crowding_distances[idx1] > crowding_distances[idx2]:
                selected_indices.append(idx1)
            else:
                selected_indices.append(idx2)
        
        return selected_indices 