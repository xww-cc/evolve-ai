import asyncio
import time
import torch
import numpy as np
import random
from typing import List, Dict, Tuple, Any, Optional
from models.advanced_reasoning_net import AdvancedReasoningNet
from evaluators.enhanced_evaluator import EnhancedEvaluator
from config.optimized_logging import setup_optimized_logging
from utils.visualization import EvolutionVisualizer

logger = setup_optimized_logging()

class AdvancedEvolution:
    """高级进化算法 - 支持异构结构和可视化"""
    
    def __init__(self, population_size: int = 8, mutation_rate: float = 0.1, 
                 crossover_rate: float = 0.8, elite_size: int = 2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.diversity_threshold = 0.1  # 添加多样性阈值
        self.evolution_history = []
        self.visualizer = EvolutionVisualizer()
        
        # 自适应参数
        self.adaptive_mutation_rate = mutation_rate
        self.adaptive_crossover_rate = crossover_rate
        
    async def evolve_population(self, population: List[AdvancedReasoningNet], 
                              fitness_scores: List[float]) -> List[AdvancedReasoningNet]:
        """进化种群 - 异步版本"""
        # 选择
        selected = self._multi_objective_selection(population, fitness_scores)
        
        # 交叉
        offspring = await self._advanced_crossover(selected)
        
        # 变异
        offspring = self._intelligent_mutation(offspring)
        
        # 精英保留
        new_population = self._elitism_with_diversity(population, fitness_scores, offspring)
        
        return new_population
    
    def evolve(self, population: List[AdvancedReasoningNet], 
               evaluator: EnhancedEvaluator, generations: int = 5) -> List[AdvancedReasoningNet]:
        """执行高级进化 - 同步版本，集成可视化"""
        logger.log_important("🔄 开始高级进化过程...")
        
        for generation in range(generations):
            # 计算适应度
            fitness_scores = []
            for individual in population:
                fitness = self._calculate_fitness(individual, evaluator)
                fitness_scores.append(fitness)
            
            # 计算统计信息
            best_fitness = max(fitness_scores)
            avg_fitness = np.mean(fitness_scores)
            diversity = self._calculate_diversity(population)
            
            # 记录可视化数据
            self.visualizer.record_generation(
                generation=generation + 1,
                population=population,
                fitness_scores=fitness_scores,
                diversity=diversity,
                best_fitness=best_fitness,
                avg_fitness=avg_fitness
            )
            
            # 自适应参数调整
            self._adaptive_parameter_adjustment(diversity, fitness_scores)
            
            # 多目标选择
            selected_parents = self._multi_objective_selection(population, fitness_scores)
            
            # 生成后代
            offspring = []
            for i in range(0, len(selected_parents), 2):
                if i + 1 < len(selected_parents):
                    parent1, parent2 = selected_parents[i], selected_parents[i + 1]
                    child1, child2 = self._advanced_parameter_crossover(parent1, parent2)
                    offspring.extend([child1, child2])
            
            logger.log_important(f"🔔 高级交叉完成，生成了 {len(offspring)} 个后代")
            
            # 智能变异
            mutated_count = 0
            for i, individual in enumerate(offspring):
                if random.random() < self.adaptive_mutation_rate:
                    offspring[i] = self._intelligent_parameter_mutation(individual)
                    mutated_count += 1
            
            logger.log_important(f"🔔 智能变异完成，变异了 {mutated_count} 个个体")
            
            # 精英保留与多样性维护
            elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
            elite_individuals = [population[i] for i in elite_indices]
            
            # 合并精英和后代
            new_population = elite_individuals + offspring[:self.population_size - self.elite_size]
            
            # 确保种群大小
            while len(new_population) < self.population_size:
                new_population.append(random.choice(offspring))
            
            population = new_population[:self.population_size]
            
            logger.log_important(f"🔔 精英保留与多样性维护完成，保留了 {len(elite_individuals)} 个精英个体")
            
            # 记录进化历史
            self.evolution_history.append({
                'generation': generation + 1,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'diversity': diversity
            })
            
            logger.log_important(f"🔔 高级进化历史记录: 第{generation + 1}代, 最佳适应度: {best_fitness:.3f}, 平均适应度: {avg_fitness:.3f}, 多样性: {diversity:.3f}")
        
        # 生成可视化
        self._generate_visualizations()
        
        return population
    
    def _generate_visualizations(self):
        """生成可视化图表"""
        try:
            # 绘制进化曲线
            curves_file = self.visualizer.plot_evolution_curves()
            logger.log_important(f"📊 进化曲线已保存: {curves_file}")
            
            # 绘制多样性热力图
            heatmap_file = self.visualizer.plot_diversity_heatmap()
            logger.log_important(f"📊 多样性热力图已保存: {heatmap_file}")
            
            # 生成进化报告
            report_file = self.visualizer.generate_evolution_report()
            logger.log_important(f"📊 进化报告已保存: {report_file}")
            
            # 保存可视化数据
            data_file = self.visualizer.save_visualization_data()
            logger.log_important(f"📊 可视化数据已保存: {data_file}")
            
        except Exception as e:
            logger.log_warning(f"可视化生成失败: {e}")
    
    def _calculate_fitness(self, model: AdvancedReasoningNet, evaluator: EnhancedEvaluator) -> float:
        """计算适应度分数"""
        try:
            # 异步评估
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果已经在事件循环中，直接调用
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, evaluator.evaluate_enhanced_reasoning(model, max_tasks=5))
                    results = future.result()
            else:
                results = loop.run_until_complete(evaluator.evaluate_enhanced_reasoning(model, max_tasks=5))
            
            # 使用综合推理分数作为适应度
            fitness = results.get('comprehensive_reasoning', 0.0)
            return fitness
            
        except Exception as e:
            logger.log_warning(f"适应度计算失败: {e}")
            return 0.0
    
    def _calculate_diversity(self, population: List[AdvancedReasoningNet]) -> float:
        """结构+参数+行为多样性加权"""
        try:
            # 结构多样性
            def structure_vec(model):
                return np.array([
                    model.hidden_size,
                    model.reasoning_layers,
                    model.attention_heads,
                    model.memory_size,
                    model.reasoning_types
                ], dtype=np.float32)
            
            structure_distances = []
            for i in range(len(population)):
                for j in range(i+1, len(population)):
                    s1, s2 = structure_vec(population[i]), structure_vec(population[j])
                    structure_distances.append(np.linalg.norm(s1-s2))
            structure_div = np.mean(structure_distances) if structure_distances else 0.0
            
            # 参数多样性
            feature_vectors = []
            for model in population:
                params = list(model.parameters())
                param_features = []
                for param in params:
                    param_features.extend([
                        param.mean().item(),
                        param.std().item(),
                        param.max().item(),
                        param.min().item()
                    ])
                feature_vectors.append(param_features)
            
            param_distances = []
            for i in range(len(feature_vectors)):
                for j in range(i+1, len(feature_vectors)):
                    if len(feature_vectors[i]) == len(feature_vectors[j]):
                        param_distances.append(np.linalg.norm(np.array(feature_vectors[i])-np.array(feature_vectors[j])))
            param_div = np.mean(param_distances) if param_distances else 0.0
            
            # 行为多样性（简化版本，避免维度不匹配）
            behavior_outputs = []
            test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
            
            for model in population:
                try:
                    model.eval()
                    with torch.no_grad():
                        output = model(test_input)
                        # 使用comprehensive_reasoning作为行为特征
                        if 'comprehensive_reasoning' in output:
                            behavior_score = output['comprehensive_reasoning'].mean().item()
                        else:
                            behavior_score = 0.0
                        behavior_outputs.append(behavior_score)
                except Exception as e:
                    # 如果模型推理失败，使用默认值
                    behavior_outputs.append(0.0)
            
            behavior_distances = []
            for i in range(len(behavior_outputs)):
                for j in range(i+1, len(behavior_outputs)):
                    behavior_distances.append(abs(behavior_outputs[i] - behavior_outputs[j]))
            behavior_div = np.mean(behavior_distances) if behavior_distances else 0.0
            
            # 加权
            alpha, beta, gamma = 0.3, 0.4, 0.3
            diversity = alpha*structure_div + beta*param_div + gamma*behavior_div
            return diversity
            
        except Exception as e:
            logger.log_warning(f"多样性计算失败: {e}")
            return 0.0
    
    def _adaptive_parameter_adjustment(self, diversity: float, fitness_scores: List[float]):
        """自适应参数调整"""
        # 根据多样性调整变异率
        if diversity < self.diversity_threshold:
            self.adaptive_mutation_rate = min(0.3, self.adaptive_mutation_rate * 1.2)
        else:
            self.adaptive_mutation_rate = max(0.05, self.adaptive_mutation_rate * 0.95)
        
        # 根据适应度分布调整交叉率
        fitness_std = np.std(fitness_scores)
        if fitness_std < 0.1:  # 适应度集中
            self.adaptive_crossover_rate = min(0.95, self.adaptive_crossover_rate * 1.1)
        else:
            self.adaptive_crossover_rate = max(0.7, self.adaptive_crossover_rate * 0.95)
        
        logger.log_important(f"自适应参数调整: 变异率={self.adaptive_mutation_rate:.3f}, "
                           f"交叉率={self.adaptive_crossover_rate:.3f}, 多样性={diversity:.3f}")
    
    def _multi_objective_selection(self, population: List[AdvancedReasoningNet], 
                                 fitness_scores: List[float]) -> List[AdvancedReasoningNet]:
        """多目标选择"""
        selected = []
        
        # 1. 基于适应度的选择
        fitness_selected = self._fitness_based_selection(population, fitness_scores, 
                                                       int(self.population_size * 0.6))
        
        # 2. 基于多样性的选择
        diversity_selected = self._diversity_based_selection(population, fitness_scores,
                                                           int(self.population_size * 0.4))
        
        selected.extend(fitness_selected)
        selected.extend(diversity_selected)
        
        logger.log_important(f"多目标选择完成: 适应度选择={len(fitness_selected)}, "
                           f"多样性选择={len(diversity_selected)}")
        return selected
    
    def _fitness_based_selection(self, population: List[AdvancedReasoningNet], 
                                fitness_scores: List[float], num_select: int) -> List[AdvancedReasoningNet]:
        """基于适应度的选择"""
        selected = []
        
        for _ in range(num_select):
            # 锦标赛选择
            tournament_size = min(5, len(population))
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # 选择最佳个体
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        
        return selected
    
    def _diversity_based_selection(self, population: List[AdvancedReasoningNet], 
                                 fitness_scores: List[float], num_select: int) -> List[AdvancedReasoningNet]:
        """基于多样性的选择"""
        selected = []
        
        # 计算每个个体的多样性贡献
        diversity_scores = []
        for i, model in enumerate(population):
            # 计算与其他个体的平均距离
            distances = []
            for j, other_model in enumerate(population):
                if i != j:
                    try:
                        # 使用结构参数计算距离，避免参数维度不匹配
                        structure_dist = abs(model.hidden_size - other_model.hidden_size) + \
                                      abs(model.reasoning_layers - other_model.reasoning_layers) + \
                                      abs(model.attention_heads - other_model.attention_heads) + \
                                      abs(model.memory_size - other_model.memory_size) + \
                                      abs(model.reasoning_types - other_model.reasoning_types)
                        distances.append(structure_dist)
                    except Exception as e:
                        # 如果计算失败，使用默认距离
                        distances.append(1.0)
            
            diversity_scores.append(np.mean(distances) if distances else 0)
        
        # 选择多样性最高的个体
        diversity_indices = np.argsort(diversity_scores)[::-1]
        for i in range(min(num_select, len(diversity_indices))):
            selected.append(population[diversity_indices[i]])
        
        return selected
    
    async def _advanced_crossover(self, selected: List[AdvancedReasoningNet]) -> List[AdvancedReasoningNet]:
        """高级交叉操作"""
        offspring = []
        
        for i in range(0, len(selected), 2):
            if i + 1 < len(selected):
                parent1 = selected[i]
                parent2 = selected[i + 1]
                
                if random.random() < self.adaptive_crossover_rate:
                    # 执行高级交叉
                    child1, child2 = self._advanced_parameter_crossover(parent1, parent2)
                    offspring.extend([child1, child2])
                else:
                    # 直接复制
                    offspring.extend([parent1, parent2])
            else:
                offspring.append(selected[i])
        
        logger.log_important(f"高级交叉完成，生成了 {len(offspring)} 个后代")
        return offspring
    
    def _advanced_parameter_crossover(self, parent1: AdvancedReasoningNet, parent2: AdvancedReasoningNet) -> Tuple[AdvancedReasoningNet, AdvancedReasoningNet]:
        """异构结构下的高级参数交叉"""
        # 随机选择父代结构参数
        def pick_structure(p1, p2):
            return np.random.choice([p1, p2])
        
        # 确保hidden_size能被attention_heads整除
        def adjust_hidden_size(hidden_size, attention_heads):
            return (hidden_size // attention_heads) * attention_heads
        
        # 创建子代，使用相同的结构以避免维度不匹配
        child1_hidden = adjust_hidden_size(
            pick_structure(parent1.hidden_size, parent2.hidden_size),
            pick_structure(parent1.attention_heads, parent2.attention_heads)
        )
        child2_hidden = adjust_hidden_size(
            pick_structure(parent1.hidden_size, parent2.hidden_size),
            pick_structure(parent1.attention_heads, parent2.attention_heads)
        )
        
        child1 = type(parent1)(
            parent1.input_size,
            child1_hidden,
            pick_structure(parent1.reasoning_layers, parent2.reasoning_layers),
            pick_structure(parent1.attention_heads, parent2.attention_heads),
            pick_structure(parent1.memory_size, parent2.memory_size),
            pick_structure(parent1.reasoning_types, parent2.reasoning_types)
        )
        child2 = type(parent1)(
            parent2.input_size,
            child2_hidden,
            pick_structure(parent1.reasoning_layers, parent2.reasoning_layers),
            pick_structure(parent1.attention_heads, parent2.attention_heads),
            pick_structure(parent1.memory_size, parent2.memory_size),
            pick_structure(parent1.reasoning_types, parent2.reasoning_types)
        )
        
        # 参数交叉 - 只处理形状兼容的参数
        parent1_params = list(parent1.parameters())
        parent2_params = list(parent2.parameters())
        child1_params = list(child1.parameters())
        child2_params = list(child2.parameters())
        
        for i in range(min(len(child1_params), len(parent1_params), len(parent2_params))):
            param1, param2 = parent1_params[i], parent2_params[i]
            child_param1, child_param2 = child1_params[i], child2_params[i]
            
            with torch.no_grad():
                # 只处理形状完全匹配的参数
                if param1.shape == param2.shape and param1.shape == child_param1.shape:
                    crossover_point = np.random.rand()
                    noise1 = torch.randn_like(param1) * 0.01
                    noise2 = torch.randn_like(param2) * 0.01
                    child_param1.copy_(crossover_point * (param1 + noise1) + (1 - crossover_point) * (param2 + noise2))
                    child_param2.copy_((1 - crossover_point) * (param1 + noise1) + crossover_point * (param2 + noise2))
                else:
                    # 形状不兼容时，使用随机初始化
                    # 这样可以避免维度不匹配问题
                    pass  # 保持默认初始化
        
        return child1, child2

    def _intelligent_mutation(self, offspring: List[AdvancedReasoningNet]) -> List[AdvancedReasoningNet]:
        """智能变异操作"""
        mutated = []
        
        for individual in offspring:
            if random.random() < self.adaptive_mutation_rate:
                # 执行智能变异
                mutated_individual = self._intelligent_parameter_mutation(individual)
                mutated.append(mutated_individual)
            else:
                mutated.append(individual)
        
        logger.log_important(f"智能变异完成，变异了 {len([m for m in mutated if m != offspring[mutated.index(m)]])} 个个体")
        return mutated
    
    def _intelligent_parameter_mutation(self, individual: AdvancedReasoningNet) -> AdvancedReasoningNet:
        """异构结构下的智能参数变异，允许结构参数小概率变异"""
        # 结构参数小概率变异
        def maybe_mutate(val, choices, prob=0.05):  # 降低结构变异概率
            return int(np.random.choice(choices)) if np.random.rand() < prob else val
        
        # 确保hidden_size能被attention_heads整除
        def adjust_hidden_size(hidden_size, attention_heads):
            return (hidden_size // attention_heads) * attention_heads
        
        mutated_hidden = adjust_hidden_size(
            maybe_mutate(individual.hidden_size, [128, 192, 256, 320, 384, 512]),
            maybe_mutate(individual.attention_heads, [4, 8, 16])
        )
        
        mutated = type(individual)(
            individual.input_size,
            mutated_hidden,
            maybe_mutate(individual.reasoning_layers, [3, 4, 5, 6]),
            maybe_mutate(individual.attention_heads, [4, 8, 16]),
            maybe_mutate(individual.memory_size, [10, 15, 20, 25, 30]),
            maybe_mutate(individual.reasoning_types, [8, 10, 12, 15])
        )
        
        individual_params = list(individual.parameters())
        mutated_params = list(mutated.parameters())
        
        for i in range(min(len(individual_params), len(mutated_params))):
            param, mutated_param = individual_params[i], mutated_params[i]
            
            with torch.no_grad():
                # 只处理形状匹配的参数
                if param.shape == mutated_param.shape:
                    mutation_strength = 0.01 * (1 + np.random.rand()) * (1 + param.std().item())
                    noise = torch.randn_like(param) * mutation_strength
                    mutated_param.copy_(param + noise)
                else:
                    # 形状不兼容时，保持默认初始化
                    pass  # 避免复杂的截断和填充操作
        
        return mutated
    
    def _elitism_with_diversity(self, population: List[AdvancedReasoningNet], 
                               fitness_scores: List[float], 
                               offspring: List[AdvancedReasoningNet]) -> List[AdvancedReasoningNet]:
        """精英保留与多样性维护"""
        # 排序种群
        sorted_indices = np.argsort(fitness_scores)[::-1]  # 降序
        
        # 保留精英
        elite = [population[i] for i in sorted_indices[:self.elite_size]]
        
        # 从后代中选择剩余个体，考虑多样性
        remaining_size = self.population_size - self.elite_size
        selected_offspring = self._diversity_based_selection(offspring, 
                                                          [0.5] * len(offspring), 
                                                          remaining_size)
        
        new_population = elite + selected_offspring
        
        logger.log_important(f"精英保留与多样性维护完成，保留了 {len(elite)} 个精英个体")
        return new_population
    
    def _record_advanced_evolution(self, old_population: List[AdvancedReasoningNet], 
                                 fitness_scores: List[float], 
                                 new_population: List[AdvancedReasoningNet],
                                 diversity_score: float):
        """记录高级进化历史"""
        generation_info = {
            'generation': len(self.evolution_history) + 1,
            'timestamp': time.time(),
            'best_fitness': max(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'fitness_std': np.std(fitness_scores),
            'population_size': len(new_population),
            'diversity_score': diversity_score,
            'adaptive_mutation_rate': self.adaptive_mutation_rate,
            'adaptive_crossover_rate': self.adaptive_crossover_rate
        }
        
        self.evolution_history.append(generation_info)
        
        logger.log_important(f"高级进化历史记录: 第{generation_info['generation']}代, "
                           f"最佳适应度: {generation_info['best_fitness']:.3f}, "
                           f"平均适应度: {generation_info['avg_fitness']:.3f}, "
                           f"多样性: {generation_info['diversity_score']:.3f}")

class MultiObjectiveAdvancedEvolution:
    """多目标高级进化算法"""
    
    def __init__(self, population_size: int = 15):
        self.population_size = population_size
        self.evolution = AdvancedEvolution(population_size)
        
    async def evolve_multi_objective(self, population: List[AdvancedReasoningNet], 
                                   objectives: Dict[str, List[float]]) -> List[AdvancedReasoningNet]:
        """多目标高级进化"""
        logger.log_important("🎯 开始多目标高级进化...")
        
        # 计算帕累托前沿
        pareto_front = self._calculate_pareto_front(objectives)
        
        # 计算拥挤度距离
        crowding_distances = self._calculate_crowding_distance(objectives, pareto_front)
        
        # 基于拥挤度距离的选择
        selected_indices = self._advanced_tournament_selection(
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
    
    def _advanced_tournament_selection(self, objectives: Dict[str, List[float]], 
                                     crowding_distances: List[float], 
                                     selection_size: int) -> List[int]:
        """高级锦标赛选择"""
        selected_indices = []
        
        for _ in range(selection_size):
            # 随机选择多个个体
            tournament_size = min(7, len(crowding_distances))
            tournament_indices = random.sample(range(len(crowding_distances)), tournament_size)
            tournament_distances = [crowding_distances[i] for i in tournament_indices]
            
            # 选择拥挤度距离最大的个体
            winner_idx = tournament_indices[np.argmax(tournament_distances)]
            selected_indices.append(winner_idx)
        
        return selected_indices 