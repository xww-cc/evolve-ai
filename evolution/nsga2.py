import random
import logging
from typing import List, Tuple, Dict
from evolution.population import create_initial_population
from optimizers.autophagy import cellular_autophagy_model
from evaluators.realworld_evaluator import RealWorldEvaluator
from optimizers.mutation import mutate_modular_net, crossover_modular_nets
from utils.parallel_utils import parallel_map
from evolution.stagnation_detector import detect_stagnation
from config.global_constants import *
from config.logging_setup import setup_logging, get_logger
import numpy as np
from models.modular_net import ModularMathReasoningNet
from torch.distributions import Normal
import asyncio
import torch

logger = setup_logging()

def non_dominated_sort(population_with_objectives: List[Tuple[ModularMathReasoningNet, List[float]]]) -> List[List[ModularMathReasoningNet]]:
    """非支配排序 - 完整"""
    fronts = []
    n_p = [0] * len(population_with_objectives)
    S_p = [[] for _ in range(len(population_with_objectives))]
    current_front = []

    for i in range(len(population_with_objectives)):
        p_obj = population_with_objectives[i][1]
        for j in range(len(population_with_objectives)):
            if i == j: continue
            q_obj = population_with_objectives[j][1]

            p_dominates_q = all(p_obj[k] >= q_obj[k] for k in range(len(p_obj))) and any(p_obj[k] > q_obj[k] for k in range(len(p_obj)))
            q_dominates_p = all(q_obj[k] >= p_obj[k] for k in range(len(p_obj))) and any(q_obj[k] > p_obj[k] for k in range(len(p_obj)))
            
            if p_dominates_q:
                S_p[i].append(j)
            elif q_dominates_p:
                n_p[i] += 1

        if n_p[i] == 0:
            current_front.append(i)
            
    fronts.append(current_front)

    f_idx = 0
    while len(fronts[f_idx]) > 0:
        next_front = []
        for p_idx in fronts[f_idx]:
            for q_idx in S_p[p_idx]:
                n_p[q_idx] -= 1
                if n_p[q_idx] == 0:
                    next_front.append(q_idx)
        f_idx += 1
        fronts.append(next_front)
    
    return [[population_with_objectives[idx][0] for idx in front] for front in fronts if front]

def calculate_crowding_distance(front_individuals_with_objectives: List[Tuple[ModularMathReasoningNet, List[float]]]) -> Dict[ModularMathReasoningNet, float]:
    """计算拥挤距离 - 完整"""
    distances = {ind[0]: 0.0 for ind in front_individuals_with_objectives}
    num_objectives = len(front_individuals_with_objectives[0][1])

    for obj_idx in range(num_objectives):
        sorted_front = sorted(front_individuals_with_objectives, key=lambda x: x[1][obj_idx])
        
        distances[sorted_front[0][0]] = float('inf')
        distances[sorted_front[-1][0]] = float('inf')

        if len(sorted_front) > 2:
            obj_min = sorted_front[0][1][obj_idx]
            obj_max = sorted_front[-1][1][obj_idx]
            
            if obj_max == obj_min:
                continue

            for i in range(1, len(sorted_front) - 1):
                distances[sorted_front[i][0]] += (sorted_front[i+1][1][obj_idx] - sorted_front[i-1][1][obj_idx]) / (obj_max - obj_min)
    
    return distances

async def evolve_population_nsga2(population: List[ModularMathReasoningNet], num_generations: int = NUM_GENERATIONS, level: int = 6) -> Tuple[List[ModularMathReasoningNet], List[float], List[float]]:
    """生态系统进化循环 (NSGA-II) - 完整"""
    logger.info(f"开始NSGA-II进化 - 级别: {level}, 世代数: {num_generations}, 种群大小: {len(population)}")
    
    history_avg_score = []
    history_best_score = []
    
    evaluator = RealWorldEvaluator()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"使用设备: {device}")

    for gen in range(num_generations):
        logger.info(f"=== 第 {gen+1} 世代开始 ===")
        
        # 细胞自噬阶段
        logger.debug("开始细胞自噬处理")
        autophagy_tasks = [model.get_state() for model in population]
        autophagy_results = parallel_map(cellular_autophagy_model, autophagy_tasks)
        
        for i, model_state in enumerate(autophagy_results):
            population[i].load_state(model_state)
        logger.debug("细胞自噬处理完成")

        # 评估阶段
        logger.debug("开始种群评估")
        eval_tasks = [model.get_state() for model in population] 
        results_objectives = []
        for state in eval_tasks:
            result = await evaluator.evaluate_model_real_world(state, device, level)
            results_objectives.append(result)

        population_with_objectives = []
        for i, model in enumerate(population):
            if results_objectives[i][0] > -float('inf'): 
                population_with_objectives.append((model, results_objectives[i]))
        
        # 记录评估结果
        logger.info(f"第 {gen+1} 世代评估结果 - 有效个体数: {len(population_with_objectives)}/{len(population)}")
        if len(population_with_objectives) > 0:
            best_score = max([obj[0] for _, obj in population_with_objectives])
            avg_score = np.mean([obj[0] for _, obj in population_with_objectives])
            logger.info(f"  最佳得分: {best_score:.4f}")
            logger.info(f"  平均得分: {avg_score:.4f}")
            history_avg_score.append(avg_score)
            history_best_score.append(best_score)
        else:
            logger.warning("没有有效个体，重新创建种群")
            population = create_initial_population(len(population))
            continue

        # 停滞检测
        is_stagnated = detect_stagnation(history_avg_score)
        if is_stagnated:
            logger.info(f"检测到停滞，应用环境压力")
            for model in population:
                model.epigenetic_markers.data[0] += ENVIRONMENT_PRESSURE_STRENGTH 
                model.epigenetic_markers.clamp() 

        # 生成子代
        logger.debug("开始生成子代")
        offspring_population = []
        num_offspring = len(population)
        
        initial_mutation_strength_base = 0.15
        min_mutation_strength_base = 0.02
        base_mutation_strength = initial_mutation_strength_base * (1 - gen / (num_generations - 1)) if num_generations > 1 else initial_mutation_strength_base
        base_mutation_strength = max(min_mutation_strength_base, base_mutation_strength)

        for offspring_idx in range(num_offspring):
            parent1_model = random.choice(population_with_objectives)[0]
            parent2_model = random.choice(population_with_objectives)[0]
            
            child_epigenetic_markers = random.choice([parent1_model.epigenetic_markers, parent2_model.epigenetic_markers]).clone().detach()
            if random.random() < EPIGENETIC_MUTATION_RATE:
                child_epigenetic_markers += Normal(0, EPIGENETIC_MUTATION_STRENGTH).sample(child_epigenetic_markers.shape)
                child_epigenetic_markers.clamp()

            child_modules_config = crossover_modular_nets(parent1_model.modules_config, parent2_model.modules_config, MIN_SUB_MODULES, MAX_SUB_MODULES, MIN_SUBNET_WIDTH, MAX_SUBNET_WIDTH, WIDTH_STEP, ACTIVATION_FNS)
            
            mutated_modules_config = await mutate_modular_net(child_modules_config, child_epigenetic_markers, MIN_SUB_MODULES, MAX_SUB_MODULES, MIN_SUBNET_WIDTH, MAX_SUBNET_WIDTH, WIDTH_STEP, ACTIVATION_FNS, BASE_MUTATION_RATE_STRUCTURE, EPIGENETIC_MUTATION_RATE)
            
            child = ModularMathReasoningNet(mutated_modules_config, child_epigenetic_markers)
            
            # 继承父代权重
            for i, child_module in enumerate(child.subnet_modules):
                parent_module_found = False
                if i < len(parent1_model.subnet_modules):
                    p1_m = parent1_model.subnet_modules[i]
                    p1_cfg = parent1_model.modules_config[i]
                    if p1_cfg == child.modules_config[i]:
                        child_module.load_state_dict(p1_m.state_dict())
                        parent_module_found = True
                if not parent_module_found and i < len(parent2_model.subnet_modules):
                    p2_m = parent2_model.subnet_modules[i]
                    p2_cfg = parent2_model.modules_config[i]
                    if p2_cfg == child.modules_config[i]:
                        child_module.load_state_dict(p2_m.state_dict())
                        parent_module_found = True
            
            random_parent_model = random.choice([parent1_model, parent2_model])
            try:
                child.final_output_layer.load_state_dict(random_parent_model.final_output_layer.state_dict())
            except RuntimeError:
                pass

            # 权重变异
            current_weight_mutation_strength = base_mutation_strength * (1 + child_epigenetic_markers[0].item()) 
            current_weight_mutation_strength = max(min_mutation_strength_base, current_weight_mutation_strength) 

            for param in child.parameters():
                param.data += torch.randn_like(param.data) * current_weight_mutation_strength
            
            offspring_population.append(child)

        logger.debug(f"子代生成完成，共 {len(offspring_population)} 个个体")

        # 合并种群并重新评估
        combined_population = population + offspring_population
        logger.debug("开始合并种群评估")
        
        combined_eval_tasks = [model.get_state() for model in combined_population]
        combined_results_objectives = []
        for state in combined_eval_tasks:
            result = await evaluator.evaluate_model_real_world(state, device, level)
            combined_results_objectives.append(result)

        R_t_with_objectives = []
        for i, model in enumerate(combined_population):
            if combined_results_objectives[i][0] > -float('inf'):
                R_t_with_objectives.append((model, combined_results_objectives[i]))
                
        if not R_t_with_objectives:
            logger.warning("合并种群中没有有效个体，重新创建种群")
            population = create_initial_population(len(population))
            continue

        # NSGA-II选择
        logger.debug("开始NSGA-II选择")
        fronts = non_dominated_sort(R_t_with_objectives)
        logger.debug(f"非支配排序完成，共 {len(fronts)} 个前沿")
        
        new_population = []
        current_pop_size = 0
        
        for front_idx, front in enumerate(fronts):
            front_with_objectives_for_cd = []
            for ind in front:
                for m, obj in R_t_with_objectives:
                    if m == ind:
                        front_with_objectives_for_cd.append((m, obj))
                        break

            crowding_distances = calculate_crowding_distance(front_with_objectives_for_cd)
            
            sorted_front = sorted(front, key=lambda ind: crowding_distances.get(ind, -float('inf')), reverse=True)
            
            if current_pop_size + len(sorted_front) <= len(population):
                new_population.extend(sorted_front)
                current_pop_size += len(sorted_front)
                logger.debug(f"前沿 {front_idx}: 添加 {len(sorted_front)} 个个体")
            else:
                remaining_slots = len(population) - current_pop_size
                new_population.extend(sorted_front[:remaining_slots])
                current_pop_size += remaining_slots
                logger.debug(f"前沿 {front_idx}: 添加 {remaining_slots} 个个体（部分）")
                break

        population = new_population
        
        logger.info(f"第 {gen+1} 世代完成 - 平均得分: {avg_score:.4f}, 最佳得分: {best_score:.4f}, 种群大小: {len(population)}") 

    return population, history_avg_score, history_best_score

def evolve_population_nsga2_simple(population: List[ModularMathReasoningNet], 
                           fitness_scores: List[Tuple[float, float]], 
                           mutation_rate: float = 0.8,  # 增加变异率
                           crossover_rate: float = 0.8,  # 增加交叉率
                           elite_size: int = 2,
                           diversity_boost: bool = True,
                           adaptive_mutation: bool = True) -> List[ModularMathReasoningNet]:
    """NSGA-II进化算法 - 增强多样性版本"""
    logger = get_logger(__name__)
    logger.info(f"开始NSGA-II进化 - 种群大小: {len(population)}, 变异率: {mutation_rate}, 交叉率: {crossover_rate}")
    
    if len(population) < 2:
        return population
    
    # 计算当前多样性
    diversity = calculate_population_diversity(population)
    logger.info(f"当前种群多样性: {diversity:.3f}")
    
    # 自适应变异率
    if adaptive_mutation and diversity < 0.5:
        mutation_rate = min(0.95, mutation_rate + 0.1)
        logger.info(f"检测到低多样性，调整变异率为: {mutation_rate}")
    
    # 非支配排序
    sorted_indices = _fast_non_dominated_sort(fitness_scores)
    
    # 选择父代 - 使用排序后的索引
    selected_parents = sorted_indices[:len(population) // 2]
    
    # 生成新个体
    new_population = []
    
    # 保留精英个体
    elite_indices = selected_parents[:elite_size]
    for idx in elite_indices:
        new_population.append(population[idx])
    
    # 生成新个体
    while len(new_population) < len(population):
        # 选择父代
        parent1_idx = random.choice(selected_parents)
        parent2_idx = random.choice(selected_parents)
        
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]
        
        # 交叉
        if random.random() < crossover_rate:
            child = _crossover_modules(parent1, parent2)
        else:
            child = parent1  # 直接复制
            
        # 变异
        if random.random() < mutation_rate:
            child = _enhanced_mutate_individual(child, diversity)
        
        new_population.append(child)
    
    # 多样性增强
    if diversity_boost and diversity < 0.6:
        # 添加随机个体
        random_individual = create_random_individual()
        new_population[-1] = random_individual
        logger.info("检测到低多样性，添加随机个体")
    
    logger.info(f"进化完成 - 新种群大小: {len(new_population)}")
    return new_population

def _fast_non_dominated_sort(fitness_scores: List[Tuple[float, float]]) -> List[int]:
    """快速非支配排序"""
    n = len(fitness_scores)
    domination_count = [0] * n
    dominated_solutions = [[] for _ in range(n)]
    rank = [0] * n
    
    # 计算支配关系
    for i in range(n):
        for j in range(n):
            if i != j:
                if _dominates(fitness_scores[i], fitness_scores[j]):
                    dominated_solutions[i].append(j)
                elif _dominates(fitness_scores[j], fitness_scores[i]):
                    domination_count[i] += 1
    
    # 分配等级
    current_rank = 0
    while True:
        current_front = [i for i in range(n) if domination_count[i] == 0 and rank[i] == 0]
        if not current_front:
            break
        
        for i in current_front:
            rank[i] = current_rank
        
        for i in current_front:
            for j in dominated_solutions[i]:
                domination_count[j] -= 1
        
        current_rank += 1
    
    # 返回排序后的索引
    return sorted(range(n), key=lambda x: rank[x])

def _dominates(score1: Tuple[float, float], score2: Tuple[float, float]) -> bool:
    """判断score1是否支配score2"""
    return (score1[0] >= score2[0] and score1[1] >= score2[1] and 
            (score1[0] > score2[0] or score1[1] > score2[1]))

def _tournament_selection(fitness_scores: List[Tuple[float, float]], tournament_size: int = 3) -> int:
    """锦标赛选择"""
    tournament_indices = random.sample(range(len(fitness_scores)), tournament_size)
    best_idx = tournament_indices[0]
    best_score = sum(fitness_scores[best_idx])
    
    for idx in tournament_indices[1:]:
        score = sum(fitness_scores[idx])
        if score > best_score:
            best_score = score
            best_idx = idx
    
    return best_idx

def _crossover_modular_net(config1: List[Dict], config2: List[Dict]) -> List[Dict]:
    """模块网络配置交叉"""
    if not config1 or not config2:
        return config1 if config1 else config2
    
    # 选择较短的配置作为基础
    base_config = config1 if len(config1) <= len(config2) else config2
    other_config = config2 if len(config1) <= len(config2) else config1
    
    # 单点交叉
    crossover_point = len(base_config) // 2
    
    child_config = []
    for i in range(len(base_config)):
        if i < crossover_point:
            child_config.append(base_config[i].copy())
        else:
            # 从另一个配置中选择对应模块，如果存在
            if i < len(other_config):
                child_config.append(other_config[i].copy())
            else:
                child_config.append(base_config[i].copy())
    
    return child_config

def _mutate_modular_net(config: List[Dict]) -> List[Dict]:
    """模块网络配置变异"""
    if not config:
        return config
    
    mutated_config = []
    for module_cfg in config:
        if not isinstance(module_cfg, dict):
            continue
            
        mutated_module = module_cfg.copy()
        
        # 随机变异参数
        if random.random() < 0.3:
            # 变异宽度
            if 'widths' in mutated_module:
                mutated_module['widths'] = [
                    max(1, w + random.randint(-2, 2)) 
                    for w in mutated_module['widths']
                ]
        
        if random.random() < 0.2:
            # 变异激活函数
            activation_fns = ['relu', 'tanh', 'sigmoid', 'leaky_relu']
            mutated_module['activation'] = random.choice(activation_fns)
        
        if random.random() < 0.1:
            # 变异模块类型
            module_types = ['add_sub', 'trig', 'exp_log', 'prod', 'calculus', 'linear_alg', 'agi', 'generic']
            mutated_module['module_type'] = random.choice(module_types)
        
        mutated_config.append(mutated_module)
    
    return mutated_config

def _enhanced_mutate_modular_net(config: List[Dict], mutation_rate: float) -> List[Dict]:
    """增强变异 - 更多探索"""
    mutated_config = []
    
    for module_config in config:
        new_module = module_config.copy()
        
        # 增加变异类型
        mutation_type = random.random()
        
        if mutation_type < 0.3:
            # 结构变异
            if 'hidden_size' in new_module:
                new_module['hidden_size'] = max(1, new_module['hidden_size'] + random.randint(-2, 2))
            if 'num_layers' in new_module:
                new_module['num_layers'] = max(1, new_module['num_layers'] + random.randint(-1, 1))
                
        elif mutation_type < 0.6:
            # 激活函数变异
            activation_functions = ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'gelu']
            if 'activation' in new_module:
                new_module['activation'] = random.choice(activation_functions)
                
        elif mutation_type < 0.8:
            # 正则化变异
            if random.random() < 0.5:
                new_module['dropout'] = random.uniform(0.0, 0.5)
            if random.random() < 0.5:
                new_module['batch_norm'] = random.choice([True, False])
                
        else:
            # 激进变异 - 完全重新生成模块
            if random.random() < mutation_rate * 0.3:
                new_module = _generate_random_module_config()
        
        mutated_config.append(new_module)
    
    return mutated_config

def _generate_random_config() -> List[Dict]:
    """生成随机配置"""
    num_modules = random.randint(1, 4)
    config = []
    
    for _ in range(num_modules):
        config.append(_generate_random_module_config())
    
    return config

def _generate_random_module_config() -> Dict:
    """生成随机模块配置 - 修复版本"""
    module_types = ['add_sub', 'trig', 'exp_log', 'prod', 'calculus', 'linear_alg', 'agi', 'generic']
    module_type = random.choice(module_types)
    
    # 生成随机宽度列表
    num_layers = random.randint(1, 3)
    widths = [random.randint(8, 32) for _ in range(num_layers)]
    
    config = {
        'module_type': module_type,
        'hidden_size': random.randint(8, 64),
        'num_layers': num_layers,
        'activation': random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu']),
        'dropout': random.uniform(0.0, 0.3),
        'batch_norm': random.choice([True, False]),
        'output_dim': random.randint(1, 10),  # 添加必需的output_dim字段
        'input_dim': random.randint(1, 10),   # 添加必需的input_dim字段
        'widths': widths,                     # 添加必需的widths字段
        'activation_fn_name': random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu']),  # 添加必需的activation_fn_name字段
        'use_batchnorm': random.choice([True, False])  # 添加必需的use_batchnorm字段
    }
    
    return config

def _enhance_diversity(population: List[ModularMathReasoningNet]) -> List[ModularMathReasoningNet]:
    """增强种群多样性"""
    # 计算配置相似度
    config_hashes = [hash(str(model.modules_config)) for model in population]
    
    # 如果多样性太低，添加更多随机个体
    unique_configs = len(set(config_hashes))
    diversity_ratio = unique_configs / len(population)
    
    if diversity_ratio < 0.7:  # 多样性阈值
        logger.info(f"检测到低多样性 ({diversity_ratio:.2f})，添加随机个体")
        
        # 替换一些相似个体
        for i in range(len(population) // 4):  # 替换25%的个体
            if random.random() < 0.5:
                random_config = _generate_random_config()
                population[i] = ModularMathReasoningNet(random_config)
    
    return population

def calculate_population_diversity(population: List[ModularMathReasoningNet]) -> float:
    """计算种群多样性"""
    if len(population) < 2:
        return 0.0
    
    # 计算模块配置的差异
    configs = []
    for individual in population:
        config_str = str(individual.modules_config)
        configs.append(hash(config_str))
    
    # 计算配置的多样性
    unique_configs = len(set(configs))
    diversity = unique_configs / len(population)
    
    return diversity

def _enhanced_mutate_individual(individual: ModularMathReasoningNet, diversity: float) -> ModularMathReasoningNet:
    """增强的个体变异 - 根据多样性调整变异强度"""
    # 根据多样性调整变异强度
    mutation_strength = 1.0 if diversity < 0.5 else 0.5
    
    # 随机选择变异类型
    mutation_types = ['add_module', 'remove_module', 'modify_module', 'swap_modules', 'random_config']
    
    # 低多样性时增加激进变异
    if diversity < 0.3:
        mutation_types.extend(['completely_random', 'hybrid_mutation'])
    
    mutation_type = random.choice(mutation_types)
    
    if mutation_type == 'add_module':
        return _add_random_module(individual)
    elif mutation_type == 'remove_module':
        return _remove_random_module(individual)
    elif mutation_type == 'modify_module':
        return _modify_random_module(individual, mutation_strength)
    elif mutation_type == 'swap_modules':
        return _swap_modules(individual)
    elif mutation_type == 'random_config':
        return _randomize_config(individual)
    elif mutation_type == 'completely_random':
        return create_random_individual()
    elif mutation_type == 'hybrid_mutation':
        return _hybrid_mutation(individual)
    else:
        return individual

def _add_random_module(individual: ModularMathReasoningNet) -> ModularMathReasoningNet:
    """添加随机模块"""
    new_config = _generate_random_module_config()
    individual.modules_config.append(new_config)
    return individual

def _remove_random_module(individual: ModularMathReasoningNet) -> ModularMathReasoningNet:
    """移除随机模块"""
    if len(individual.modules_config) > 1:
        idx = random.randint(0, len(individual.modules_config) - 1)
        individual.modules_config.pop(idx)
    return individual

def _modify_random_module(individual: ModularMathReasoningNet, strength: float) -> ModularMathReasoningNet:
    """修改随机模块"""
    if not individual.modules_config:
        return individual
    
    idx = random.randint(0, len(individual.modules_config) - 1)
    module_config = individual.modules_config[idx].copy()
    
    # 根据强度修改不同参数
    if random.random() < strength:
        module_config['hidden_size'] = random.randint(8, 64)
    if random.random() < strength:
        module_config['activation'] = random.choice(['relu', 'tanh', 'sigmoid', 'leaky_relu'])
    if random.random() < strength:
        module_config['dropout'] = random.uniform(0.0, 0.5)
    
    individual.modules_config[idx] = module_config
    return individual

def _swap_modules(individual: ModularMathReasoningNet) -> ModularMathReasoningNet:
    """交换模块位置"""
    if len(individual.modules_config) < 2:
        return individual
    
    idx1, idx2 = random.sample(range(len(individual.modules_config)), 2)
    individual.modules_config[idx1], individual.modules_config[idx2] = \
        individual.modules_config[idx2], individual.modules_config[idx1]
    
    return individual

def _randomize_config(individual: ModularMathReasoningNet) -> ModularMathReasoningNet:
    """随机化配置"""
    individual.modules_config = [_generate_random_module_config() for _ in range(len(individual.modules_config))]
    return individual

def _hybrid_mutation(individual: ModularMathReasoningNet) -> ModularMathReasoningNet:
    """混合变异 - 多种变异组合"""
    individual = _add_random_module(individual)
    individual = _modify_random_module(individual, 1.0)
    individual = _swap_modules(individual)
    return individual

def create_random_individual() -> ModularMathReasoningNet:
    """创建完全随机的个体"""
    num_modules = random.randint(1, 5)
    modules_config = [_generate_random_module_config() for _ in range(num_modules)]
    
    individual = ModularMathReasoningNet(modules_config)
    return individual

def _crossover_modules(parent1: ModularMathReasoningNet, parent2: ModularMathReasoningNet) -> ModularMathReasoningNet:
    """交叉两个模块网络"""
    # 获取父代配置
    config1 = parent1.modules_config
    config2 = parent2.modules_config
    
    if not config1 and not config2:
        # 创建默认配置
        default_config = [_generate_random_module_config()]
        child = ModularMathReasoningNet(default_config)
        return child
    
    if not config1:
        child = ModularMathReasoningNet(config2.copy())
        return child
    
    if not config2:
        child = ModularMathReasoningNet(config1.copy())
        return child
    
    # 随机选择交叉点
    min_len = min(len(config1), len(config2))
    if min_len == 0:
        default_config = [_generate_random_module_config()]
        child = ModularMathReasoningNet(default_config)
        return child
    
    crossover_point = random.randint(1, min_len)
    
    # 执行交叉
    child_config = []
    child_config.extend(config1[:crossover_point])
    child_config.extend(config2[crossover_point:])
    
    # 确保至少有一个模块
    if not child_config:
        child_config = [config1[0] if config1 else config2[0]]
    
    child = ModularMathReasoningNet(child_config)
    return child