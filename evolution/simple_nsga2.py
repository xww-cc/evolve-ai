#!/usr/bin/env python3
"""
简化版本的NSGA-II进化算法
避免复杂的异步操作和阻塞
"""

import random
import logging
from typing import List, Tuple
import torch
import numpy as np
from models.modular_net import ModularMathReasoningNet
from models.epigenetic import EpigeneticMarkers
from config.global_constants import *
from config.logging_setup import setup_logging

logger = setup_logging()

def simple_evolution_step(population: List[ModularMathReasoningNet], generation: int) -> Tuple[List[float], List[float]]:
    """简化的进化步骤"""
    logger.info(f"=== 第 {generation+1} 世代开始 ===")
    
    # 简单的评估（不使用复杂的异步评估器）
    scores = []
    for i, model in enumerate(population):
        try:
            # 测试模型前向传播
            test_input = torch.randn(1, 4)
            with torch.no_grad():
                model.eval()  # 设置为评估模式
                output = model(test_input)
            # 简单的评分：基于输出的均值和方差
            score = torch.mean(output).item() + torch.var(output).item() * 0.1
            scores.append(score)
            logger.debug(f"个体 {i+1} 得分: {score:.4f}")
        except Exception as e:
            logger.error(f"个体 {i+1} 评估失败: {e}")
            scores.append(0.0)
    
    # 计算统计信息
    if scores:
        avg_score = np.mean(scores)
        best_score = max(scores)
        logger.info(f"第 {generation+1} 世代 - 平均得分: {avg_score:.4f}, 最佳得分: {best_score:.4f}")
    else:
        avg_score = 0.0
        best_score = 0.0
        logger.warning("没有有效得分")
    
    return [avg_score], [best_score]

def simple_crossover(parent1: ModularMathReasoningNet, parent2: ModularMathReasoningNet) -> ModularMathReasoningNet:
    """简化的交叉操作"""
    # 随机选择父代的配置
    if random.random() < 0.5:
        modules_config = parent1.modules_config
    else:
        modules_config = parent2.modules_config
    
    # 创建新的表观遗传标记
    epigenetic_markers = EpigeneticMarkers()
    
    # 创建子代
    child = ModularMathReasoningNet(modules_config=modules_config, epigenetic_markers=epigenetic_markers)
    
    return child

def simple_mutation(individual: ModularMathReasoningNet, mutation_rate: float = 0.1) -> ModularMathReasoningNet:
    """简化的变异操作"""
    if random.random() < mutation_rate:
        # 简单的权重变异
        for param in individual.parameters():
            if random.random() < 0.1:  # 10%的参数变异
                param.data += torch.randn_like(param.data) * 0.01
    
    return individual

def simple_selection(population: List[ModularMathReasoningNet], scores: List[float], elite_size: int = 2) -> List[ModularMathReasoningNet]:
    """简化的选择操作"""
    # 按得分排序
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    # 保留精英个体
    elite = [population[i] for i in sorted_indices[:elite_size]]
    
    # 生成新个体
    new_population = elite.copy()
    
    while len(new_population) < len(population):
        # 随机选择父代
        parent1 = random.choice(population)
        parent2 = random.choice(population)
        
        # 交叉
        child = simple_crossover(parent1, parent2)
        
        # 变异
        child = simple_mutation(child)
        
        new_population.append(child)
    
    return new_population

def evolve_population_simple(population: List[ModularMathReasoningNet], num_generations: int = 3) -> Tuple[List[ModularMathReasoningNet], List[float], List[float]]:
    """简化的进化算法"""
    logger.info(f"开始简化进化 - 世代数: {num_generations}, 种群大小: {len(population)}")
    
    history_avg_score = []
    history_best_score = []
    
    for gen in range(num_generations):
        # 评估当前种群
        avg_scores, best_scores = simple_evolution_step(population, gen)
        
        history_avg_score.extend(avg_scores)
        history_best_score.extend(best_scores)
        
        # 计算当前得分
        current_scores = []
        for i, model in enumerate(population):
            try:
                test_input = torch.randn(1, 4)
                with torch.no_grad():
                    model.eval()
                    output = model(test_input)
                score = torch.mean(output).item() + torch.var(output).item() * 0.1
                current_scores.append(score)
            except Exception as e:
                current_scores.append(0.0)
        
        # 选择和新个体生成
        population = simple_selection(population, current_scores)
        
        logger.info(f"第 {gen+1} 世代完成")
    
    return population, history_avg_score, history_best_score 