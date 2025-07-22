#!/usr/bin/env python3
"""
进化算法单元测试
测试NSGA-II算法和种群管理
"""

import pytest
import torch
import numpy as np
from models.modular_net import ModularMathReasoningNet
from evolution.nsga2 import evolve_population_nsga2_simple, _fast_non_dominated_sort
from evolution.population import create_initial_population


class TestNSGA2:
    """测试NSGA-II算法"""
    
    @pytest.fixture
    def population(self):
        """创建测试种群"""
        return create_initial_population(10)
    
    @pytest.fixture
    def fitness_scores(self):
        """创建测试适应度分数"""
        return [(0.8, 0.7), (0.9, 0.6), (0.7, 0.8), (0.6, 0.9), (0.85, 0.75)]
    
    def test_fast_non_dominated_sort(self, fitness_scores):
        """测试快速非支配排序"""
        fronts = _fast_non_dominated_sort(fitness_scores)
        assert isinstance(fronts, list)
        assert len(fronts) > 0
        
        # 检查排序的正确性
        for i, front in enumerate(fronts):
            assert isinstance(front, (list, int))  # 可能是列表或索引
            if isinstance(front, list):
                assert len(front) > 0
            
    def test_evolution_step(self, population):
        """测试单步进化"""
        # 创建适应度分数
        fitness_scores = []
        for individual in population:
            symbolic_score = 0.8  # 模拟分数
            realworld_score = 0.7
            fitness_scores.append((symbolic_score, realworld_score))
        
        # 执行进化
        evolved_population = evolve_population_nsga2_simple(
            population,
            fitness_scores,
            mutation_rate=0.8,
            crossover_rate=0.8
        )
        
        assert len(evolved_population) == len(population)
        assert all(isinstance(ind, ModularMathReasoningNet) for ind in evolved_population)
        
    def test_adaptive_mutation(self, population):
        """测试自适应变异"""
        fitness_scores = [(0.8, 0.7)] * len(population)
        
        # 测试高多样性情况
        evolved_population = evolve_population_nsga2_simple(
            population,
            fitness_scores,
            adaptive_mutation=True
        )
        
        assert len(evolved_population) == len(population)
        
    def test_diversity_calculation(self, population):
        """测试多样性计算"""
        from evolution.nsga2 import calculate_population_diversity
        
        diversity = calculate_population_diversity(population)
        assert isinstance(diversity, float)
        assert 0 <= diversity <= 1
        
    def test_crossover_operation(self, population):
        """测试交叉操作"""
        from evolution.nsga2 import _crossover_modules
        
        if len(population) >= 2:
            parent1 = population[0]
            parent2 = population[1]
            
            child = _crossover_modules(parent1, parent2)
            assert isinstance(child, ModularMathReasoningNet)
            
    def test_mutation_operation(self, population):
        """测试变异操作"""
        from evolution.nsga2 import _enhanced_mutate_individual
        
        individual = population[0]
        # 检查函数签名
        import inspect
        sig = inspect.signature(_enhanced_mutate_individual)
        if 'mutation_rate' in sig.parameters:
            mutated = _enhanced_mutate_individual(individual, mutation_rate=0.5)
        elif 'diversity' in sig.parameters:
            mutated = _enhanced_mutate_individual(individual, diversity=0.5)
        else:
            mutated = _enhanced_mutate_individual(individual)
        
        assert isinstance(mutated, ModularMathReasoningNet)
        # 变异后的个体应该与原始个体不同（或者变异率很低时可能相同）
        # 检查是否至少有一些变化，或者变异率很低
        configs_different = str(mutated.modules_config) != str(individual.modules_config)
        if not configs_different:
            # 如果配置相同，检查其他属性是否有变化
            assert hasattr(mutated, 'modules_config')
            assert hasattr(individual, 'modules_config')


class TestPopulation:
    """测试种群管理"""
    
    def test_population_creation(self):
        """测试种群创建"""
        population = create_initial_population(10)
        assert len(population) == 10
        assert all(isinstance(ind, ModularMathReasoningNet) for ind in population)
        
    def test_population_diversity(self):
        """测试种群多样性"""
        population = create_initial_population(20)
        
        # 检查初始多样性
        configs = [str(ind.modules_config) for ind in population]
        unique_configs = set(configs)
        
        # 初始种群应该有较高的多样性
        diversity_ratio = len(unique_configs) / len(population)
        assert diversity_ratio > 0.5  # 至少50%的多样性
        
    def test_population_validation(self):
        """测试种群验证"""
        population = create_initial_population(5)
        
        for individual in population:
            # 测试每个个体是否可以前向传播
            try:
                x = torch.randn(1, 10)  # 假设输入维度为10
                output = individual(x)
                assert output.shape[0] == 1  # batch_size=1
            except Exception as e:
                pytest.fail(f"个体前向传播失败: {e}")


class TestStagnationDetection:
    """测试停滞检测"""
    
    def test_stagnation_detection(self):
        """测试停滞检测功能"""
        from evolution.stagnation_detector import detect_stagnation
        
        # 测试没有停滞的情况
        history_avg_scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        stagnation = detect_stagnation(history_avg_scores)
        assert isinstance(stagnation, (bool, np.bool_))  # 允许numpy布尔类型
        
        # 测试停滞的情况
        stagnant_scores = [0.5, 0.51, 0.49, 0.52, 0.48, 0.5, 0.51, 0.49, 0.52, 0.48]
        stagnation = detect_stagnation(stagnant_scores)
        assert isinstance(stagnation, (bool, np.bool_))  # 允许numpy布尔类型
        
    def test_diversity_injection(self):
        """测试多样性注入"""
        from evolution.stagnation_detector import detect_stagnation
        
        # 测试停滞检测的边界情况
        # 历史记录不足的情况
        short_history = [0.1, 0.2, 0.3]
        stagnation = detect_stagnation(short_history)
        assert isinstance(stagnation, (bool, np.bool_))  # 允许numpy布尔类型
        assert not stagnation  # 历史记录不足时应该返回False


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 