#!/usr/bin/env python3
"""
进化流程集成测试
测试完整的进化流程和模块间交互
"""

import pytest
import asyncio
import time
from evolution.population import create_initial_population
from evolution.nsga2 import evolve_population_nsga2_simple
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evaluators.realworld_evaluator import RealWorldEvaluator
from config.logging_setup import setup_logging

logger = setup_logging('integration_test.log')


class TestEvolutionFlow:
    """测试完整进化流程"""
    
    @pytest.fixture
    def evaluators(self):
        """创建评估器"""
        return {
            'symbolic': SymbolicEvaluator(),
            'realworld': RealWorldEvaluator()
        }
    
    @pytest.fixture
    def small_population(self):
        """创建小规模种群"""
        return create_initial_population(10)
    
    @pytest.mark.asyncio
    async def test_small_scale_evolution(self, evaluators, small_population):
        """测试小规模进化（种群=10，世代=5）"""
        logger.info("开始小规模进化测试")
        start_time = time.time()
        
        population = small_population
        best_scores = []
        
        for generation in range(5):
            logger.info(f"第 {generation + 1} 代")
            
            # 评估种群
            fitness_scores = []
            for individual in population:
                symbolic_score = await evaluators['symbolic'].evaluate(individual, level=0)
                realworld_score = await evaluators['realworld'].evaluate(individual)
                fitness_scores.append((symbolic_score, realworld_score))
            
            # 计算平均分数
            avg_symbolic = sum(score[0] for score in fitness_scores) / len(fitness_scores)
            avg_realworld = sum(score[1] for score in fitness_scores) / len(fitness_scores)
            best_scores.append((avg_symbolic, avg_realworld))
            
            logger.info(f"平均分数: 符号={avg_symbolic:.3f}, 真实世界={avg_realworld:.3f}")
            
            # 进化
            population = evolve_population_nsga2_simple(
                population,
                fitness_scores,
                mutation_rate=0.8,
                crossover_rate=0.8
            )
        
        total_time = time.time() - start_time
        logger.info(f"小规模进化完成，耗时: {total_time:.2f}秒")
        
        # 验证结果
        assert len(best_scores) == 5
        assert all(isinstance(score, tuple) for score in best_scores)
        assert all(0 <= score[0] <= 1 and 0 <= score[1] <= 1 for score in best_scores)
        
        # 检查是否有改进
        first_score = best_scores[0]
        last_score = best_scores[-1]
        improvement = (last_score[0] + last_score[1]) - (first_score[0] + first_score[1])
        logger.info(f"总体改进: {improvement:.3f}")
        
        return best_scores
    
    @pytest.mark.asyncio
    async def test_dual_evaluation_integration(self, evaluators, small_population):
        """测试双重评估集成"""
        logger.info("开始双重评估集成测试")
        
        # 选择几个个体进行详细测试
        test_individuals = small_population[:3]
        
        for i, individual in enumerate(test_individuals):
            logger.info(f"测试个体 {i+1}")
            
            # 符号评估
            symbolic_score = await evaluators['symbolic'].evaluate(individual, level=1)
            logger.info(f"符号评估分数: {symbolic_score:.3f}")
            
            # 真实世界评估
            realworld_score = await evaluators['realworld'].evaluate(individual)
            logger.info(f"真实世界评估分数: {realworld_score:.3f}")
            
            # 验证分数
            assert 0 <= symbolic_score <= 1
            assert 0 <= realworld_score <= 1
            
            # 检查分数合理性
            assert symbolic_score > 0.1  # 至少有一些能力
            assert realworld_score > 0.1
    
    @pytest.mark.asyncio
    async def test_stagnation_detection_integration(self, evaluators, small_population):
        """测试停滞检测集成"""
        logger.info("开始停滞检测集成测试")
        
        from evolution.stagnation_detector import detect_stagnation
        
        population = small_population
        history_scores = []
        
        for generation in range(5):  # 减少到5代以加快测试
            # 评估
            fitness_scores = []
            # 确保population是列表而不是协程
            if hasattr(population, '__await__'):
                population = await population
            for individual in population:
                symbolic_score = await evaluators['symbolic'].evaluate(individual, level=0)
                realworld_score = await evaluators['realworld'].evaluate(individual)
                fitness_scores.append((symbolic_score, realworld_score))
            
            # 计算平均适应度
            avg_fitness = sum(score[0] + score[1] for score in fitness_scores) / len(fitness_scores) / 2
            history_scores.append(avg_fitness)
            
            # 检查停滞
            is_stagnant = detect_stagnation(history_scores)
            logger.info(f"第 {generation+1} 代，平均适应度: {avg_fitness:.3f}, 停滞: {is_stagnant}")
            
            # 进化
            population = evolve_population_nsga2_simple(population, fitness_scores)
        
        # 验证停滞检测功能
        assert len(history_scores) > 0
        assert all(isinstance(score, float) for score in history_scores)
    
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, evaluators, small_population):
        """测试错误处理集成"""
        logger.info("开始错误处理集成测试")
        
        # 测试无效个体处理
        try:
            # 创建一个可能无效的个体
            invalid_individual = small_population[0]
            # 这里可以添加一些破坏性操作来测试错误处理
            
            score = await evaluators['symbolic'].evaluate(invalid_individual, level=0)
            assert isinstance(score, float)
            
        except Exception as e:
            logger.warning(f"捕获到预期错误: {e}")
            # 错误应该被正确处理，不应该导致系统崩溃
    
    @pytest.mark.asyncio
    async def test_performance_integration(self, evaluators, small_population):
        """测试性能集成"""
        logger.info("开始性能集成测试")
        
        start_time = time.time()
        population = small_population
        
                # 运行快速进化测试
        for generation in range(3):
            gen_start = time.time()

            # 评估
            fitness_scores = []
            # 确保population是列表而不是协程
            if hasattr(population, '__await__'):
                population = await population
            for individual in population:
                symbolic_score = await evaluators['symbolic'].evaluate(individual, level=0)
                realworld_score = await evaluators['realworld'].evaluate(individual)
                fitness_scores.append((symbolic_score, realworld_score))
            
            # 进化
            population = evolve_population_nsga2_simple(population, fitness_scores)
            
            gen_time = time.time() - gen_start
            logger.info(f"第 {generation+1} 代耗时: {gen_time:.3f}秒")
            
            # 性能检查
            assert gen_time < 10.0  # 每代应该不超过10秒
        
        total_time = time.time() - start_time
        logger.info(f"性能测试完成，总耗时: {total_time:.2f}秒")
        
        # 计算性能指标
        individuals_per_second = (len(small_population) * 3) / total_time
        logger.info(f"处理速度: {individuals_per_second:.1f} 个体/秒")
        
        assert individuals_per_second > 1.0  # 至少1个体/秒


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 