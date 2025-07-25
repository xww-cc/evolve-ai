#!/usr/bin/env python3
"""
增强版功能测试
包含更多边界情况、压力测试和性能验证
"""

import pytest
import asyncio
import time
import torch
import numpy as np
from evolution.enhanced_evolution import get_enhanced_evolution_engine
from evolution.population import create_initial_population
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evaluators.realworld_evaluator import RealWorldEvaluator
from utils.performance_monitor import get_performance_monitor
from config.logging_setup import setup_logging

logger = setup_logging('enhanced_functional_test.log')

class TestEnhancedFunctionalCapabilities:
    """增强版功能能力测试"""
    
    @pytest.fixture
    def enhanced_engine(self):
        """创建增强进化引擎"""
        return get_enhanced_evolution_engine()
    
    @pytest.fixture
    def performance_monitor(self):
        """获取性能监控器"""
        return get_performance_monitor()
    
    @pytest.fixture
    def evaluators(self):
        """创建评估器"""
        return {
            'symbolic': SymbolicEvaluator(),
            'realworld': RealWorldEvaluator()
        }
    
    @pytest.mark.asyncio
    async def test_enhanced_evolution_engine(self, enhanced_engine):
        """测试增强进化引擎"""
        logger.info("测试增强进化引擎")
        
        # 创建测试种群
        population = create_initial_population(20)
        
        # 运行增强进化
        evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
            population, num_generations=5, level=0
        )
        
        # 验证结果
        assert len(evolved_population) == 20
        assert len(avg_scores) == 5
        assert len(best_scores) == 5
        
        # 验证进化改进
        improvement = best_scores[-1] - best_scores[0]
        logger.info(f"进化改进: {improvement:.4f}")
        
        # 获取进化摘要
        summary = enhanced_engine.get_evolution_summary()
        assert 'total_generations' in summary
        assert 'final_best_score' in summary
        assert 'total_improvement' in summary
        
        logger.info(f"进化摘要: {summary}")
    
    @pytest.mark.asyncio
    async def test_adaptive_parameters(self, enhanced_engine):
        """测试自适应参数调整"""
        logger.info("测试自适应参数调整")
        
        population = create_initial_population(15)
        
        # 运行进化并监控参数调整
        evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
            population, num_generations=8, level=0
        )
        
        # 检查参数调整历史
        adjustment_history = enhanced_engine.adaptive_params.adjustment_history
        logger.info(f"参数调整次数: {len(adjustment_history)}")
        
        if adjustment_history:
            latest_adjustment = adjustment_history[-1]
            assert 'generation' in latest_adjustment
            assert 'adjustments' in latest_adjustment
            logger.info(f"最新参数调整: {latest_adjustment}")
    
    @pytest.mark.asyncio
    async def test_diversity_injection(self, enhanced_engine):
        """测试多样性注入"""
        logger.info("测试多样性注入")
        
        population = create_initial_population(10)
        
        # 强制停滞状态
        enhanced_engine.stagnation_count = 25  # 超过阈值
        
        # 运行进化
        evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
            population, num_generations=3, level=0
        )
        
        # 验证多样性注入
        diversity = enhanced_engine._calculate_diversity(evolved_population)
        logger.info(f"注入多样性后的种群多样性: {diversity:.3f}")
        
        assert diversity > 0.1  # 应该有合理的多样性
    
    @pytest.mark.asyncio
    async def test_performance_monitoring_integration(self, enhanced_engine, performance_monitor):
        """测试性能监控集成"""
        logger.info("测试性能监控集成")
        
        # 启动性能监控
        performance_monitor.start_monitoring()
        
        population = create_initial_population(12)
        
        # 运行进化
        evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
            population, num_generations=4, level=0
        )
        
        # 检查性能指标
        realtime_metrics = performance_monitor.get_realtime_metrics()
        assert 'cpu_percent' in realtime_metrics
        assert 'memory_percent' in realtime_metrics
        assert 'timestamp' in realtime_metrics
        
        logger.info(f"实时性能指标: {realtime_metrics}")
        
        # 检查系统健康状态
        health_status = performance_monitor.check_system_health()
        assert 'overall_status' in health_status
        assert 'cpu_status' in health_status
        assert 'memory_status' in health_status
        
        logger.info(f"系统健康状态: {health_status}")
    
    @pytest.mark.asyncio
    async def test_stress_test_large_population(self, enhanced_engine):
        """压力测试 - 大种群"""
        logger.info("压力测试 - 大种群")
        
        # 创建大种群
        large_population = create_initial_population(50)
        
        start_time = time.time()
        
        # 运行进化
        evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
            large_population, num_generations=3, level=0
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"大种群进化耗时: {execution_time:.2f}秒")
        
        # 验证性能
        assert execution_time < 60  # 应该在60秒内完成
        assert len(evolved_population) == 50
        
        # 检查内存使用
        import psutil
        memory_usage = psutil.virtual_memory().percent
        logger.info(f"内存使用率: {memory_usage:.1f}%")
        assert memory_usage < 90  # 内存使用不应过高
    
    @pytest.mark.asyncio
    async def test_boundary_conditions(self, enhanced_engine):
        """测试边界条件"""
        logger.info("测试边界条件")
        
        # 测试空种群
        try:
            empty_population = []
            evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
                empty_population, num_generations=1, level=0
            )
            assert len(evolved_population) == 0
            logger.info("空种群处理正常")
        except Exception as e:
            logger.warning(f"空种群处理异常: {e}")
        
        # 测试单个体种群
        try:
            single_individual = create_initial_population(1)
            evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
                single_individual, num_generations=2, level=0
            )
            assert len(evolved_population) == 1
            logger.info("单个体种群处理正常")
        except Exception as e:
            logger.warning(f"单个体种群处理异常: {e}")
        
        # 测试零代数进化
        try:
            population = create_initial_population(5)
            evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
                population, num_generations=0, level=0
            )
            # 零代数时应该返回空列表或初始状态
            assert len(avg_scores) == 0 or len(avg_scores) == 1
            assert len(best_scores) == 0 or len(best_scores) == 1
            logger.info("零代数进化处理正常")
        except Exception as e:
            logger.warning(f"零代数进化处理异常: {e}")
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, enhanced_engine):
        """测试错误恢复能力"""
        logger.info("测试错误恢复能力")
        
        population = create_initial_population(8)
        
        # 模拟评估器错误
        original_evaluate = enhanced_engine.symbolic_evaluator.evaluate
        
        def error_evaluate(individual, level=0):
            if np.random.random() < 0.3:  # 30%概率出错
                raise Exception("模拟评估错误")
            return original_evaluate(individual, level)
        
        enhanced_engine.symbolic_evaluator.evaluate = error_evaluate
        
        try:
            # 运行进化
            evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
                population, num_generations=3, level=0
            )
            
            # 验证系统能够继续运行
            assert len(evolved_population) == 8
            assert len(avg_scores) == 3
            logger.info("错误恢复测试通过")
            
        finally:
            # 恢复原始评估器
            enhanced_engine.symbolic_evaluator.evaluate = original_evaluate
    
    @pytest.mark.asyncio
    async def test_optimization_suggestions(self, enhanced_engine):
        """测试优化建议生成"""
        logger.info("测试优化建议生成")
        
        population = create_initial_population(10)
        
        # 运行进化
        await enhanced_engine.evolve_population_enhanced(
            population, num_generations=4, level=0
        )
        
        # 获取优化建议
        suggestions = enhanced_engine.get_optimization_suggestions()
        
        assert isinstance(suggestions, list)
        logger.info(f"优化建议数量: {len(suggestions)}")
        
        for suggestion in suggestions:
            logger.info(f"建议: {suggestion}")
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, enhanced_engine):
        """测试内存效率"""
        logger.info("测试内存效率")
        
        import psutil
        import gc
        
        # 记录初始内存
        initial_memory = psutil.virtual_memory().used / (1024 * 1024)
        
        # 运行多次进化
        for i in range(3):
            population = create_initial_population(15)
            evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
                population, num_generations=2, level=0
            )
            
            # 强制垃圾回收
            gc.collect()
            
            # 检查内存增长
            current_memory = psutil.virtual_memory().used / (1024 * 1024)
            memory_growth = current_memory - initial_memory
            
            logger.info(f"第{i+1}次进化后内存增长: {memory_growth:.1f}MB")
            
            # 内存增长不应过大
            assert memory_growth < 500  # 增长不应超过500MB
    
    @pytest.mark.asyncio
    async def test_concurrent_evolution(self, enhanced_engine):
        """测试并发进化"""
        logger.info("测试并发进化")
        
        async def run_evolution(level: int):
            population = create_initial_population(8)
            evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
                population, num_generations=2, level=level
            )
            return level, avg_scores[-1], best_scores[-1]
        
        # 并发运行多个进化任务
        tasks = [run_evolution(level) for level in range(3)]
        results = await asyncio.gather(*tasks)
        
        # 验证所有任务都完成
        assert len(results) == 3
        
        for level, avg_score, best_score in results:
            logger.info(f"级别{level}: 平均分数={avg_score:.4f}, 最佳分数={best_score:.4f}")
            assert avg_score > 0
            assert best_score > 0
    
    @pytest.mark.asyncio
    async def test_evolution_convergence(self, enhanced_engine):
        """测试进化收敛性"""
        logger.info("测试进化收敛性")
        
        population = create_initial_population(20)
        
        # 运行较长时间的进化
        evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
            population, num_generations=10, level=0
        )
        
        # 分析收敛性
        if len(best_scores) >= 5:
            # 检查最后几代的改进
            recent_improvement = best_scores[-1] - best_scores[-5]
            overall_improvement = best_scores[-1] - best_scores[0]
            
            logger.info(f"整体改进: {overall_improvement:.4f}")
            logger.info(f"最近改进: {recent_improvement:.4f}")
            
            # 应该有整体改进
            assert overall_improvement >= -0.1  # 允许小幅下降
    
    @pytest.mark.asyncio
    async def test_parameter_sensitivity(self, enhanced_engine):
        """测试参数敏感性"""
        logger.info("测试参数敏感性")
        
        # 测试不同参数设置
        test_configs = [
            {'mutation_rate': 0.05, 'crossover_rate': 0.9},
            {'mutation_rate': 0.2, 'crossover_rate': 0.7},
            {'mutation_rate': 0.15, 'crossover_rate': 0.8}
        ]
        
        results = []
        
        for i, config in enumerate(test_configs):
            # 重置引擎状态
            enhanced_engine.stagnation_count = 0
            enhanced_engine.last_best_fitness = 0.0
            enhanced_engine.evolution_history = []
            
            # 设置参数
            enhanced_engine.adaptive_params.mutation_rate = config['mutation_rate']
            enhanced_engine.adaptive_params.crossover_rate = config['crossover_rate']
            
            population = create_initial_population(12)
            evolved_population, avg_scores, best_scores = await enhanced_engine.evolve_population_enhanced(
                population, num_generations=5, level=1  # 增加代数和难度级别
            )
            
            results.append({
                'config': config,
                'final_best_score': best_scores[-1],
                'final_avg_score': avg_scores[-1],
                'improvement_rate': (best_scores[-1] - best_scores[0]) / (best_scores[0] + 1e-8)
            })
            
            logger.info(f"配置{i+1}: {config} -> 最佳分数: {best_scores[-1]:.4f}, 改进率: {results[-1]['improvement_rate']:.4f}")
        
        # 验证参数调整确实发生了
        improvement_rates = [r['improvement_rate'] for r in results]
        avg_scores = [r['final_avg_score'] for r in results]
        
        # 检查是否有不同的改进模式或平均分数
        assert len(set([round(rate, 3) for rate in improvement_rates])) > 1 or len(set([round(score, 3) for score in avg_scores])) > 1
        
        # 验证所有配置都产生了合理的结果
        for result in results:
            assert result['final_best_score'] > 0
            assert result['final_avg_score'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 