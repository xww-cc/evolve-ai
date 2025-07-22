#!/usr/bin/env python3
"""
性能基准测试
测试系统的吞吐量、资源消耗和扩展性
"""

import pytest
import asyncio
import time
import numpy as np
import psutil
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evolution.nsga2 import evolve_population_nsga2_simple
from config.logging_setup import setup_logging

logger = setup_logging()

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.metrics = []
    
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.metrics = []
    
    def record_metric(self, name, value, unit=""):
        """记录指标"""
        self.metrics.append({
            'name': name,
            'value': value,
            'unit': unit,
            'timestamp': time.time()
        })
    
    def get_memory_usage(self):
        """获取内存使用"""
        return psutil.virtual_memory().percent
    
    def get_cpu_usage(self):
        """获取CPU使用"""
        return psutil.cpu_percent()
    
    def get_elapsed_time(self):
        """获取经过时间"""
        if self.start_time:
            return time.time() - self.start_time
        return 0
    
    def print_summary(self):
        """打印摘要"""
        logger.info("性能监控摘要:")
        for metric in self.metrics:
            logger.info(f"  {metric['name']}: {metric['value']:.3f} {metric['unit']}")

class TestPerformanceBenchmarks:
    """性能基准测试"""
    
    @pytest.fixture
    def monitor(self):
        return PerformanceMonitor()
    
    @pytest.fixture
    def evaluators(self):
        return {
            'symbolic': SymbolicEvaluator(),
            'realworld': RealWorldEvaluator()
        }
    
    @pytest.mark.asyncio
    async def test_population_creation_performance(self, monitor):
        """测试种群创建性能"""
        logger.info("测试种群创建性能")
        monitor.start_monitoring()
        
        # 测试不同大小的种群创建
        population_sizes = [10, 20, 50, 100]
        creation_times = []
        
        for size in population_sizes:
            start_time = time.time()
            population = create_initial_population(size)
            creation_time = time.time() - start_time
            
            creation_times.append(creation_time)
            monitor.record_metric(f"种群{size}创建时间", creation_time, "秒")
            
            logger.info(f"种群大小 {size}: {creation_time:.3f}秒")
        
        # 计算性能指标
        avg_creation_time = np.mean(creation_times)
        individuals_per_second = sum(population_sizes) / sum(creation_times)
        
        monitor.record_metric("平均创建时间", avg_creation_time, "秒")
        monitor.record_metric("创建速度", individuals_per_second, "个体/秒")
        
        logger.info(f"平均创建时间: {avg_creation_time:.3f}秒")
        logger.info(f"创建速度: {individuals_per_second:.2f} 个体/秒")
        
        monitor.print_summary()
        
        # 性能断言
        assert avg_creation_time < 1.0  # 平均创建时间不超过1秒
        assert individuals_per_second > 10  # 至少10个体/秒
    
    @pytest.mark.asyncio
    async def test_evaluation_throughput(self, monitor, evaluators):
        """测试评估吞吐量"""
        logger.info("测试评估吞吐量")
        monitor.start_monitoring()
        
        # 创建测试种群
        population = create_initial_population(20)
        
        # 评估所有个体
        evaluation_times = []
        fitness_scores = []
        
        for i, individual in enumerate(population):
            start_time = time.time()
            
            # 双重评估
            symbolic_score = await evaluators['symbolic'].evaluate(individual, level=0)
            realworld_score = await evaluators['realworld'].evaluate(individual)
            
            evaluation_time = time.time() - start_time
            evaluation_times.append(evaluation_time)
            fitness_scores.append((symbolic_score, realworld_score))
            
            logger.info(f"个体{i+1}: 符号={symbolic_score:.3f}, 真实世界={realworld_score:.3f}, 耗时={evaluation_time:.3f}s, CPU={psutil.cpu_percent():.1f}%, 内存={psutil.virtual_memory().percent:.1f}%")
        
        # 计算性能指标
        avg_evaluation_time = np.mean(evaluation_times)
        evaluations_per_second = len(population) / sum(evaluation_times)
        
        monitor.record_metric("平均评估时间", avg_evaluation_time, "秒")
        monitor.record_metric("评估速度", evaluations_per_second, "个体/秒")
        monitor.record_metric("总评估时间", sum(evaluation_times), "秒")
        
        logger.info(f"平均评估时间: {avg_evaluation_time:.3f}秒")
        logger.info(f"评估速度: {evaluations_per_second:.2f} 个体/秒")
        
        monitor.print_summary()
        
        # 性能断言
        assert avg_evaluation_time < 5.0  # 平均评估时间不超过5秒
        assert evaluations_per_second > 0.1  # 至少0.1个体/秒
    
    @pytest.mark.asyncio
    async def test_evolution_performance(self, monitor, evaluators):
        """测试进化性能"""
        logger.info("测试进化性能")
        monitor.start_monitoring()
        
        # 创建测试种群
        population = create_initial_population(15)
        generations = 3
        evolution_times = []
        
        for generation in range(generations):
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
            evolution_times.append(gen_time)
            
            logger.info(f"第 {generation+1} 代: {gen_time:.3f}秒")
        
        # 计算性能指标
        avg_evolution_time = np.mean(evolution_times)
        generations_per_second = generations / sum(evolution_times)
        
        monitor.record_metric("平均进化时间", avg_evolution_time, "秒")
        monitor.record_metric("进化速度", generations_per_second, "代/秒")
        monitor.record_metric("总进化时间", sum(evolution_times), "秒")
        
        logger.info(f"平均进化时间: {avg_evolution_time:.3f}秒")
        logger.info(f"进化速度: {generations_per_second:.2f} 代/秒")
        
        monitor.print_summary()
        
        # 性能断言
        assert avg_evolution_time < 30.0  # 每代不超过30秒
        assert generations_per_second > 0.1  # 至少0.1代/秒
    
    @pytest.mark.asyncio
    async def test_resource_consumption(self, monitor, evaluators):
        """测试资源消耗"""
        logger.info("测试资源消耗")
        monitor.start_monitoring()
        
        # 运行密集测试
        population = create_initial_population(30)
        
        # 记录初始资源使用
        initial_memory = psutil.virtual_memory().used
        initial_cpu = psutil.cpu_percent()
        
        # 运行多轮评估和进化
        for round_num in range(3):
            round_start = time.time()
            
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
            
            round_time = time.time() - round_start
            
            # 记录资源使用
            current_memory = psutil.virtual_memory().used
            current_cpu = psutil.cpu_percent()
            
            memory_increase = (current_memory - initial_memory) / 1024 / 1024  # MB
            cpu_usage = current_cpu
            
            monitor.record_metric(f"第{round_num+1}轮内存增加", memory_increase, "MB")
            monitor.record_metric(f"第{round_num+1}轮CPU使用", cpu_usage, "%")
            monitor.record_metric(f"第{round_num+1}轮时间", round_time, "秒")
            
            logger.info(f"第{round_num+1}轮: 内存+{memory_increase:.1f}MB, CPU:{cpu_usage:.1f}%, 时间:{round_time:.2f}秒")
        
        monitor.print_summary()
        
        # 资源使用断言
        final_memory_increase = monitor.metrics[-3]['value']  # 最后一轮的内存增加
        assert final_memory_increase < 1000  # 内存增加不超过1GB
        assert monitor.metrics[-2]['value'] < 90  # CPU使用不超过90%
    
    @pytest.mark.asyncio
    async def test_scalability_test(self, monitor, evaluators):
        """测试扩展性"""
        logger.info("测试扩展性")
        monitor.start_monitoring()
        
        # 测试不同种群大小的性能
        population_sizes = [10, 20, 30]
        scalability_results = {}
        
        for size in population_sizes:
            logger.info(f"测试种群大小: {size}")
            size_start = time.time()
            
            population = create_initial_population(size)
            
            # 运行3代进化
            for gen in range(3):
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
            
            size_time = time.time() - size_start
            scalability_results[size] = size_time
            
            monitor.record_metric(f"种群{size}总时间", size_time, "秒")
            monitor.record_metric(f"种群{size}效率", size / size_time, "个体/秒")
            
            logger.info(f"种群大小 {size}: {size_time:.3f}秒, 效率: {size/size_time:.2f} 个体/秒")
        
        monitor.print_summary()
        
        # 扩展性断言
        assert scalability_results[10] < scalability_results[20] < scalability_results[30]  # 时间应该随种群大小增加
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, monitor, evaluators):
        """测试内存泄漏检测"""
        logger.info("测试内存泄漏检测")
        monitor.start_monitoring()
        
        # 创建初始种群
        population = create_initial_population(20)
        initial_memory = psutil.virtual_memory().used
        
        # 运行多轮操作
        memory_usage = []
        
        for round_num in range(5):
            round_start = time.time()
            
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
            
            # 记录内存使用
            current_memory = psutil.virtual_memory().used
            memory_increase = (current_memory - initial_memory) / 1024 / 1024  # MB
            memory_usage.append(memory_increase)
            
            round_time = time.time() - round_start
            
            monitor.record_metric(f"第{round_num+1}轮内存增加", memory_increase, "MB")
            monitor.record_metric(f"第{round_num+1}轮时间", round_time, "秒")
            
            logger.info(f"第{round_num+1}轮: 内存+{memory_increase:.1f}MB, 时间:{round_time:.2f}秒")
        
        # 检查内存泄漏
        memory_growth_rate = (memory_usage[-1] - memory_usage[0]) / len(memory_usage)
        
        monitor.record_metric("内存增长率", memory_growth_rate, "MB/轮")
        monitor.print_summary()
        
        # 内存泄漏断言
        assert memory_growth_rate < 10  # 每轮内存增长不超过10MB
        assert memory_usage[-1] < 500  # 总内存增长不超过500MB


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 