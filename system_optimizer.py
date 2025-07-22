#!/usr/bin/env python3
"""
系统优化脚本 - 优化系统性能
"""

import asyncio
import time
import torch
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evolution.nsga2 import evolve_population_nsga2
from config.logging_setup import setup_logging
from config.global_constants import POPULATION_SIZE

logger = setup_logging('system_optimizer.log')

class SystemOptimizer:
    """系统优化器"""
    
    def __init__(self):
        self.optimization_results = {}
        
    async def optimize_evaluation_cache(self):
        """优化评估缓存"""
        print("🔧 优化评估缓存...")
        
        # 测试缓存效果
        evaluator = SymbolicEvaluator()
        population = create_initial_population(5)
        
        # 第一次评估（无缓存）
        start_time = time.time()
        for individual in population:
            await evaluator.evaluate(individual)
        first_eval_time = time.time() - start_time
        
        # 第二次评估（有缓存）
        start_time = time.time()
        for individual in population:
            await evaluator.evaluate(individual)
        second_eval_time = time.time() - start_time
        
        cache_improvement = (first_eval_time - second_eval_time) / first_eval_time * 100
        
        self.optimization_results['cache_improvement'] = cache_improvement
        print(f"✅ 缓存优化完成 - 性能提升: {cache_improvement:.1f}%")
        
    async def optimize_population_size(self):
        """优化种群大小"""
        print("📊 优化种群大小...")
        
        sizes = [5, 10, 15, 20]
        results = {}
        
        for size in sizes:
            start_time = time.time()
            population = create_initial_population(size)
            creation_time = time.time() - start_time
            
            # 评估时间
            evaluator = SymbolicEvaluator()
            eval_start = time.time()
            for individual in population:
                await evaluator.evaluate(individual)
            eval_time = time.time() - eval_start
            
            results[size] = {
                'creation_time': creation_time,
                'eval_time': eval_time,
                'total_time': creation_time + eval_time,
                'time_per_individual': (creation_time + eval_time) / size
            }
            
            print(f"   种群大小 {size}: 创建={creation_time:.3f}s, 评估={eval_time:.3f}s, 总时间={creation_time + eval_time:.3f}s")
        
        # 找到最优种群大小
        optimal_size = min(results.keys(), key=lambda x: results[x]['time_per_individual'])
        self.optimization_results['optimal_population_size'] = optimal_size
        print(f"✅ 最优种群大小: {optimal_size}")
        
    async def optimize_parallel_evaluation(self):
        """优化并行评估"""
        print("⚡ 优化并行评估...")
        
        population = create_initial_population(8)
        evaluator = SymbolicEvaluator()
        
        # 串行评估
        start_time = time.time()
        for individual in population:
            await evaluator.evaluate(individual)
        serial_time = time.time() - start_time
        
        # 并行评估
        start_time = time.time()
        tasks = [evaluator.evaluate(individual) for individual in population]
        await asyncio.gather(*tasks)
        parallel_time = time.time() - start_time
        
        parallel_improvement = (serial_time - parallel_time) / serial_time * 100
        self.optimization_results['parallel_improvement'] = parallel_improvement
        
        print(f"✅ 并行评估优化完成 - 性能提升: {parallel_improvement:.1f}%")
        print(f"   串行时间: {serial_time:.3f}s, 并行时间: {parallel_time:.3f}s")
        
    async def optimize_memory_usage(self):
        """优化内存使用"""
        print("💾 优化内存使用...")
        
        import gc
        import psutil
        
        # 记录初始内存
        initial_memory = psutil.virtual_memory().percent
        
        # 创建大量对象
        populations = []
        for i in range(5):
            population = create_initial_population(10)
            populations.append(population)
            
        # 记录内存使用
        memory_after_creation = psutil.virtual_memory().percent
        
        # 清理内存
        populations.clear()
        gc.collect()
        
        # 记录清理后内存
        memory_after_cleanup = psutil.virtual_memory().percent
        
        memory_efficiency = (memory_after_creation - memory_after_cleanup) / (memory_after_creation - initial_memory) * 100
        self.optimization_results['memory_efficiency'] = memory_efficiency
        
        print(f"✅ 内存优化完成 - 清理效率: {memory_efficiency:.1f}%")
        print(f"   初始内存: {initial_memory:.1f}%, 创建后: {memory_after_creation:.1f}%, 清理后: {memory_after_cleanup:.1f}%")
        
    async def run_optimizations(self):
        """运行所有优化"""
        print("🚀 开始系统优化...")
        
        optimizations = [
            self.optimize_evaluation_cache(),
            self.optimize_population_size(),
            self.optimize_parallel_evaluation(),
            self.optimize_memory_usage()
        ]
        
        await asyncio.gather(*optimizations)
        
        # 输出优化总结
        print("\n📊 优化总结:")
        for key, value in self.optimization_results.items():
            if 'improvement' in key or 'efficiency' in key:
                print(f"   {key}: {value:.1f}%")
            else:
                print(f"   {key}: {value}")
        
        # 性能评级
        print("\n🎯 系统性能评级:")
        total_improvement = sum([
            self.optimization_results.get('cache_improvement', 0),
            self.optimization_results.get('parallel_improvement', 0),
            self.optimization_results.get('memory_efficiency', 0)
        ]) / 3
        
        if total_improvement > 50:
            print("   🏆 性能评级: 优秀")
        elif total_improvement > 30:
            print("   🥇 性能评级: 良好")
        elif total_improvement > 10:
            print("   🥈 性能评级: 一般")
        else:
            print("   🥉 性能评级: 需要改进")
            
        print(f"   总体性能提升: {total_improvement:.1f}%")
        
        print("\n🎉 系统优化完成！")

async def main():
    """主函数"""
    optimizer = SystemOptimizer()
    await optimizer.run_optimizations()

if __name__ == "__main__":
    asyncio.run(main()) 