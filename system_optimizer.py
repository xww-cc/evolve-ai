#!/usr/bin/env python3
"""
系统优化脚本
用于系统参数调优和资源优化
"""

import asyncio
import time
import psutil
import gc
import torch
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from config.logging_setup import setup_logging

logger = setup_logging()

@dataclass
class OptimizationConfig:
    """优化配置"""
    # 进化参数
    population_size: int = 20
    mutation_rate: float = 0.8
    crossover_rate: float = 0.8
    num_generations: int = 10
    
    # 评估参数
    evaluation_batch_size: int = 5
    max_evaluation_time: float = 30.0
    
    # 资源参数
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 85.0
    max_gpu_memory_percent: float = 90.0
    
    # 性能参数
    target_evaluation_speed: float = 10.0  # 评估/秒
    target_evolution_speed: float = 2.0    # 进化/秒

class SystemOptimizer:
    """系统优化器"""
    
    def __init__(self):
        self.logger = logger
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_config = OptimizationConfig()
        
    async def optimize_system_parameters(self) -> Dict[str, Any]:
        """优化系统参数"""
        self.logger.info("🔧 开始系统参数优化...")
        
        # 检查当前系统状态
        system_status = await self._check_system_status()
        self.logger.info(f"当前系统状态: {system_status}")
        
        # 优化进化参数
        evolution_config = await self._optimize_evolution_parameters()
        
        # 优化评估参数
        evaluation_config = await self._optimize_evaluation_parameters()
        
        # 优化资源参数
        resource_config = await self._optimize_resource_parameters()
        
        # 合并优化结果
        optimized_config = OptimizationConfig(
            population_size=evolution_config.get('population_size', 20),
            mutation_rate=evolution_config.get('mutation_rate', 0.8),
            crossover_rate=evolution_config.get('crossover_rate', 0.8),
            num_generations=evolution_config.get('num_generations', 10),
            evaluation_batch_size=evaluation_config.get('batch_size', 5),
            max_evaluation_time=evaluation_config.get('max_time', 30.0),
            max_cpu_percent=resource_config.get('max_cpu', 80.0),
            max_memory_percent=resource_config.get('max_memory', 85.0),
            max_gpu_memory_percent=resource_config.get('max_gpu', 90.0)
        )
        
        self.current_config = optimized_config
        
        # 记录优化历史
        optimization_record = {
            "timestamp": time.time(),
            "original_config": system_status,
            "optimized_config": optimized_config.__dict__,
            "improvements": {
                "evolution": evolution_config.get('improvement', 0),
                "evaluation": evaluation_config.get('improvement', 0),
                "resource": resource_config.get('improvement', 0)
            }
        }
        self.optimization_history.append(optimization_record)
        
        self.logger.info("✅ 系统参数优化完成")
        return optimization_record
        
    async def _check_system_status(self) -> Dict[str, Any]:
        """检查系统状态"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # 检查GPU状态
        gpu_memory_percent = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_percent = (gpu_memory / gpu_memory_total) * 100
            
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "gpu_memory_percent": gpu_memory_percent,
            "available_memory_mb": memory.available / (1024 * 1024)
        }
        
    async def _optimize_evolution_parameters(self) -> Dict[str, Any]:
        """优化进化参数"""
        self.logger.info("优化进化参数...")
        
        # 测试不同的种群大小
        population_sizes = [10, 15, 20, 25, 30]
        best_population_size = 20
        best_performance = 0
        
        for size in population_sizes:
            # 模拟性能测试
            performance = await self._simulate_evolution_performance(size)
            if performance > best_performance:
                best_performance = performance
                best_population_size = size
                
        # 测试不同的变异率
        mutation_rates = [0.6, 0.7, 0.8, 0.9]
        best_mutation_rate = 0.8
        best_mutation_performance = 0
        
        for rate in mutation_rates:
            performance = await self._simulate_mutation_performance(rate)
            if performance > best_mutation_performance:
                best_mutation_performance = performance
                best_mutation_rate = rate
                
        improvement = ((best_performance + best_mutation_performance) / 2) - 0.5
        
        return {
            "population_size": best_population_size,
            "mutation_rate": best_mutation_rate,
            "crossover_rate": 0.8,  # 保持默认值
            "num_generations": 10,   # 保持默认值
            "improvement": improvement
        }
        
    async def _optimize_evaluation_parameters(self) -> Dict[str, Any]:
        """优化评估参数"""
        self.logger.info("优化评估参数...")
        
        # 测试不同的批处理大小
        batch_sizes = [3, 5, 8, 10]
        best_batch_size = 5
        best_evaluation_performance = 0
        
        for batch_size in batch_sizes:
            performance = await self._simulate_evaluation_performance(batch_size)
            if performance > best_evaluation_performance:
                best_evaluation_performance = performance
                best_batch_size = batch_size
                
        # 测试不同的最大评估时间
        max_times = [20.0, 30.0, 45.0, 60.0]
        best_max_time = 30.0
        best_time_performance = 0
        
        for max_time in max_times:
            performance = await self._simulate_time_performance(max_time)
            if performance > best_time_performance:
                best_time_performance = performance
                best_max_time = max_time
                
        improvement = ((best_evaluation_performance + best_time_performance) / 2) - 0.5
        
        return {
            "batch_size": best_batch_size,
            "max_time": best_max_time,
            "improvement": improvement
        }
        
    async def _optimize_resource_parameters(self) -> Dict[str, Any]:
        """优化资源参数"""
        self.logger.info("优化资源参数...")
        
        system_status = await self._check_system_status()
        
        # 根据当前系统状态调整资源限制
        current_cpu = system_status["cpu_percent"]
        current_memory = system_status["memory_percent"]
        current_gpu = system_status["gpu_memory_percent"]
        
        # 动态调整资源限制
        if current_cpu > 70:
            max_cpu = min(85.0, current_cpu + 10)
        else:
            max_cpu = 80.0
            
        if current_memory > 70:
            max_memory = min(90.0, current_memory + 10)
        else:
            max_memory = 85.0
            
        if current_gpu > 70:
            max_gpu = min(95.0, current_gpu + 10)
        else:
            max_gpu = 90.0
            
        improvement = 0.1  # 资源优化通常带来小幅改进
        
        return {
            "max_cpu": max_cpu,
            "max_memory": max_memory,
            "max_gpu": max_gpu,
            "improvement": improvement
        }
        
    async def _simulate_evolution_performance(self, population_size: int) -> float:
        """模拟进化性能"""
        # 简单的性能模拟
        base_performance = 0.5
        size_factor = min(1.0, population_size / 20.0)
        return base_performance + (size_factor * 0.3)
        
    async def _simulate_mutation_performance(self, mutation_rate: float) -> float:
        """模拟变异性能"""
        # 简单的性能模拟
        base_performance = 0.5
        rate_factor = abs(mutation_rate - 0.8) / 0.2  # 0.8是最佳值
        return base_performance + ((1 - rate_factor) * 0.3)
        
    async def _simulate_evaluation_performance(self, batch_size: int) -> float:
        """模拟评估性能"""
        # 简单的性能模拟
        base_performance = 0.5
        batch_factor = min(1.0, batch_size / 8.0)
        return base_performance + (batch_factor * 0.3)
        
    async def _simulate_time_performance(self, max_time: float) -> float:
        """模拟时间性能"""
        # 简单的性能模拟
        base_performance = 0.5
        time_factor = min(1.0, max_time / 45.0)
        return base_performance + (time_factor * 0.2)
        
    def optimize_memory_usage(self) -> Dict[str, Any]:
        """优化内存使用"""
        self.logger.info("优化内存使用...")
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理PyTorch缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # 检查内存使用情况
        memory_before = psutil.virtual_memory().used
        gc.collect()
        memory_after = psutil.virtual_memory().used
        memory_freed = memory_before - memory_after
        
        self.logger.info(f"释放内存: {memory_freed / (1024 * 1024):.1f} MB")
        
        return {
            "memory_freed_mb": memory_freed / (1024 * 1024),
            "optimization_success": memory_freed > 0
        }
        
    def optimize_cpu_usage(self) -> Dict[str, Any]:
        """优化CPU使用"""
        self.logger.info("优化CPU使用...")
        
        # 检查当前CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 根据CPU使用率提供建议
        if cpu_percent > 80:
            recommendation = "建议减少并行任务数量"
        elif cpu_percent > 60:
            recommendation = "建议优化算法效率"
        else:
            recommendation = "CPU使用率正常"
            
        return {
            "current_cpu_percent": cpu_percent,
            "recommendation": recommendation,
            "optimization_needed": cpu_percent > 80
        }
        
    def get_optimization_summary(self) -> str:
        """获取优化摘要"""
        if not self.optimization_history:
            return "暂无优化历史"
            
        latest_optimization = self.optimization_history[-1]
        improvements = latest_optimization["improvements"]
        
        total_improvement = sum(improvements.values())
        avg_improvement = total_improvement / len(improvements)
        
        summary = f"""
🔧 系统优化摘要
================

📊 最新优化时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(latest_optimization['timestamp']))}

🎯 优化改进:
   进化参数: {improvements['evolution']:.1%}
   评估参数: {improvements['evaluation']:.1%}
   资源参数: {improvements['resource']:.1%}
   平均改进: {avg_improvement:.1%}

⚙️  当前配置:
   种群大小: {self.current_config.population_size}
   变异率: {self.current_config.mutation_rate}
   交叉率: {self.current_config.crossover_rate}
   评估批大小: {self.current_config.evaluation_batch_size}
   最大评估时间: {self.current_config.max_evaluation_time}秒
   CPU限制: {self.current_config.max_cpu_percent}%
   内存限制: {self.current_config.max_memory_percent}%

📈 优化历史: {len(self.optimization_history)} 次优化
"""
        
        return summary
        
    def apply_optimizations(self) -> Dict[str, Any]:
        """应用优化"""
        self.logger.info("应用系统优化...")
        
        # 内存优化
        memory_result = self.optimize_memory_usage()
        
        # CPU优化
        cpu_result = self.optimize_cpu_usage()
        
        # 返回优化结果
        return {
            "memory_optimization": memory_result,
            "cpu_optimization": cpu_result,
            "optimization_success": memory_result["optimization_success"] or not cpu_result["optimization_needed"]
        }

async def demo_system_optimization():
    """演示系统优化功能"""
    optimizer = SystemOptimizer()
    
    # 执行系统优化
    optimization_result = await optimizer.optimize_system_parameters()
    
    # 应用优化
    applied_result = optimizer.apply_optimizations()
    
    # 生成摘要
    summary = optimizer.get_optimization_summary()
    print(summary)
    
    print(f"\n优化结果:")
    print(f"内存释放: {applied_result['memory_optimization']['memory_freed_mb']:.1f} MB")
    print(f"CPU使用率: {applied_result['cpu_optimization']['current_cpu_percent']:.1f}%")
    print(f"优化建议: {applied_result['cpu_optimization']['recommendation']}")

if __name__ == "__main__":
    asyncio.run(demo_system_optimization()) 