#!/usr/bin/env python3
"""
性能监控脚本 - 监控和优化系统性能
"""

import asyncio
import time
import psutil
import torch
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evolution.nsga2 import evolve_population_nsga2
from config.logging_setup import setup_logging

logger = setup_logging('performance_monitor.log')

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.cpu_usage = []
        self.evaluation_times = []
        self.evolution_times = []
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        logger.info("🚀 开始性能监控...")
        
    def record_metrics(self):
        """记录性能指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append(cpu_percent)
        
        # 内存使用率
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.percent)
        
        return cpu_percent, memory.percent
    
    def record_evaluation_time(self, duration):
        """记录评估时间"""
        self.evaluation_times.append(duration)
        
    def record_evolution_time(self, duration):
        """记录进化时间"""
        self.evolution_times.append(duration)
        
    def get_performance_summary(self):
        """获取性能总结"""
        if not self.start_time:
            return "未开始监控"
            
        total_time = time.time() - self.start_time
        
        summary = {
            "总运行时间": f"{total_time:.2f}秒",
            "平均CPU使用率": f"{sum(self.cpu_usage)/len(self.cpu_usage):.1f}%" if self.cpu_usage else "N/A",
            "平均内存使用率": f"{sum(self.memory_usage)/len(self.memory_usage):.1f}%" if self.memory_usage else "N/A",
            "平均评估时间": f"{sum(self.evaluation_times)/len(self.evaluation_times):.3f}秒" if self.evaluation_times else "N/A",
            "平均进化时间": f"{sum(self.evolution_times)/len(self.evolution_times):.3f}秒" if self.evolution_times else "N/A",
            "GPU可用": torch.cuda.is_available(),
            "GPU内存": f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
        }
        
        return summary

async def performance_test():
    """性能测试"""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    
    try:
        # 1. 测试种群创建性能
        print("📊 测试种群创建性能...")
        start_time = time.time()
        population = create_initial_population(10)
        creation_time = time.time() - start_time
        print(f"✅ 种群创建完成 - 耗时: {creation_time:.3f}秒")
        
        # 2. 测试评估器性能
        print("🔧 测试评估器性能...")
        realworld_evaluator = RealWorldEvaluator()
        symbolic_evaluator = SymbolicEvaluator()
        
        # 3. 测试评估性能
        print("📈 测试评估性能...")
        evaluation_start = time.time()
        fitness_scores = []
        
        for i, individual in enumerate(population):
            individual_start = time.time()
            symbolic_score = await symbolic_evaluator.evaluate(individual)
            realworld_score = await realworld_evaluator.evaluate(individual)
            individual_time = time.time() - individual_start
            
            fitness_scores.append((symbolic_score, realworld_score))
            monitor.record_evaluation_time(individual_time)
            
            cpu, memory = monitor.record_metrics()
            print(f"   个体{i+1}: 符号={symbolic_score:.3f}, 真实世界={realworld_score:.3f}, 耗时={individual_time:.3f}s, CPU={cpu:.1f}%, 内存={memory:.1f}%")
        
        evaluation_time = time.time() - evaluation_start
        monitor.record_evaluation_time(evaluation_time)
        print(f"✅ 评估完成 - 总耗时: {evaluation_time:.3f}秒")
        
        # 4. 测试进化性能
        print("🔄 测试进化性能...")
        evolution_start = time.time()
        evolved_population = evolve_population_nsga2(population, fitness_scores)
        evolution_time = time.time() - evolution_start
        monitor.record_evolution_time(evolution_time)
        print(f"✅ 进化完成 - 耗时: {evolution_time:.3f}秒")
        
        # 5. 性能总结
        print("\n📊 性能监控总结:")
        summary = monitor.get_performance_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # 6. 性能评估
        print("\n🎯 性能评估:")
        if evaluation_time < 1.0:
            print("   ⚡ 评估性能: 优秀")
        elif evaluation_time < 3.0:
            print("   🟡 评估性能: 良好")
        else:
            print("   🔴 评估性能: 需要优化")
            
        if evolution_time < 0.1:
            print("   ⚡ 进化性能: 优秀")
        elif evolution_time < 0.5:
            print("   🟡 进化性能: 良好")
        else:
            print("   🔴 进化性能: 需要优化")
            
        avg_cpu = sum(monitor.cpu_usage) / len(monitor.cpu_usage) if monitor.cpu_usage else 0
        if avg_cpu < 50:
            print("   🟢 资源使用: 正常")
        elif avg_cpu < 80:
            print("   🟡 资源使用: 较高")
        else:
            print("   🔴 资源使用: 过高")
        
        print("\n🎉 性能测试完成！")
        
    except Exception as e:
        print(f"❌ 性能测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(performance_test()) 