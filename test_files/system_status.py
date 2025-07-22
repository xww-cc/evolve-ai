#!/usr/bin/env python3
"""
系统状态检查脚本 - 提供全面的系统状态报告
"""

import asyncio
import time
import torch
import psutil
import os
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evolution.nsga2 import evolve_population_nsga2
from config.logging_setup import setup_logging
from config.global_constants import POPULATION_SIZE, NUM_GENERATIONS, LEVEL_DESCRIPTIONS

logger = setup_logging('system_status.log')

class SystemStatusChecker:
    """系统状态检查器"""
    
    def __init__(self):
        self.status_report = {}
        
    def check_system_resources(self):
        """检查系统资源"""
        print("💻 检查系统资源...")
        
        # CPU信息
        cpu_count = psutil.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存信息
        memory = psutil.virtual_memory()
        
        # 磁盘信息
        disk = psutil.disk_usage('/')
        
        # GPU信息
        gpu_available = torch.cuda.is_available()
        gpu_info = {}
        if gpu_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_info = {
                'count': gpu_count,
                'name': gpu_name,
                'memory_gb': gpu_memory
            }
        
        self.status_report['system_resources'] = {
            'cpu_count': cpu_count,
            'cpu_percent': cpu_percent,
            'memory_total_gb': memory.total / 1e9,
            'memory_available_gb': memory.available / 1e9,
            'memory_percent': memory.percent,
            'disk_total_gb': disk.total / 1e9,
            'disk_free_gb': disk.free / 1e9,
            'disk_percent': (disk.used / disk.total) * 100,
            'gpu_available': gpu_available,
            'gpu_info': gpu_info
        }
        
        print(f"✅ CPU: {cpu_count}核心, 使用率: {cpu_percent:.1f}%")
        print(f"✅ 内存: {memory.total/1e9:.1f}GB, 使用率: {memory.percent:.1f}%")
        print(f"✅ 磁盘: {disk.total/1e9:.1f}GB, 使用率: {(disk.used/disk.total)*100:.1f}%")
        if gpu_available:
            print(f"✅ GPU: {gpu_info['name']}, 内存: {gpu_info['memory_gb']:.1f}GB")
        else:
            print("⚠️  GPU: 不可用")
            
    def check_python_environment(self):
        """检查Python环境"""
        print("🐍 检查Python环境...")
        
        import sys
        import numpy as np
        
        python_version = sys.version
        numpy_version = np.__version__
        torch_version = torch.__version__
        
        self.status_report['python_environment'] = {
            'python_version': python_version,
            'numpy_version': numpy_version,
            'torch_version': torch_version
        }
        
        print(f"✅ Python版本: {python_version.split()[0]}")
        print(f"✅ NumPy版本: {numpy_version}")
        print(f"✅ PyTorch版本: {torch_version}")
        
    async def check_core_components(self):
        """检查核心组件"""
        print("🔧 检查核心组件...")
        
        components_status = {}
        
        # 检查种群创建
        try:
            start_time = time.time()
            population = create_initial_population(5)
            creation_time = time.time() - start_time
            components_status['population_creation'] = {
                'status': 'OK',
                'time': creation_time,
                'size': len(population)
            }
            print(f"✅ 种群创建: 正常 (耗时: {creation_time:.3f}s)")
        except Exception as e:
            components_status['population_creation'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"❌ 种群创建: 错误 - {e}")
        
        # 检查评估器
        try:
            realworld_evaluator = RealWorldEvaluator()
            symbolic_evaluator = SymbolicEvaluator()
            components_status['evaluators'] = {
                'status': 'OK',
                'realworld': 'OK',
                'symbolic': 'OK'
            }
            print("✅ 评估器: 正常")
        except Exception as e:
            components_status['evaluators'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"❌ 评估器: 错误 - {e}")
        
        # 检查进化算法
        try:
            start_time = time.time()
            fitness_scores = [(0.8, 0.6), (0.9, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.8)]
            evolved_population = evolve_population_nsga2(population, fitness_scores)
            evolution_time = time.time() - start_time
            components_status['evolution_algorithm'] = {
                'status': 'OK',
                'time': evolution_time,
                'output_size': len(evolved_population)
            }
            print(f"✅ 进化算法: 正常 (耗时: {evolution_time:.3f}s)")
        except Exception as e:
            components_status['evolution_algorithm'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            print(f"❌ 进化算法: 错误 - {e}")
        
        self.status_report['core_components'] = components_status
        
    async def check_performance_benchmarks(self):
        """检查性能基准"""
        print("📊 检查性能基准...")
        
        benchmarks = {}
        
        # 评估性能基准
        try:
            population = create_initial_population(10)
            evaluator = SymbolicEvaluator()
            
            start_time = time.time()
            for individual in population:
                await evaluator.evaluate(individual)
            eval_time = time.time() - start_time
            
            benchmarks['evaluation_performance'] = {
                'time_per_individual': eval_time / len(population),
                'total_time': eval_time,
                'individuals_per_second': len(population) / eval_time
            }
            
            print(f"✅ 评估性能: {len(population)/eval_time:.1f} 个体/秒")
        except Exception as e:
            benchmarks['evaluation_performance'] = {
                'error': str(e)
            }
            print(f"❌ 评估性能: 错误 - {e}")
        
        # 进化性能基准
        try:
            fitness_scores = [(0.8, 0.6)] * len(population)
            start_time = time.time()
            evolved_population = evolve_population_nsga2(population, fitness_scores)
            evolution_time = time.time() - start_time
            
            benchmarks['evolution_performance'] = {
                'time': evolution_time,
                'generations_per_second': 1 / evolution_time if evolution_time > 0 else 0
            }
            
            print(f"✅ 进化性能: {1/evolution_time:.1f} 代/秒")
        except Exception as e:
            benchmarks['evolution_performance'] = {
                'error': str(e)
            }
            print(f"❌ 进化性能: 错误 - {e}")
        
        self.status_report['performance_benchmarks'] = benchmarks
        
    def generate_status_report(self):
        """生成状态报告"""
        print("\n📋 系统状态报告:")
        print("=" * 50)
        
        # 系统资源状态
        resources = self.status_report.get('system_resources', {})
        print("💻 系统资源:")
        print(f"   CPU: {resources.get('cpu_count', 'N/A')}核心, {resources.get('cpu_percent', 'N/A')}%使用")
        print(f"   内存: {resources.get('memory_total_gb', 'N/A'):.1f}GB, {resources.get('memory_percent', 'N/A')}%使用")
        print(f"   磁盘: {resources.get('disk_total_gb', 'N/A'):.1f}GB, {resources.get('disk_percent', 'N/A')}%使用")
        if resources.get('gpu_available'):
            gpu_info = resources.get('gpu_info', {})
            print(f"   GPU: {gpu_info.get('name', 'N/A')}, {gpu_info.get('memory_gb', 'N/A'):.1f}GB")
        
        # 环境状态
        env = self.status_report.get('python_environment', {})
        print("\n🐍 Python环境:")
        print(f"   Python: {env.get('python_version', 'N/A').split()[0]}")
        print(f"   NumPy: {env.get('numpy_version', 'N/A')}")
        print(f"   PyTorch: {env.get('torch_version', 'N/A')}")
        
        # 组件状态
        components = self.status_report.get('core_components', {})
        print("\n🔧 核心组件:")
        for component, status in components.items():
            if status.get('status') == 'OK':
                print(f"   ✅ {component}: 正常")
            else:
                print(f"   ❌ {component}: 错误 - {status.get('error', 'Unknown')}")
        
        # 性能基准
        benchmarks = self.status_report.get('performance_benchmarks', {})
        print("\n📊 性能基准:")
        eval_perf = benchmarks.get('evaluation_performance', {})
        if 'individuals_per_second' in eval_perf:
            print(f"   评估速度: {eval_perf['individuals_per_second']:.1f} 个体/秒")
        else:
            print(f"   评估速度: 错误 - {eval_perf.get('error', 'Unknown')}")
            
        evo_perf = benchmarks.get('evolution_performance', {})
        if 'generations_per_second' in evo_perf:
            print(f"   进化速度: {evo_perf['generations_per_second']:.1f} 代/秒")
        else:
            print(f"   进化速度: 错误 - {evo_perf.get('error', 'Unknown')}")
        
        # 系统评级
        print("\n🎯 系统评级:")
        error_count = sum(1 for comp in components.values() if comp.get('status') == 'ERROR')
        if error_count == 0:
            print("   🟢 系统状态: 优秀")
        elif error_count <= 1:
            print("   🟡 系统状态: 良好")
        elif error_count <= 2:
            print("   🟠 系统状态: 一般")
        else:
            print("   🔴 系统状态: 需要修复")
            
        print("=" * 50)

async def main():
    """主函数"""
    checker = SystemStatusChecker()
    
    # 运行所有检查
    checker.check_system_resources()
    checker.check_python_environment()
    await checker.check_core_components()
    await checker.check_performance_benchmarks()
    
    # 生成报告
    checker.generate_status_report()

if __name__ == "__main__":
    asyncio.run(main()) 