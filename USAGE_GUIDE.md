# AI自主进化系统 - 使用指南

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install -r requirements.txt

# 检查系统状态
python system_status.py
```

### 2. 基础使用

#### 2.1 运行快速验证
```bash
# 快速验证AI自主进化能力
PYTHONPATH=. python test_files/quick_evolution_validation.py
```

#### 2.2 运行完整测试
```bash
# 运行所有测试用例
python -m pytest tests/ -v

# 运行性能测试
python performance_monitor.py

# 运行系统测试
python system_test.py
```

### 3. 核心功能使用

#### 3.1 创建初始种群
```python
from evolution.population import create_initial_population

# 创建10个AI模型的初始种群
population = create_initial_population(10)
print(f"创建了 {len(population)} 个AI模型")
```

#### 3.2 执行进化过程
```python
from evolution.nsga2 import evolve_population_nsga2_simple
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evaluators.realworld_evaluator import RealWorldEvaluator

async def run_evolution():
    # 创建评估器
    symbolic_eval = SymbolicEvaluator()
    realworld_eval = RealWorldEvaluator()
    
    # 创建初始种群
    population = create_initial_population(6)
    
    # 执行进化
    for generation in range(2):
        print(f"=== 第 {generation + 1} 世代 ===")
        
        # 评估种群
        fitness_scores = []
        for individual in population:
            symbolic_score = await symbolic_eval.evaluate(individual)
            realworld_score = await realworld_eval.evaluate(individual)
            fitness_scores.append((symbolic_score, realworld_score))
        
        # 进化
        population = evolve_population_nsga2_simple(
            population, fitness_scores, mutation_rate=0.8, crossover_rate=0.8
        )
        
        # 显示结果
        avg_score = sum(sum(scores) for scores in fitness_scores) / len(fitness_scores)
        print(f"平均得分: {avg_score:.4f}")
    
    return population

# 运行进化
import asyncio
evolved_population = asyncio.run(run_evolution())
```

## 🔧 高级功能

### 1. 自定义评估器

```python
class CustomEvaluator:
    """自定义评估器"""
    
    async def evaluate(self, model):
        """自定义评估逻辑"""
        try:
            # 测试输入
            x = torch.randn(5, 4)
            output = model(x)
            
            # 自定义评分逻辑
            score = 0.0
            
            # 基于输出稳定性评分
            output_std = torch.std(output).item()
            if output_std > 0.01:
                score += 0.5
            
            # 基于模型复杂度评分
            total_params = sum(p.numel() for p in model.parameters())
            if total_params > 100:
                score += 0.3
            
            # 基于模块数量评分
            if len(model.subnet_modules) > 1:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            return 0.0
```

### 2. 自定义进化参数

```python
# 自定义进化配置
evolution_config = {
    'population_size': 20,      # 种群大小
    'generations': 50,          # 进化代数
    'mutation_rate': 0.1,       # 变异率
    'crossover_rate': 0.9,      # 交叉率
    'elite_size': 2,            # 精英个体数量
    'tournament_size': 3        # 锦标赛大小
}

async def custom_evolution():
    population = create_initial_population(evolution_config['population_size'])
    
    for gen in range(evolution_config['generations']):
        # 评估
        fitness_scores = await evaluate_population(population)
        
        # 进化
        population = evolve_population_nsga2_simple(
            population, 
            fitness_scores,
            mutation_rate=evolution_config['mutation_rate'],
            crossover_rate=evolution_config['crossover_rate']
        )
        
        # 记录进度
        if gen % 10 == 0:
            avg_fitness = sum(sum(scores) for scores in fitness_scores) / len(fitness_scores)
            print(f"世代 {gen}: 平均适应度 = {avg_fitness:.4f}")
    
    return population
```

### 3. 性能监控

```python
from performance_monitor import PerformanceMonitor

# 创建性能监控器
monitor = PerformanceMonitor()

# 监控进化过程
async def monitored_evolution():
    population = create_initial_population(10)
    
    for generation in range(10):
        # 开始监控
        monitor.start_monitoring()
        
        # 执行进化
        fitness_scores = await evaluate_population(population)
        population = evolve_population_nsga2_simple(population, fitness_scores)
        
        # 停止监控并记录
        monitor.stop_monitoring()
        monitor.record_generation(generation, fitness_scores)
        
        # 显示性能指标
        metrics = monitor.get_current_metrics()
        print(f"世代 {generation}: CPU={metrics['cpu_percent']:.1f}%, "
              f"内存={metrics['memory_percent']:.1f}%, "
              f"耗时={metrics['duration']:.2f}秒")
    
    # 生成性能报告
    monitor.generate_report()
```

### 4. 系统状态检查

```python
from system_status import SystemStatusChecker

async def check_system():
    checker = SystemStatusChecker()
    
    # 检查系统资源
    resources = await checker.check_system_resources()
    print(f"CPU使用率: {resources['data'].cpu_percent:.1f}%")
    print(f"内存使用率: {resources['data'].memory_percent:.1f}%")
    
    # 检查Python环境
    environment = await checker.check_python_environment()
    print(f"Python版本: {environment['python_version']['version']}")
    print(f"CUDA可用: {environment['gpu']['cuda_available']}")
    print(f"MPS可用: {environment['gpu']['mps_available']}")
    
    # 检查核心组件
    components = await checker.check_core_components()
    print(f"组件状态: {components['summary']['available']}/{components['summary']['total']} 正常")
    
    # 生成完整报告
    report = await checker.generate_status_report()
    print(report)
```

## 📊 测试和验证

### 1. 运行所有测试

```bash
# 运行单元测试
python -m pytest tests/unit/ -v

# 运行集成测试
python -m pytest tests/integration/ -v

# 运行性能测试
python -m pytest tests/performance/ -v

# 运行功能测试
python -m pytest tests/functional/ -v

# 运行所有测试
python -m pytest tests/ -v
```

### 2. 性能基准测试

```bash
# 运行性能基准测试
python -m pytest tests/performance/test_performance_benchmarks.py -v

# 运行内存泄漏检测
python -m pytest tests/performance/test_performance_benchmarks.py::TestPerformanceBenchmarks::test_memory_leak_detection -v
```

### 3. 系统功能验证

```bash
# 验证AI自主进化能力
PYTHONPATH=. python test_files/quick_evolution_validation.py

# 检查系统状态
python system_status.py

# 运行系统测试
python system_test.py
```

## 🛠️ 故障排除

### 1. 常见问题

#### 问题1: 模块导入错误
```bash
# 解决方案：设置PYTHONPATH
export PYTHONPATH=.
python your_script.py
```

#### 问题2: CUDA不可用
```python
# 检查GPU状态
import torch
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"MPS可用: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")

# 如果CUDA不可用，系统会自动使用MPS或CPU
```

#### 问题3: 内存不足
```python
# 优化内存使用
import gc
import torch

# 清理内存
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 减少种群大小
population = create_initial_population(5)  # 使用更小的种群
```

### 2. 调试模式

```python
# 启用调试模式
import os
os.environ['EVOLVE_AI_DEBUG'] = 'true'

# 设置详细日志
from config.logging_setup import set_log_level
set_log_level('DEBUG')
```

### 3. 性能优化

```python
# 使用GPU加速
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

# 启用并行处理
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# 优化缓存
from evaluators.symbolic_evaluator import SymbolicEvaluator
evaluator = SymbolicEvaluator()
evaluator.cache_ttl = 600  # 增加缓存时间到10分钟
```

## 📈 监控和分析

### 1. 进化过程监控

```python
class EvolutionMonitor:
    """进化过程监控器"""
    
    def __init__(self):
        self.generation_history = []
        self.fitness_history = []
    
    def record_generation(self, generation, fitness_scores):
        """记录世代信息"""
        avg_fitness = sum(sum(scores) for scores in fitness_scores) / len(fitness_scores)
        best_fitness = max(max(scores) for scores in fitness_scores)
        
        self.generation_history.append(generation)
        self.fitness_history.append({
            'avg': avg_fitness,
            'best': best_fitness
        })
    
    def plot_evolution_curve(self):
        """绘制进化曲线"""
        import matplotlib.pyplot as plt
        
        generations = self.generation_history
        avg_fitness = [f['avg'] for f in self.fitness_history]
        best_fitness = [f['best'] for f in self.fitness_history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_fitness, label='平均适应度', marker='o')
        plt.plot(generations, best_fitness, label='最佳适应度', marker='s')
        plt.xlabel('世代')
        plt.ylabel('适应度')
        plt.title('AI进化过程')
        plt.legend()
        plt.grid(True)
        plt.savefig('evolution_curve.png')
        plt.show()

# 使用监控器
monitor = EvolutionMonitor()

async def monitored_evolution():
    population = create_initial_population(10)
    
    for generation in range(20):
        # 评估和进化
        fitness_scores = await evaluate_population(population)
        population = evolve_population_nsga2_simple(population, fitness_scores)
        
        # 记录数据
        monitor.record_generation(generation, fitness_scores)
        
        # 每5代显示进度
        if generation % 5 == 0:
            avg_fitness = sum(sum(scores) for scores in fitness_scores) / len(fitness_scores)
            print(f"世代 {generation}: 平均适应度 = {avg_fitness:.4f}")
    
    # 绘制进化曲线
    monitor.plot_evolution_curve()
```

### 2. 系统性能分析

```python
# 性能分析工具
import time
import psutil

class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
    
    def start_analysis(self):
        """开始分析"""
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().percent
    
    def end_analysis(self):
        """结束分析"""
        duration = time.time() - self.start_time
        end_memory = psutil.virtual_memory().percent
        memory_increase = end_memory - self.start_memory
        
        print(f"执行时间: {duration:.2f} 秒")
        print(f"内存增长: {memory_increase:.1f}%")
        print(f"最终内存使用: {end_memory:.1f}%")

# 使用性能分析器
analyzer = PerformanceAnalyzer()

async def analyzed_evolution():
    analyzer.start_analysis()
    
    # 执行进化过程
    population = create_initial_population(20)
    
    for generation in range(10):
        fitness_scores = await evaluate_population(population)
        population = evolve_population_nsga2_simple(population, fitness_scores)
    
    analyzer.end_analysis()
```

## 🎯 最佳实践

### 1. 配置优化

```python
# 推荐配置
recommended_config = {
    'population_size': 10,      # 适中的种群大小
    'generations': 20,          # 合理的进化代数
    'mutation_rate': 0.1,       # 适中的变异率
    'crossover_rate': 0.8,      # 较高的交叉率
    'evaluation_cache_ttl': 300, # 5分钟缓存
    'log_level': 'INFO'         # 生产环境日志级别
}
```

### 2. 错误处理

```python
# 健壮的错误处理
async def robust_evolution():
    try:
        population = create_initial_population(10)
        
        for generation in range(10):
            try:
                # 评估
                fitness_scores = await evaluate_population(population)
                
                # 进化
                population = evolve_population_nsga2_simple(population, fitness_scores)
                
                print(f"世代 {generation} 完成")
                
            except Exception as e:
                print(f"世代 {generation} 失败: {e}")
                # 继续下一代
                continue
        
        return population
        
    except Exception as e:
        print(f"进化过程失败: {e}")
        return None
```

### 3. 资源管理

```python
# 资源管理最佳实践
import gc
import torch

def cleanup_resources():
    """清理资源"""
    # 清理Python垃圾回收
    gc.collect()
    
    # 清理PyTorch缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 清理日志缓存
    import logging
    for handler in logging.getLogger().handlers:
        handler.flush()

# 在进化过程中定期清理
async def resource_managed_evolution():
    population = create_initial_population(10)
    
    for generation in range(20):
        # 执行进化
        fitness_scores = await evaluate_population(population)
        population = evolve_population_nsga2_simple(population, fitness_scores)
        
        # 每5代清理一次资源
        if generation % 5 == 0:
            cleanup_resources()
            print(f"世代 {generation}: 资源已清理")
```

## 📋 总结

AI自主进化系统提供了完整的自主进化能力，通过以下方式使用：

1. **快速开始**: 使用 `quick_evolution_validation.py` 验证系统能力
2. **基础使用**: 创建种群、执行进化、评估结果
3. **高级功能**: 自定义评估器、监控分析、性能优化
4. **故障排除**: 常见问题解决方案和调试技巧
5. **最佳实践**: 配置优化、错误处理、资源管理

系统设计为模块化和可扩展的，支持自定义评估器、进化参数和监控工具，满足不同应用场景的需求。

---

*使用指南生成时间: 2025-07-23 09:35*  
*系统版本: v1.0.0*  
*适用场景: AI模型进化、多目标优化、自主学习* 