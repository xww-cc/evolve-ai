#!/usr/bin/env python3
"""
日志系统演示测试 - 展示优化的日志输出
"""

import asyncio
import time
import torch
import numpy as np
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evaluators.complex_reasoning_evaluator import ComplexReasoningEvaluator
from config.optimized_logging import setup_optimized_logging

# 设置优化的日志系统
logger = setup_optimized_logging()

async def logging_demo_test():
    """日志系统演示测试"""
    logger.log_important("🧬 日志系统演示测试开始")
    logger.log_important("=" * 50)
    
    start_time = time.time()
    
    try:
        # 1. 系统初始化
        logger.log_important("🔧 初始化系统组件...")
        population = create_initial_population(4)
        realworld_evaluator = RealWorldEvaluator()
        symbolic_evaluator = SymbolicEvaluator()
        complex_evaluator = ComplexReasoningEvaluator()
        
        # 2. 初始评估
        logger.log_important("📊 执行初始评估...")
        initial_results = await evaluate_population_with_logging(
            population, symbolic_evaluator, realworld_evaluator, complex_evaluator, 0
        )
        
        # 记录初始结果
        log_comprehensive_results("初始", initial_results)
        
        # 3. 进化测试
        logger.log_important("🔄 执行进化测试...")
        
        # 手动模拟进化过程
        evolved_population = []
        for i, model in enumerate(population):
            # 简单的模型复制和轻微修改
            evolved_model = copy_model_with_slight_modification(model)
            evolved_population.append(evolved_model)
        
        # 进化后评估
        evolved_results = await evaluate_population_with_logging(
            evolved_population, symbolic_evaluator, realworld_evaluator, complex_evaluator, 1
        )
        
        # 记录进化结果
        log_comprehensive_results("进化后", evolved_results)
        
        # 计算改进
        improvements = calculate_improvements(initial_results, evolved_results)
        logger.log_evolution_summary(1, improvements)
        
        # 4. 系统性能监控
        system_metrics = get_system_metrics()
        logger.log_performance_metrics(system_metrics)
        
        # 5. 总结
        total_time = time.time() - start_time
        logger.log_success(f"测试完成！总耗时: {total_time:.2f}秒")
        
        # 计算总体改进
        total_improvements = calculate_improvements(initial_results, evolved_results)
        logger.log_evolution_summary(-1, total_improvements)
        
        return True
        
    except Exception as e:
        logger.log_error(f"测试失败: {e}", "日志演示测试")
        return False

def copy_model_with_slight_modification(model):
    """复制模型并进行轻微修改"""
    try:
        # 创建新模型实例
        new_model = type(model)(model.modules_config, model.epigenetic_markers)
        
        # 复制参数并添加轻微噪声
        for param, new_param in zip(model.parameters(), new_model.parameters()):
            with torch.no_grad():
                noise = torch.randn_like(param) * 0.01  # 1%的噪声
                new_param.copy_(param + noise)
        
        return new_model
    except Exception as e:
        logger.log_warning(f"模型复制失败: {e}")
        return model

async def evaluate_population_with_logging(population, symbolic_evaluator, 
                                         realworld_evaluator, complex_evaluator, level):
    """带日志的种群评估"""
    results = {
        'symbolic_scores': [],
        'realworld_scores': [],
        'complex_scores': {
            'mathematical_logic': [],
            'symbolic_reasoning': [],
            'abstract_reasoning': [],
            'pattern_recognition': [],
            'reasoning_chain': []
        },
        'model_ids': []
    }
    
    for i, model in enumerate(population):
        model_id = f"M{i+1:02d}"
        results['model_ids'].append(model_id)
        
        try:
            # 基础评估
            symbolic_score = await symbolic_evaluator.evaluate(model, level)
            realworld_score = await realworld_evaluator.evaluate(model)
            
            results['symbolic_scores'].append(symbolic_score)
            results['realworld_scores'].append(realworld_score)
            
            # 复杂推理评估
            complex_scores = await complex_evaluator.evaluate_complex_reasoning(model, level)
            
            for key in results['complex_scores']:
                results['complex_scores'][key].append(complex_scores.get(key, 0.0))
            
            # 记录单个模型结果 - 使用优化的日志
            logger.log_evaluation_results(model_id, symbolic_score, realworld_score, complex_scores)
            
        except Exception as e:
            logger.log_warning(f"模型 {model_id} 评估失败: {e}")
            # 使用默认值
            results['symbolic_scores'].append(0.0)
            results['realworld_scores'].append(0.0)
            for key in results['complex_scores']:
                results['complex_scores'][key].append(0.0)
    
    return results

def log_comprehensive_results(stage: str, results: dict):
    """记录综合评估结果"""
    # 计算平均分
    avg_symbolic = np.mean(results['symbolic_scores'])
    avg_realworld = np.mean(results['realworld_scores'])
    
    avg_complex = {}
    for key, scores in results['complex_scores'].items():
        avg_complex[key] = np.mean(scores)
    
    # 记录结果 - 使用优化的日志
    logger.log_important(f"📊 {stage}评估结果:")
    logger.log_important(f"   符号推理: {avg_symbolic:.3f}")
    logger.log_important(f"   真实世界: {avg_realworld:.3f}")
    
    complex_str = " | ".join([f"{k}: {v:.3f}" for k, v in avg_complex.items()])
    logger.log_important(f"   复杂推理: {complex_str}")

def calculate_improvements(initial_results: dict, final_results: dict) -> dict:
    """计算改进幅度"""
    improvements = {}
    
    # 基础指标改进
    initial_symbolic = np.mean(initial_results['symbolic_scores'])
    final_symbolic = np.mean(final_results['symbolic_scores'])
    improvements['符号推理'] = final_symbolic - initial_symbolic
    
    initial_realworld = np.mean(initial_results['realworld_scores'])
    final_realworld = np.mean(final_results['realworld_scores'])
    improvements['真实世界'] = final_realworld - initial_realworld
    
    # 复杂推理改进
    for key in initial_results['complex_scores']:
        initial_avg = np.mean(initial_results['complex_scores'][key])
        final_avg = np.mean(final_results['complex_scores'][key])
        improvements[f'复杂推理_{key}'] = final_avg - initial_avg
    
    return improvements

def get_system_metrics() -> dict:
    """获取系统性能指标"""
    try:
        import psutil
        # 内存使用率
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # CPU使用率
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 其他指标
        metrics = {
            'memory_usage': memory_usage,
            'cpu_usage': cpu_usage,
            'available_memory_gb': memory.available / (1024**3),
            'total_memory_gb': memory.total / (1024**3)
        }
        
        return metrics
    except Exception as e:
        logger.log_warning(f"无法获取系统指标: {e}")
        return {'memory_usage': 0.0, 'cpu_usage': 0.0}

async def main():
    """主函数"""
    logger.log_important("🚀 启动日志系统演示测试")
    
    success = await logging_demo_test()
    
    if success:
        logger.log_success("🎉 日志系统演示测试成功完成！")
        logger.log_important("✅ 优化日志系统工作正常")
        logger.log_important("✅ 关键信息输出清晰")
        logger.log_important("✅ 性能监控有效")
    else:
        logger.log_error("⚠️ 日志系统演示测试需要进一步优化")

if __name__ == "__main__":
    asyncio.run(main()) 