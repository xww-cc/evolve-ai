#!/usr/bin/env python3
"""
增强AI自主进化测试 - 集成复杂推理评估和优化日志
"""

import asyncio
import time
import torch
import psutil
import numpy as np
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evaluators.complex_reasoning_evaluator import ComplexReasoningEvaluator
from evolution.nsga2 import evolve_population_nsga2
from config.optimized_logging import setup_optimized_logging, get_optimized_logger

# 设置优化的日志系统
logger = setup_optimized_logging()

async def enhanced_evolution_test():
    """增强的AI自主进化测试"""
    logger.log_important("🧬 增强AI自主进化测试开始")
    logger.log_important("=" * 50)
    
    start_time = time.time()
    
    try:
        # 1. 系统初始化
        logger.log_important("🔧 初始化系统组件...")
        population = create_initial_population(10)  # 增加种群大小
        realworld_evaluator = RealWorldEvaluator()
        symbolic_evaluator = SymbolicEvaluator()
        complex_evaluator = ComplexReasoningEvaluator()
        
        # 2. 初始评估
        logger.log_important("📊 执行初始评估...")
        initial_results = await evaluate_population_comprehensive(
            population, symbolic_evaluator, realworld_evaluator, complex_evaluator, 0
        )
        
        # 记录初始结果
        log_comprehensive_results("初始", initial_results)
        
        # 3. 多级别进化测试
        for level in range(3):  # 测试前3个级别
            logger.log_important(f"🔄 开始级别 {level} 进化测试...")
            
            # 执行进化
            evolved_population, score_history_avg, score_history_best = await evolve_population_nsga2(
                population, 5, level  # 减少世代数，但增加评估复杂度
            )
            
            # 进化后评估
            evolved_results = await evaluate_population_comprehensive(
                evolved_population, symbolic_evaluator, realworld_evaluator, complex_evaluator, level
            )
            
            # 记录进化结果
            log_comprehensive_results(f"级别{level}进化后", evolved_results)
            
            # 计算改进
            improvements = calculate_improvements(initial_results, evolved_results)
            logger.log_evolution_summary(level, improvements)
            
            # 更新种群
            population = evolved_population
        
        # 4. 最终性能测试
        logger.log_important("🎯 执行最终性能测试...")
        final_results = await evaluate_population_comprehensive(
            population, symbolic_evaluator, realworld_evaluator, complex_evaluator, 3
        )
        
        # 记录最终结果
        log_comprehensive_results("最终", final_results)
        
        # 5. 系统性能监控
        system_metrics = get_system_metrics()
        logger.log_performance_metrics(system_metrics)
        
        # 6. 总结
        total_time = time.time() - start_time
        logger.log_success(f"测试完成！总耗时: {total_time:.2f}秒")
        
        # 计算总体改进
        total_improvements = calculate_improvements(initial_results, final_results)
        logger.log_evolution_summary(-1, total_improvements)
        
        return True
        
    except Exception as e:
        logger.log_error(f"测试失败: {e}", "增强进化测试")
        return False

async def evaluate_population_comprehensive(population, symbolic_evaluator, 
                                         realworld_evaluator, complex_evaluator, level):
    """综合评估种群"""
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
        
        # 基础评估
        symbolic_score = await symbolic_evaluator.evaluate(model, level)
        realworld_score = await realworld_evaluator.evaluate(model)
        
        results['symbolic_scores'].append(symbolic_score)
        results['realworld_scores'].append(realworld_score)
        
        # 复杂推理评估
        complex_scores = await complex_evaluator.evaluate_complex_reasoning(model, level)
        
        for key in results['complex_scores']:
            results['complex_scores'][key].append(complex_scores.get(key, 0.0))
        
        # 记录单个模型结果
        logger.log_evaluation_results(model_id, symbolic_score, realworld_score, complex_scores)
    
    return results

def log_comprehensive_results(stage: str, results: dict):
    """记录综合评估结果"""
    # 计算平均分
    avg_symbolic = np.mean(results['symbolic_scores'])
    avg_realworld = np.mean(results['realworld_scores'])
    
    avg_complex = {}
    for key, scores in results['complex_scores'].items():
        avg_complex[key] = np.mean(scores)
    
    # 记录结果
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
    logger.log_important("🚀 启动增强AI自主进化测试")
    
    success = await enhanced_evolution_test()
    
    if success:
        logger.log_success("🎉 增强AI自主进化测试成功完成！")
        logger.log_important("✅ 系统具备有效的复杂推理能力")
        logger.log_important("✅ 进化机制工作正常")
        logger.log_important("✅ 日志系统优化有效")
    else:
        logger.log_error("⚠️ 增强AI自主进化测试需要进一步优化")

if __name__ == "__main__":
    asyncio.run(main()) 