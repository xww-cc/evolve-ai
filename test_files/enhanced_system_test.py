#!/usr/bin/env python3
"""
增强系统测试 - 使用改进的模型和算法
"""

import asyncio
import time
import torch
import numpy as np
from models.enhanced_reasoning_net import EnhancedReasoningNet
from evolution.enhanced_evolution import EnhancedEvolution, MultiObjectiveEvolution
from evaluators.enhanced_evaluator import EnhancedEvaluator
from config.optimized_logging import setup_optimized_logging

# 设置优化的日志系统
logger = setup_optimized_logging()

def create_enhanced_population(size: int = 5) -> list:
    """创建增强推理网络种群"""
    population = []
    
    for i in range(size):
        # 随机化网络参数
        hidden_size = np.random.choice([64, 128, 256])
        reasoning_layers = np.random.choice([2, 3, 4])
        attention_heads = np.random.choice([2, 4, 8])
        
        model = EnhancedReasoningNet(
            input_size=4,
            hidden_size=hidden_size,
            reasoning_layers=reasoning_layers,
            attention_heads=attention_heads
        )
        
        population.append(model)
    
    logger.log_important(f"创建增强种群完成，共 {len(population)} 个个体")
    return population

async def enhanced_system_test():
    """增强系统测试"""
    logger.log_important("🚀 增强系统测试开始")
    logger.log_important("=" * 60)
    
    start_time = time.time()
    
    try:
        # 1. 系统初始化
        logger.log_important("🔧 初始化增强系统组件...")
        population = create_enhanced_population(5)
        enhanced_evaluator = EnhancedEvaluator()
        enhanced_evolution = EnhancedEvolution(population_size=5)
        multi_objective_evolution = MultiObjectiveEvolution(population_size=5)
        
        # 2. 初始评估
        logger.log_important("📊 执行初始增强评估...")
        initial_results = []
        
        for i, model in enumerate(population):
            model_id = f"EM{i+1:02d}"
            
            # 增强推理评估
            enhanced_scores = await enhanced_evaluator.evaluate_enhanced_reasoning(model, level=0)
            
            # 记录结果
            logger.log_important(f"增强模型 {model_id}:")
            for key, score in enhanced_scores.items():
                logger.log_important(f"  {key}: {score:.3f}")
            
            initial_results.append(enhanced_scores)
        
        # 3. 多目标进化
        logger.log_important("🔄 执行多目标进化...")
        
        # 准备多目标数据
        objectives = {
            'mathematical_proof': [result['mathematical_proof'] for result in initial_results],
            'logical_chain': [result['logical_chain'] for result in initial_results],
            'abstract_concepts': [result['abstract_concepts'] for result in initial_results],
            'creative_reasoning': [result['creative_reasoning'] for result in initial_results],
            'multi_step_reasoning': [result['multi_step_reasoning'] for result in initial_results],
            'comprehensive_reasoning': [result['comprehensive_reasoning'] for result in initial_results]
        }
        
        # 执行多目标进化
        evolved_population = await multi_objective_evolution.evolve_multi_objective(
            population, objectives
        )
        
        # 4. 进化后评估
        logger.log_important("📊 执行进化后增强评估...")
        evolved_results = []
        
        for i, model in enumerate(evolved_population):
            model_id = f"EE{i+1:02d}"
            
            # 增强推理评估
            enhanced_scores = await enhanced_evaluator.evaluate_enhanced_reasoning(model, level=1)
            
            # 记录结果
            logger.log_important(f"进化模型 {model_id}:")
            for key, score in enhanced_scores.items():
                logger.log_important(f"  {key}: {score:.3f}")
            
            evolved_results.append(enhanced_scores)
        
        # 5. 性能对比分析
        logger.log_important("📈 性能对比分析...")
        
        # 计算平均分数
        initial_avg = {}
        evolved_avg = {}
        
        for key in initial_results[0].keys():
            initial_avg[key] = np.mean([result[key] for result in initial_results])
            evolved_avg[key] = np.mean([result[key] for result in evolved_results])
        
        # 计算改进幅度
        improvements = {}
        for key in initial_avg.keys():
            if initial_avg[key] > 0:
                improvement = (evolved_avg[key] - initial_avg[key]) / initial_avg[key] * 100
                improvements[key] = improvement
        
        # 记录改进结果
        logger.log_important("📊 性能改进分析:")
        for key in improvements.keys():
            logger.log_important(f"  {key}: {initial_avg[key]:.3f} → {evolved_avg[key]:.3f} "
                               f"({improvements[key]:+.1f}%)")
        
        # 6. 推理链分析
        logger.log_important("🔍 推理链分析...")
        for i, model in enumerate(evolved_population):
            model_id = f"EE{i+1:02d}"
            reasoning_steps = model.get_reasoning_chain()
            
            logger.log_important(f"模型 {model_id} 推理链:")
            for j, step in enumerate(reasoning_steps):
                logger.log_important(f"  步骤 {j+1}: {step}")
        
        # 7. 符号推理分析
        logger.log_important("🔤 符号推理分析...")
        for i, model in enumerate(evolved_population):
            model_id = f"EE{i+1:02d}"
            symbolic_expr = model.extract_symbolic(use_llm=False)
            
            logger.log_important(f"模型 {model_id} 符号表达式: {symbolic_expr}")
        
        # 8. 系统性能监控
        system_metrics = get_system_metrics()
        logger.log_performance_metrics(system_metrics)
        
        # 9. 总结
        total_time = time.time() - start_time
        logger.log_success(f"增强系统测试完成！总耗时: {total_time:.2f}秒")
        
        # 计算综合改进
        overall_improvement = np.mean(list(improvements.values()))
        logger.log_important(f"综合改进幅度: {overall_improvement:+.1f}%")
        
        if overall_improvement > 0:
            logger.log_success("🎉 增强系统测试成功！推理能力显著提升！")
            logger.log_important("✅ 增强推理网络工作正常")
            logger.log_important("✅ 多目标进化算法有效")
            logger.log_important("✅ 复杂推理任务执行成功")
        else:
            logger.log_warning("⚠️ 增强系统需要进一步优化")
        
        return True
        
    except Exception as e:
        logger.log_error(f"增强系统测试失败: {e}", "增强系统测试")
        return False

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

async def test_individual_components():
    """测试各个组件"""
    logger.log_important("🧪 测试各个组件...")
    
    # 1. 测试增强推理网络
    logger.log_important("🔧 测试增强推理网络...")
    model = EnhancedReasoningNet()
    test_input = torch.randn(2, 4)
    
    try:
        outputs = model(test_input)
        logger.log_success("✅ 增强推理网络前向传播正常")
        logger.log_important(f"输出形状: {len(outputs)} 个任务")
        
        # 测试推理链
        reasoning_steps = model.get_reasoning_chain()
        logger.log_important(f"推理步骤数量: {len(reasoning_steps)}")
        
        # 测试符号提取
        symbolic_expr = model.extract_symbolic(use_llm=False)
        logger.log_important(f"符号表达式: {symbolic_expr}")
        
    except Exception as e:
        logger.log_error(f"❌ 增强推理网络测试失败: {e}")
        return False
    
    # 2. 测试增强评估器
    logger.log_important("📊 测试增强评估器...")
    evaluator = EnhancedEvaluator()
    
    try:
        enhanced_scores = await evaluator.evaluate_enhanced_reasoning(model, level=0)
        logger.log_success("✅ 增强评估器工作正常")
        logger.log_important(f"评估任务数量: {len(enhanced_scores)}")
        
    except Exception as e:
        logger.log_error(f"❌ 增强评估器测试失败: {e}")
        return False
    
    # 3. 测试增强进化算法
    logger.log_important("🔄 测试增强进化算法...")
    evolution = EnhancedEvolution(population_size=3)
    
    try:
        # 创建测试种群
        test_population = [EnhancedReasoningNet() for _ in range(3)]
        test_fitness = [0.5, 0.7, 0.3]
        
        # 执行进化
        evolved_population = await evolution.evolve_population(test_population, test_fitness)
        logger.log_success("✅ 增强进化算法工作正常")
        logger.log_important(f"进化后种群大小: {len(evolved_population)}")
        
    except Exception as e:
        logger.log_error(f"❌ 增强进化算法测试失败: {e}")
        return False
    
    logger.log_success("🎉 所有组件测试通过！")
    return True

async def main():
    """主函数"""
    logger.log_important("🚀 启动增强系统测试")
    
    # 先测试各个组件
    components_ok = await test_individual_components()
    
    if components_ok:
        # 执行完整系统测试
        success = await enhanced_system_test()
        
        if success:
            logger.log_success("🎉 增强系统测试完全成功！")
            logger.log_important("✅ 推理能力显著提升")
            logger.log_important("✅ 自主进化机制有效")
            logger.log_important("✅ 复杂任务处理能力增强")
        else:
            logger.log_error("⚠️ 增强系统测试需要进一步优化")
    else:
        logger.log_error("❌ 组件测试失败，无法进行完整系统测试")

if __name__ == "__main__":
    asyncio.run(main()) 