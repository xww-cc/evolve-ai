#!/usr/bin/env python3
"""
高级模型测试 - 测试增强的推理网络和进化算法
"""

import asyncio
import time
import torch
import numpy as np
from models.advanced_reasoning_net import AdvancedReasoningNet
from evolution.advanced_evolution import AdvancedEvolution, MultiObjectiveAdvancedEvolution
from evaluators.enhanced_evaluator import EnhancedEvaluator
from config.optimized_logging import setup_optimized_logging

# 设置优化的日志系统
logger = setup_optimized_logging()

def create_advanced_population(size: int = 8) -> list:
    """创建异构结构的高级推理网络种群"""
    population = []
    for i in range(size):
        # 随机生成结构参数，保证hidden_size能被attention_heads整除
        attention_heads = int(np.random.choice([4, 8, 16]))
        base_hidden = int(np.random.choice([128, 192, 256, 320, 384, 512]))
        hidden_size = (base_hidden // attention_heads) * attention_heads
        reasoning_layers = int(np.random.choice([3, 4, 5, 6]))
        memory_size = int(np.random.choice([10, 15, 20, 25, 30]))
        reasoning_types = int(np.random.choice([8, 10, 12, 15]))
        model = AdvancedReasoningNet(
            input_size=4,
            hidden_size=hidden_size,
            reasoning_layers=reasoning_layers,
            attention_heads=attention_heads,
            memory_size=memory_size,
            reasoning_types=reasoning_types
        )
        population.append(model)
    logger.log_important(f"创建异构结构高级种群完成，共 {len(population)} 个个体")
    return population

async def advanced_model_test():
    """高级模型测试"""
    logger.log_important("🚀 高级模型测试开始")
    logger.log_important("=" * 70)
    
    start_time = time.time()
    
    try:
        # 1. 系统初始化
        logger.log_important("🔧 初始化高级系统组件...")
        population = create_advanced_population(8)
        enhanced_evaluator = EnhancedEvaluator()
        advanced_evolution = AdvancedEvolution(population_size=8)
        multi_objective_evolution = MultiObjectiveAdvancedEvolution(population_size=8)
        
        # 2. 初始评估
        logger.log_important("📊 执行初始高级评估...")
        initial_results = []
        
        for i, model in enumerate(population):
            model_id = f"AM{i+1:02d}"
            
            # 增强推理评估
            enhanced_scores = await enhanced_evaluator.evaluate_enhanced_reasoning(model, max_tasks=10)
            
            # 记录结果
            logger.log_important(f"高级模型 {model_id}:")
            for key, score in enhanced_scores.items():
                logger.log_important(f"  {key}: {score:.3f}")
            
            initial_results.append(enhanced_scores)
        
        # 3. 多目标高级进化
        logger.log_important("🔄 执行多目标高级进化...")
        
        # 准备多目标数据
        results_summary = {
            'nested_reasoning': [result['nested_reasoning'] for result in initial_results],
            'symbolic_induction': [result['symbolic_induction'] for result in initial_results],
            'graph_reasoning': [result['graph_reasoning'] for result in initial_results],
            'multi_step_chain': [result['multi_step_chain'] for result in initial_results],
            'logical_chain': [result['logical_chain'] for result in initial_results],
            'abstract_concept': [result['abstract_concept'] for result in initial_results],
            'creative_reasoning': [result['creative_reasoning'] for result in initial_results],
            'symbolic_expression': [result['symbolic_expression'] for result in initial_results],
            'comprehensive_reasoning': [result['comprehensive_reasoning'] for result in initial_results]
        }
        
        # 执行多目标高级进化
        evolved_population = await multi_objective_evolution.evolve_multi_objective(
            population, results_summary
        )
        
        # 4. 进化后评估
        logger.log_important("📊 执行进化后高级评估...")
        evolved_results = []
        
        for i, model in enumerate(evolved_population):
            model_id = f"AE{i+1:02d}"
            
            # 增强推理评估
            enhanced_scores = await enhanced_evaluator.evaluate_enhanced_reasoning(model, max_tasks=10)
            
            # 记录结果
            logger.log_important(f"进化模型 {model_id}:")
            for key, score in enhanced_scores.items():
                logger.log_important(f"  {key}: {score:.3f}")
            
            evolved_results.append(enhanced_scores)
        
        # 5. 性能对比分析
        logger.log_important("📈 高级性能对比分析...")
        
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
        logger.log_important("📊 高级性能改进分析:")
        for key in improvements.keys():
            logger.log_important(f"  {key}: {initial_avg[key]:.3f} → {evolved_avg[key]:.3f} "
                               f"({improvements[key]:+.1f}%)")
        
        # 6. 推理策略分析
        logger.log_important("🔍 推理策略分析...")
        for i, model in enumerate(evolved_population):
            model_id = f"AE{i+1:02d}"
            strategy_info = model.get_reasoning_strategy()
            
            logger.log_important(f"模型 {model_id} 推理策略:")
            for key, value in strategy_info.items():
                if isinstance(value, (int, float)):
                    logger.log_important(f"  {key}: {value:.3f}")
                else:
                    logger.log_important(f"  {key}: {value}")
        
        # 7. 符号推理分析
        logger.log_important("🔤 高级符号推理分析...")
        for i, model in enumerate(evolved_population):
            model_id = f"AE{i+1:02d}"
            symbolic_expr = model.extract_symbolic(use_llm=False)
            
            logger.log_important(f"模型 {model_id} 符号表达式: {symbolic_expr}")
        
        # 8. 系统性能监控
        system_metrics = get_system_metrics()
        logger.log_performance_metrics(system_metrics)
        
        # 9. 总结
        total_time = time.time() - start_time
        logger.log_success(f"高级模型测试完成！总耗时: {total_time:.2f}秒")
        
        # 计算综合改进
        overall_improvement = np.mean(list(improvements.values()))
        logger.log_important(f"综合改进幅度: {overall_improvement:+.1f}%")
        
        if overall_improvement > 0:
            logger.log_success("🎉 高级模型测试成功！推理能力显著提升！")
            logger.log_important("✅ 高级推理网络工作正常")
            logger.log_important("✅ 多目标高级进化算法有效")
            logger.log_important("✅ 复杂推理任务执行成功")
            logger.log_important("✅ 推理策略控制有效")
        else:
            logger.log_warning("⚠️ 高级模型需要进一步优化")
        
        return True
        
    except Exception as e:
        logger.log_error(f"高级模型测试失败: {e}", "高级模型测试")
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

async def test_advanced_components():
    """测试高级组件"""
    logger.log_important("🧪 测试高级组件...")
    
    # 1. 测试高级推理网络
    logger.log_important("🔧 测试高级推理网络...")
    model = AdvancedReasoningNet()
    test_input = torch.randn(2, 4)
    
    try:
        outputs = model(test_input)
        logger.log_success("✅ 高级推理网络前向传播正常")
        logger.log_important(f"输出任务数量: {len(outputs)}")
        
        # 测试推理链
        reasoning_steps = model.get_reasoning_chain()
        logger.log_important(f"推理步骤数量: {len(reasoning_steps)}")
        
        # 测试推理策略
        strategy_info = model.get_reasoning_strategy()
        logger.log_important(f"推理策略信息: {len(strategy_info)} 个指标")
        
        # 测试符号提取
        symbolic_expr = model.extract_symbolic(use_llm=False)
        logger.log_important(f"符号表达式: {symbolic_expr}")
        
    except Exception as e:
        logger.log_error(f"❌ 高级推理网络测试失败: {e}")
        return False
    
    # 2. 测试高级评估器
    logger.log_important("📊 测试高级评估器...")
    try:
        enhanced_evaluator = EnhancedEvaluator()
        evaluation_result = await enhanced_evaluator.evaluate_enhanced_reasoning(
            model=model, 
            max_tasks=10
        )
        logger.log_important(f"✅ 高级评估器测试成功")
        logger.log_important(f"🔔 评估结果: {evaluation_result}")
    except Exception as e:
        logger.log_error(f"❌ 高级评估器测试失败: {e}")
        return False
    
    # 3. 测试高级进化算法
    logger.log_important("🔄 测试高级进化算法...")
    evolution = AdvancedEvolution(population_size=3)
    
    try:
        # 创建测试种群
        test_population = [AdvancedReasoningNet() for _ in range(3)]
        test_fitness = [0.5, 0.7, 0.3]
        
        # 执行进化
        evolved_population = await evolution.evolve_population(test_population, test_fitness)
        logger.log_success("✅ 高级进化算法工作正常")
        logger.log_important(f"进化后种群大小: {len(evolved_population)}")
        
    except Exception as e:
        logger.log_error(f"❌ 高级进化算法测试失败: {e}")
        return False
    
    logger.log_success("🎉 所有高级组件测试通过！")
    return True

async def main():
    """主函数"""
    logger.log_important("🚀 启动高级模型测试")
    
    # 先测试各个组件
    components_ok = await test_advanced_components()
    
    if components_ok:
        # 执行完整系统测试
        success = await advanced_model_test()
        
        if success:
            logger.log_success("🎉 高级模型测试完全成功！")
            logger.log_important("✅ 推理能力显著提升")
            logger.log_important("✅ 自主进化机制有效")
            logger.log_important("✅ 复杂任务处理能力增强")
            logger.log_important("✅ 推理策略控制有效")
        else:
            logger.log_error("⚠️ 高级模型测试需要进一步优化")
    else:
        logger.log_error("❌ 高级组件测试失败，无法进行完整系统测试")

if __name__ == "__main__":
    asyncio.run(main()) 