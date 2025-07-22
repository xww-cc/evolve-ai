#!/usr/bin/env python3
"""
简化进化测试 - 验证AI自主进化能力
"""

import asyncio
import time
import torch
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evolution.nsga2 import evolve_population_nsga2
from config.logging_setup import setup_logging

logger = setup_logging()

async def test_evolution_capability():
    """测试AI自主进化能力"""
    print("🧬 开始AI自主进化能力测试...")
    start_time = time.time()
    
    try:
        # 1. 创建初始种群
        print("📊 创建初始种群...")
        population = create_initial_population(10)
        print(f"✅ 初始种群创建成功 - 大小: {len(population)}")
        
        # 2. 初始化评估器
        print("🔧 初始化评估器...")
        realworld_evaluator = RealWorldEvaluator()
        symbolic_evaluator = SymbolicEvaluator()
        
        # 3. 初始评估
        print("📈 执行初始评估...")
        initial_scores = []
        for i, individual in enumerate(population):
            symbolic_score = await symbolic_evaluator.evaluate(individual, level=0)
            realworld_score = await realworld_evaluator.evaluate(individual)
            initial_scores.append((symbolic_score, realworld_score))
            print(f"   个体{i+1}: 符号={symbolic_score:.3f}, 真实世界={realworld_score:.3f}")
        
        avg_initial_symbolic = sum(score[0] for score in initial_scores) / len(initial_scores)
        avg_initial_realworld = sum(score[1] for score in initial_scores) / len(initial_scores)
        print(f"📊 初始平均分数: 符号={avg_initial_symbolic:.3f}, 真实世界={avg_initial_realworld:.3f}")
        
        # 4. 执行进化
        print("🔄 开始进化过程...")
        evolved_population, _, _ = await evolve_population_nsga2(population, 3, 0)  # 3代进化，级别0
        print(f"✅ 进化完成 - 新种群大小: {len(evolved_population)}")
        
        # 5. 进化后评估
        print("📈 执行进化后评估...")
        evolved_scores = []
        for i, individual in enumerate(evolved_population):
            symbolic_score = await symbolic_evaluator.evaluate(individual, level=0)
            realworld_score = await realworld_evaluator.evaluate(individual)
            evolved_scores.append((symbolic_score, realworld_score))
            print(f"   个体{i+1}: 符号={symbolic_score:.3f}, 真实世界={realworld_score:.3f}")
        
        avg_evolved_symbolic = sum(score[0] for score in evolved_scores) / len(evolved_scores)
        avg_evolved_realworld = sum(score[1] for score in evolved_scores) / len(evolved_scores)
        print(f"📊 进化后平均分数: 符号={avg_evolved_symbolic:.3f}, 真实世界={avg_evolved_realworld:.3f}")
        
        # 6. 分析进化效果
        print("📊 分析进化效果...")
        symbolic_improvement = avg_evolved_symbolic - avg_initial_symbolic
        realworld_improvement = avg_evolved_realworld - avg_initial_realworld
        
        print(f"📈 符号能力改进: {symbolic_improvement:+.3f}")
        print(f"📈 真实世界能力改进: {realworld_improvement:+.3f}")
        
        # 7. 验证自主进化有效性
        total_improvement = symbolic_improvement + realworld_improvement
        if total_improvement > 0:
            print("✅ AI自主进化有效 - 整体能力得到提升")
            evolution_status = "有效"
        else:
            print("⚠️ AI自主进化效果有限 - 需要进一步优化")
            evolution_status = "有限"
        
        # 8. 计算多样性
        print("🌐 分析种群多样性...")
        initial_diversity = len(set(str(ind.modules_config) for ind in population))
        evolved_diversity = len(set(str(ind.modules_config) for ind in evolved_population))
        
        print(f"📊 初始多样性: {initial_diversity}")
        print(f"📊 进化后多样性: {evolved_diversity}")
        
        # 9. 生成测试报告
        total_time = time.time() - start_time
        print(f"\n⏱️ 总耗时: {total_time:.2f}秒")
        
        print(f"\n🎯 AI自主进化测试结果:")
        print(f"   📊 进化状态: {evolution_status}")
        print(f"   📈 符号能力改进: {symbolic_improvement:+.3f}")
        print(f"   📈 真实世界能力改进: {realworld_improvement:+.3f}")
        print(f"   🌐 多样性保持: {'良好' if evolved_diversity >= initial_diversity * 0.8 else '需要改进'}")
        print(f"   ⚡ 性能: {'优秀' if total_time < 10 else '良好'}")
        
        return evolution_status == "有效"
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_evolution_capability())
    if success:
        print("\n🎉 AI自主进化系统验证成功！")
    else:
        print("\n⚠️ AI自主进化系统需要进一步优化") 