#!/usr/bin/env python3
"""
快速AI自主进化验证 - 核心功能测试
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

async def quick_evolution_validation():
    """快速验证AI自主进化能力"""
    print("🧬 快速AI自主进化验证")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # 1. 基础功能验证
        print("🔧 验证基础功能...")
        population = create_initial_population(6)
        realworld_evaluator = RealWorldEvaluator()
        symbolic_evaluator = SymbolicEvaluator()
        
        # 初始评估
        initial_scores = []
        for i, individual in enumerate(population):
            symbolic_score = await symbolic_evaluator.evaluate(individual, level=0)
            realworld_score = await realworld_evaluator.evaluate(individual)
            initial_scores.append((symbolic_score, realworld_score))
            print(f"   个体{i+1}: 符号={symbolic_score:.3f}, 真实世界={realworld_score:.3f}")
        
        avg_initial_symbolic = sum(score[0] for score in initial_scores) / len(initial_scores)
        avg_initial_realworld = sum(score[1] for score in initial_scores) / len(initial_scores)
        print(f"📊 初始平均: 符号={avg_initial_symbolic:.3f}, 真实世界={avg_initial_realworld:.3f}")
        
        # 2. 进化验证
        print("🔄 执行进化...")
        evolved_population, _, _ = await evolve_population_nsga2(population, 2, 0)
        
        # 进化后评估
        evolved_scores = []
        for i, individual in enumerate(evolved_population):
            symbolic_score = await symbolic_evaluator.evaluate(individual, level=0)
            realworld_score = await realworld_evaluator.evaluate(individual)
            evolved_scores.append((symbolic_score, realworld_score))
            print(f"   个体{i+1}: 符号={symbolic_score:.3f}, 真实世界={realworld_score:.3f}")
        
        avg_evolved_symbolic = sum(score[0] for score in evolved_scores) / len(evolved_scores)
        avg_evolved_realworld = sum(score[1] for score in evolved_scores) / len(evolved_scores)
        print(f"📊 进化后平均: 符号={avg_evolved_symbolic:.3f}, 真实世界={avg_evolved_realworld:.3f}")
        
        # 3. 计算改进
        symbolic_improvement = avg_evolved_symbolic - avg_initial_symbolic
        realworld_improvement = avg_evolved_realworld - avg_initial_realworld
        total_improvement = symbolic_improvement + realworld_improvement
        
        print(f"📈 改进分析:")
        print(f"   符号能力改进: {symbolic_improvement:+.3f}")
        print(f"   真实世界能力改进: {realworld_improvement:+.3f}")
        print(f"   总改进: {total_improvement:+.3f}")
        
        # 4. 验证结果
        total_time = time.time() - start_time
        print(f"\n⏱️ 总耗时: {total_time:.2f}秒")
        
        # 5. 系统有效性评估
        print(f"\n🎯 AI自主进化系统验证结果:")
        
        if total_improvement > 0:
            print("   ✅ AI自主进化有效")
            print("   🎉 系统能够成功改进AI模型性能")
            print("   📊 进化机制工作正常")
            return True
        else:
            print("   ⚠️ AI自主进化效果有限")
            print("   🔧 建议进一步优化进化算法")
            return False
            
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

async def main():
    """主函数"""
    success = await quick_evolution_validation()
    
    if success:
        print("\n🎉 AI自主进化系统验证成功！")
        print("✅ 系统具备有效的自主进化能力")
    else:
        print("\n⚠️ AI自主进化系统需要进一步优化")

if __name__ == "__main__":
    asyncio.run(main()) 