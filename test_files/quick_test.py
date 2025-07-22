#!/usr/bin/env python3
"""
快速测试脚本 - 日常系统状态验证
"""

import asyncio
import time
import sys
from evolution.population import create_initial_population
from evolution.nsga2 import evolve_population_nsga2
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from config.logging_setup import setup_logging

logger = setup_logging('quick_test.log')

async def quick_system_test():
    """快速系统测试"""
    print("🚀 开始快速系统测试...")
    start_time = time.time()
    
    try:
        # 1. 测试种群创建
        print("📊 测试种群创建...")
        population = create_initial_population(5)
        print(f"✅ 种群创建成功 - 大小: {len(population)}")
        
        # 2. 测试评估器
        print("🔧 测试评估器...")
        realworld_evaluator = RealWorldEvaluator()
        symbolic_evaluator = SymbolicEvaluator()
        
        # 3. 测试评估
        print("📈 测试评估功能...")
        fitness_scores = []
        for i, individual in enumerate(population):
            symbolic_score = await symbolic_evaluator.evaluate(individual)
            realworld_score = await realworld_evaluator.evaluate(individual)
            fitness_scores.append((symbolic_score, realworld_score))
            print(f"   个体{i+1}: 符号={symbolic_score:.3f}, 真实世界={realworld_score:.3f}")
        
        # 4. 测试进化
        print("🔄 测试进化算法...")
        evolved_population = evolve_population_nsga2(population, fitness_scores)
        print(f"✅ 进化成功 - 新种群大小: {len(evolved_population)}")
        
        # 5. 测试结果
        print("📊 测试结果统计...")
        avg_symbolic = sum(score[0] for score in fitness_scores) / len(fitness_scores)
        avg_realworld = sum(score[1] for score in fitness_scores) / len(fitness_scores)
        print(f"   平均符号得分: {avg_symbolic:.3f}")
        print(f"   平均真实世界得分: {avg_realworld:.3f}")
        
        # 6. 性能统计
        total_time = time.time() - start_time
        print(f"⏱️  总耗时: {total_time:.2f}秒")
        
        # 7. 系统状态评估
        print("\n🎯 系统状态评估:")
        if total_time < 5:
            print("   ⚡ 性能: 优秀")
        elif total_time < 10:
            print("   🟡 性能: 良好")
        else:
            print("   🔴 性能: 需要优化")
        
        if avg_symbolic > 0.8 and avg_realworld > 0.6:
            print("   🟢 评估: 正常")
        else:
            print("   🟡 评估: 需要改进")
        
        print("   ✅ 框架: 运行正常")
        
        print("\n🎉 快速测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = asyncio.run(quick_system_test())
    if success:
        print("\n✅ 系统状态正常")
        sys.exit(0)
    else:
        print("\n❌ 系统状态异常")
        sys.exit(1)

if __name__ == "__main__":
    main() 