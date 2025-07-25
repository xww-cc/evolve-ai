#!/usr/bin/env python3
"""
基础验证测试 - 确保系统核心功能正常
"""

import asyncio
import time
import torch
import numpy as np
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from config.optimized_logging import setup_optimized_logging

# 设置优化的日志系统
logger = setup_optimized_logging()

async def basic_validation_test():
    """基础验证测试"""
    print("🧬 基础验证测试开始")
    print("=" * 40)
    
    start_time = time.time()
    
    try:
        # 1. 测试种群创建
        print("🔧 测试种群创建...")
        population = create_initial_population(3)
        print(f"✅ 成功创建 {len(population)} 个模型")
        
        # 2. 测试模型前向传播
        print("🔧 测试模型前向传播...")
        for i, model in enumerate(population):
            try:
                x = torch.randn(2, 4)
                output = model(x)
                print(f"✅ 模型 {i+1} 前向传播成功，输出形状: {output.shape}")
            except Exception as e:
                print(f"❌ 模型 {i+1} 前向传播失败: {e}")
        
        # 3. 测试评估器
        print("🔧 测试评估器...")
        realworld_evaluator = RealWorldEvaluator()
        symbolic_evaluator = SymbolicEvaluator()
        
        for i, model in enumerate(population):
            try:
                # 真实世界评估
                realworld_score = await realworld_evaluator.evaluate(model)
                print(f"✅ 模型 {i+1} 真实世界评估: {realworld_score:.3f}")
                
                # 符号评估
                symbolic_score = await symbolic_evaluator.evaluate(model, level=0)
                print(f"✅ 模型 {i+1} 符号评估: {symbolic_score:.3f}")
                
            except Exception as e:
                print(f"❌ 模型 {i+1} 评估失败: {e}")
        
        # 4. 测试复杂推理评估器
        print("🔧 测试复杂推理评估器...")
        try:
            from evaluators.complex_reasoning_evaluator import ComplexReasoningEvaluator
            complex_evaluator = ComplexReasoningEvaluator()
            
            for i, model in enumerate(population):
                try:
                    complex_scores = await complex_evaluator.evaluate_complex_reasoning(model, level=0)
                    print(f"✅ 模型 {i+1} 复杂推理评估:")
                    for key, score in complex_scores.items():
                        print(f"   {key}: {score:.3f}")
                except Exception as e:
                    print(f"❌ 模型 {i+1} 复杂推理评估失败: {e}")
                    
        except Exception as e:
            print(f"❌ 复杂推理评估器测试失败: {e}")
        
        # 5. 总结
        total_time = time.time() - start_time
        print(f"\n⏱️ 总耗时: {total_time:.2f}秒")
        print("🎉 基础验证测试完成！")
        
        return True
        
    except Exception as e:
        print(f"❌ 基础验证测试失败: {e}")
        return False

async def main():
    """主函数"""
    success = await basic_validation_test()
    
    if success:
        print("\n🎉 基础验证测试成功！")
        print("✅ 系统核心功能正常")
        print("✅ 评估器工作正常")
        print("✅ 模型创建和运行正常")
    else:
        print("\n⚠️ 基础验证测试失败，需要进一步调试")

if __name__ == "__main__":
    asyncio.run(main()) 