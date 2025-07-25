#!/usr/bin/env python3
"""
调试测试 - 定位张量转换问题
"""

import torch
import numpy as np
from models.advanced_reasoning_net import AdvancedReasoningNet

def debug_test():
    """调试测试"""
    print("开始调试测试...")
    
    try:
        # 创建模型
        model = AdvancedReasoningNet()
        print("✅ 模型创建成功")
        
        # 创建测试输入
        test_input = torch.randn(2, 4)
        print("✅ 测试输入创建成功")
        
        # 测试前向传播
        outputs = model(test_input)
        print("✅ 前向传播成功")
        print(f"输出任务数量: {len(outputs)}")
        
        # 测试推理链
        reasoning_steps = model.get_reasoning_chain()
        print(f"推理步骤数量: {len(reasoning_steps)}")
        
        # 测试推理策略
        strategy_info = model.get_reasoning_strategy()
        print(f"推理策略信息: {len(strategy_info)} 个指标")
        
        # 测试符号提取
        symbolic_expr = model.extract_symbolic(use_llm=False)
        print(f"符号表达式: {symbolic_expr}")
        
        print("🎉 所有测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_test() 