#!/usr/bin/env python3
"""
可视化功能测试
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import torch
from models.advanced_reasoning_net import AdvancedReasoningNet
from evaluators.enhanced_evaluator import EnhancedEvaluator
from evolution.advanced_evolution import AdvancedEvolution
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

async def test_visualization():
    """测试可视化功能"""
    logger.log_important("🔔 🚀 启动可视化功能测试")
    
    try:
        # 1. 创建同构种群（避免异构结构问题）
        population = []
        base_model = AdvancedReasoningNet(
            input_size=4,
            hidden_size=256,
            reasoning_layers=5,
            attention_heads=8,
            memory_size=20,
            reasoning_types=10
        )
        
        # 复制相同结构的模型
        for i in range(4):
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            # 复制参数
            model.load_state_dict(base_model.state_dict())
            population.append(model)
        
        logger.log_important(f"🔔 创建测试种群完成，共 {len(population)} 个个体")
        
        # 2. 创建评估器
        evaluator = EnhancedEvaluator()
        
        # 3. 创建进化算法
        evolution = AdvancedEvolution(
            population_size=4,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elite_size=1
        )
        
        # 4. 执行进化（2代）
        logger.log_important("🔔 开始进化过程...")
        evolved_population = evolution.evolve(
            population=population,
            evaluator=evaluator,
            generations=2
        )
        
        logger.log_important(f"🔔 进化完成，最终种群大小: {len(evolved_population)}")
        
        # 5. 检查可视化文件
        import glob
        plot_files = glob.glob("evolution_plots/*.png")
        json_files = glob.glob("evolution_plots/*.json")
        
        logger.log_important(f"🔔 生成的可视化文件:")
        for file in plot_files + json_files:
            logger.log_important(f"  📊 {file}")
        
        if plot_files or json_files:
            logger.log_success("✅ 可视化功能测试成功！")
            return True
        else:
            logger.log_error("❌ 未找到可视化文件")
            return False
            
    except Exception as e:
        logger.log_error(f"❌ 可视化功能测试失败: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_visualization()) 