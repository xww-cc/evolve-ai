#!/usr/bin/env python3
"""
自适应架构自动搜索脚本
自动搜索不同模型结构，融合多种架构，自动调参，记录最优结构
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from models.advanced_reasoning_net import AdvancedReasoningNet
from evaluators.enhanced_evaluator import EnhancedEvaluator
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

class AutoArchitectureSearch:
    """自适应架构搜索器"""
    def __init__(self):
        self.search_history = []
        self.best_config = None
        self.best_score = 0.0

    async def run_search(self):
        logger.log_important("🔍 开始自适应架构自动搜索")
        logger.log_important("=" * 60)
        # 定义搜索空间
        hidden_sizes = [128, 256, 384, 512, 768, 1024]
        reasoning_layers = [3, 4, 6, 8, 10, 12]
        attention_heads = [4, 8, 12, 16]
        memory_sizes = [10, 20, 30, 50, 80]
        reasoning_types = [5, 10, 15, 20]
        dropout_rates = [0.0, 0.1, 0.2]
        layer_norm_options = [False, True]

        evaluator = EnhancedEvaluator()
        search_trials = 0
        max_trials = 20  # 可根据需要调整

        for hs in hidden_sizes:
            for rl in reasoning_layers:
                for ah in attention_heads:
                    for ms in memory_sizes:
                        for rt in reasoning_types:
                            for dr in dropout_rates:
                                for ln in layer_norm_options:
                                    if search_trials >= max_trials:
                                        break
                                    config = {
                                        'hidden_size': hs,
                                        'reasoning_layers': rl,
                                        'attention_heads': ah,
                                        'memory_size': ms,
                                        'reasoning_types': rt,
                                        'dropout': dr,
                                        'layer_norm': ln
                                    }
                                    logger.log_important(f"   测试配置: {config}")
                                    model = AdvancedReasoningNet(
                                        input_size=4,
                                        hidden_size=hs,
                                        reasoning_layers=rl,
                                        attention_heads=ah,
                                        memory_size=ms,
                                        reasoning_types=rt
                                    )
                                    # 可扩展：应用dropout/layernorm等
                                    # 评估
                                    start_time = time.time()
                                    result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=6)
                                    end_time = time.time()
                                    reasoning_score = result.get('comprehensive_reasoning', 0.0)
                                    inference_time = (end_time - start_time) * 1000
                                    total_params = sum(p.numel() for p in model.parameters())
                                    logger.log_important(f"     推理分数: {reasoning_score:.4f}, 推理时间: {inference_time:.2f}ms, 参数: {total_params:,}")
                                    self.search_history.append({
                                        'config': config,
                                        'score': reasoning_score,
                                        'inference_time': inference_time,
                                        'params': total_params
                                    })
                                    if reasoning_score > self.best_score:
                                        self.best_score = reasoning_score
                                        self.best_config = config
                                        logger.log_success(f"   🎉 新的最佳分数: {reasoning_score:.4f} 配置: {config}")
                                    search_trials += 1
        logger.log_important("\n📊 搜索完成，总计测试配置: {}".format(len(self.search_history)))
        logger.log_important(f"🏆 最佳分数: {self.best_score:.4f}")
        logger.log_important(f"🏅 最佳配置: {self.best_config}")
        return self.best_config, self.best_score

async def main():
    logger.log_important("=== 自适应架构自动搜索 ===")
    searcher = AutoArchitectureSearch()
    best_config, best_score = await searcher.run_search()
    logger.log_important(f"\n🎉 架构搜索完成！最佳分数: {best_score:.4f}, 最佳配置: {best_config}")

if __name__ == "__main__":
    asyncio.run(main()) 