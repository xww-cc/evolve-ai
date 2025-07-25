#!/usr/bin/env python3
"""
自动化回归测试体系脚本
自动评测推理分数、稳定性、泛化能力等多维度，输出回归测试报告
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

class AutoRegressionTest:
    """自动化回归测试体系"""
    def __init__(self):
        self.results = {}

    async def run(self):
        logger.log_important("🧪 自动化回归测试开始")
        logger.log_important("=" * 60)
        evaluator = EnhancedEvaluator()
        # 测试模型配置
        configs = [
            {'name': 'baseline', 'hidden_size': 128, 'reasoning_layers': 4, 'attention_heads': 8, 'memory_size': 10, 'reasoning_types': 10},
            {'name': 'best_gnn', 'hidden_size': 128, 'reasoning_layers': 4, 'attention_heads': 8, 'memory_size': 10, 'reasoning_types': 10},
            {'name': 'large', 'hidden_size': 512, 'reasoning_layers': 8, 'attention_heads': 8, 'memory_size': 30, 'reasoning_types': 20}
        ]
        # 1. 推理分数测试
        for cfg in configs:
            logger.log_important(f"\n🔬 测试模型: {cfg['name']}")
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=cfg['hidden_size'],
                reasoning_layers=cfg['reasoning_layers'],
                attention_heads=cfg['attention_heads'],
                memory_size=cfg['memory_size'],
                reasoning_types=cfg['reasoning_types']
            )
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
            # 简单训练
            for epoch in range(3):
                train_data = torch.randn(20, 4)
                target_data = torch.randn(20)
                optimizer.zero_grad()
                output = model(train_data)
                if isinstance(output, dict):
                    loss = nn.MSELoss()(output['comprehensive_reasoning'], target_data)
                else:
                    loss = nn.MSELoss()(output, target_data)
                loss.backward()
                optimizer.step()
            # 多轮推理分数
            scores = []
            for _ in range(5):
                result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=6)
                score = result.get('comprehensive_reasoning', 0.0)
                scores.append(score)
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            logger.log_important(f"   平均推理分数: {mean_score:.4f}，标准差: {std_score:.4f}")
            self.results[cfg['name']] = {'mean_score': mean_score, 'std_score': std_score, 'all_scores': scores}
        # 2. 稳定性测试
        logger.log_important("\n🧷 稳定性测试")
        for name, res in self.results.items():
            stable = res['std_score'] < 0.01
            logger.log_important(f"   {name}: {'稳定' if stable else '波动'} (std={res['std_score']:.4f})")
            self.results[name]['stable'] = stable
        # 3. 泛化能力测试（不同分布输入）
        logger.log_important("\n🌏 泛化能力测试")
        for cfg in configs:
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=cfg['hidden_size'],
                reasoning_layers=cfg['reasoning_layers'],
                attention_heads=cfg['attention_heads'],
                memory_size=cfg['memory_size'],
                reasoning_types=cfg['reasoning_types']
            )
            # 不同分布输入
            test_inputs = [
                torch.randn(10, 4),
                torch.rand(10, 4),
                torch.abs(torch.randn(10, 4)),
                torch.ones(10, 4),
                torch.zeros(10, 4)
            ]
            gen_scores = []
            for inp in test_inputs:
                with torch.no_grad():
                    out = model(inp)
                if isinstance(out, dict):
                    score = out['comprehensive_reasoning'].mean().item()
                else:
                    score = out.mean().item()
                gen_scores.append(score)
            logger.log_important(f"   {cfg['name']} 泛化分布输出: {np.round(gen_scores, 4)}")
            self.results[cfg['name']]['generalization'] = gen_scores
        # 4. 性能与效率测试
        logger.log_important("\n⚡ 性能与效率测试")
        for cfg in configs:
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=cfg['hidden_size'],
                reasoning_layers=cfg['reasoning_layers'],
                attention_heads=cfg['attention_heads'],
                memory_size=cfg['memory_size'],
                reasoning_types=cfg['reasoning_types']
            )
            test_input = torch.randn(1, 4)
            with torch.no_grad():
                for _ in range(3):
                    _ = model(test_input)
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)
            end_time = time.time()
            avg_time = (end_time - start_time) / 10 * 1000
            logger.log_important(f"   {cfg['name']} 平均推理时间: {avg_time:.2f}ms")
            self.results[cfg['name']]['inference_time'] = avg_time
        # 5. 输出回归测试报告
        self._generate_report()
        return self.results

    def _generate_report(self):
        logger.log_important("\n📋 自动化回归测试报告")
        logger.log_important("=" * 60)
        for name, res in self.results.items():
            logger.log_important(f"\n模型: {name}")
            logger.log_important(f"   平均推理分数: {res['mean_score']:.4f}")
            logger.log_important(f"   分数标准差: {res['std_score']:.4f}")
            logger.log_important(f"   稳定性: {'稳定' if res['stable'] else '波动'}")
            logger.log_important(f"   泛化分布输出: {np.round(res['generalization'], 4)}")
            logger.log_important(f"   平均推理时间: {res['inference_time']:.2f}ms")
        logger.log_important("\n🎯 回归测试完成！")

async def main():
    logger.log_important("=== 自动化回归测试体系 ===")
    tester = AutoRegressionTest()
    results = await tester.run()
    logger.log_important(f"\n🎉 自动化回归测试全部完成！")

if __name__ == "__main__":
    asyncio.run(main()) 