#!/usr/bin/env python3
"""
深度融合优化自动闭环脚本
实现任务生成、架构搜索、训练与评估的自动循环，支持多轮自进化
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

class EvolutionaryClosedLoop:
    """深度融合优化自动闭环器"""
    def __init__(self, generations=3, population_size=5):
        self.generations = generations
        self.population_size = population_size
        self.history = []
        self.task_types = [
            'mathematical_logic', 'symbolic_reasoning', 'abstract_reasoning',
            'pattern_recognition', 'reasoning_chains', 'mathematical_proofs',
            'logical_chains', 'abstract_concepts', 'creative_reasoning',
            'multi_step_reasoning', 'nested_reasoning', 'symbolic_induction', 'graph_reasoning'
        ]
        self.difficulty_levels = ['easy', 'medium', 'hard', 'expert']

    async def run(self):
        logger.log_important("🔁 启动深度融合优化自动闭环")
        logger.log_important("=" * 60)
        evaluator = EnhancedEvaluator()
        best_overall = None
        best_score = 0.0
        for gen in range(1, self.generations + 1):
            logger.log_important(f"\n🌀 第{gen}代自进化循环")
            # 1. 任务生成
            tasks = self._generate_tasks()
            logger.log_important(f"   生成任务数: {len(tasks)}")
            # 2. 架构搜索（种群初始化）
            population = self._init_population()
            logger.log_important(f"   种群规模: {len(population)}")
            # 3. 训练与评估
            scores = []
            for idx, config in enumerate(population):
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=config['hidden_size'],
                    reasoning_layers=config['reasoning_layers'],
                    attention_heads=config['attention_heads'],
                    memory_size=config['memory_size'],
                    reasoning_types=config['reasoning_types']
                )
                optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
                # 简单训练（可扩展为多epoch/多任务）
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
                # 评估
                result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=6)
                score = result.get('comprehensive_reasoning', 0.0)
                logger.log_important(f"     个体{idx+1} 配置: {config} 分数: {score:.4f}")
                scores.append({'config': config, 'score': score})
                if score > best_score:
                    best_score = score
                    best_overall = config
            # 4. 选择与进化（简单保留top2+变异）
            scores = sorted(scores, key=lambda x: x['score'], reverse=True)
            survivors = [s['config'] for s in scores[:2]]
            new_population = survivors.copy()
            while len(new_population) < self.population_size:
                parent = np.random.choice(survivors)
                mutant = self._mutate_config(parent)
                new_population.append(mutant)
            self.history.append({'generation': gen, 'scores': scores, 'best': scores[0]})
            logger.log_success(f"   第{gen}代最佳分数: {scores[0]['score']:.4f} 配置: {scores[0]['config']}")
            # 下一代种群
            population = new_population
        logger.log_important("\n🏆 闭环优化完成！")
        logger.log_important(f"最佳分数: {best_score:.4f} 最佳配置: {best_overall}")
        return best_overall, best_score, self.history

    def _generate_tasks(self):
        tasks = []
        for level in self.difficulty_levels:
            for _ in range(2):
                task_type = np.random.choice(self.task_types)
                tasks.append({'level': level, 'type': task_type})
        return tasks

    def _init_population(self):
        # 随机初始化种群，确保hidden_size能被attention_heads整除
        pop = []
        tries = 0
        while len(pop) < self.population_size and tries < 100:
            hs = int(np.random.choice([128, 256, 384, 512]))
            ah = int(np.random.choice([4, 8, 12]))
            if hs % ah != 0:
                tries += 1
                continue
            config = {
                'hidden_size': hs,
                'reasoning_layers': int(np.random.choice([3, 4, 6, 8])),
                'attention_heads': ah,
                'memory_size': int(np.random.choice([10, 20, 30, 50])),
                'reasoning_types': int(np.random.choice([5, 10, 15, 20]))
            }
            pop.append(config)
            tries += 1
        return pop

    def _mutate_config(self, config):
        # 简单变异：随机微调一个参数，确保hidden_size能被attention_heads整除
        keys = list(config.keys())
        key = np.random.choice(keys)
        new_config = config.copy()
        tries = 0
        while tries < 20:
            if key == 'hidden_size':
                hs = int(np.random.choice([128, 256, 384, 512]))
                ah = new_config['attention_heads']
                if hs % ah == 0:
                    new_config['hidden_size'] = hs
                    break
            elif key == 'attention_heads':
                ah = int(np.random.choice([4, 8, 12]))
                hs = new_config['hidden_size']
                if hs % ah == 0:
                    new_config['attention_heads'] = ah
                    break
            elif key == 'reasoning_layers':
                new_config['reasoning_layers'] = int(np.random.choice([3, 4, 6, 8]))
                break
            elif key == 'memory_size':
                new_config['memory_size'] = int(np.random.choice([10, 20, 30, 50]))
                break
            elif key == 'reasoning_types':
                new_config['reasoning_types'] = int(np.random.choice([5, 10, 15, 20]))
                break
            tries += 1
        return new_config

async def main():
    logger.log_important("=== 深度融合优化自动闭环 ===")
    loop = EvolutionaryClosedLoop(generations=3, population_size=5)
    best_config, best_score, history = await loop.run()
    logger.log_important(f"\n🎉 闭环优化完成！最佳分数: {best_score:.4f}, 最佳配置: {best_config}")

if __name__ == "__main__":
    asyncio.run(main()) 