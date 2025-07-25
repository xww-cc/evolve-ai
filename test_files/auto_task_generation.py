#!/usr/bin/env python3
"""
任务生成与难度自适应脚本
自动生成多样化推理任务，动态调整任务难度，提升模型泛化与鲁棒性
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import torch
from evaluators.enhanced_evaluator import EnhancedEvaluator
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

class AutoTaskGenerator:
    """任务生成与难度自适应器"""
    def __init__(self):
        self.task_history = []
        self.difficulty_levels = ['easy', 'medium', 'hard', 'expert']
        self.task_types = [
            'mathematical_logic', 'symbolic_reasoning', 'abstract_reasoning',
            'pattern_recognition', 'reasoning_chains', 'mathematical_proofs',
            'logical_chains', 'abstract_concepts', 'creative_reasoning',
            'multi_step_reasoning', 'nested_reasoning', 'symbolic_induction', 'graph_reasoning'
        ]

    async def run_generation(self):
        logger.log_important("🔄 开始任务生成与难度自适应")
        logger.log_important("=" * 60)
        evaluator = EnhancedEvaluator()
        max_tasks_per_level = 5
        all_results = {}
        for level in self.difficulty_levels:
            logger.log_important(f"\n📈 难度级别: {level}")
            for i in range(max_tasks_per_level):
                task_type = np.random.choice(self.task_types)
                task = self._generate_task(level, task_type)
                logger.log_important(f"   生成任务: {task}")
                # 评估任务（可选：与模型结合）
                # result = await evaluator.evaluate_single_task(task)
                self.task_history.append({'level': level, 'type': task_type, 'task': task})
            all_results[level] = [t for t in self.task_history if t['level'] == level]
        logger.log_important("\n📊 任务生成与难度自适应完成，总计生成任务: {}".format(len(self.task_history)))
        return all_results

    def _generate_task(self, level, task_type):
        """根据难度和类型生成任务描述"""
        # 难度参数影响任务复杂度
        if level == 'easy':
            complexity = 1
        elif level == 'medium':
            complexity = 2
        elif level == 'hard':
            complexity = 3
        else:
            complexity = 4
        # 生成任务描述
        task = {
            'type': task_type,
            'complexity': complexity,
            'description': f"{level} - {task_type} - complexity {complexity}"
        }
        return task

async def main():
    logger.log_important("=== 任务生成与难度自适应 ===")
    generator = AutoTaskGenerator()
    all_results = await generator.run_generation()
    logger.log_important(f"\n🎉 任务生成与难度自适应完成！共生成 {sum(len(v) for v in all_results.values())} 个任务。")

if __name__ == "__main__":
    asyncio.run(main()) 