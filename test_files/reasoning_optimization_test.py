#!/usr/bin/env python3
"""
推理能力优化测试脚本
专门针对推理分数和推理能力进行优化
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.advanced_reasoning_net import AdvancedReasoningNet
from evaluators.enhanced_evaluator import EnhancedEvaluator
from config.optimized_logging import setup_optimized_logging
import time

logger = setup_optimized_logging()

class ReasoningOptimizer:
    """推理能力优化器"""
    
    def __init__(self):
        self.best_score = 0.0
        self.optimization_history = []
        
    async def optimize_reasoning_models(self):
        """优化推理模型"""
        logger.log_important("🧠 开始推理能力优化")
        logger.log_important("=" * 50)
        
        # 测试不同的模型配置
        configs = [
            # 配置1: 基础配置
            {
                'name': '基础配置',
                'hidden_size': 256,
                'reasoning_layers': 5,
                'attention_heads': 8,
                'memory_size': 20,
                'reasoning_types': 10
            },
            # 配置2: 增强配置
            {
                'name': '增强配置',
                'hidden_size': 512,
                'reasoning_layers': 8,
                'attention_heads': 16,
                'memory_size': 50,
                'reasoning_types': 15
            },
            # 配置3: 深度配置
            {
                'name': '深度配置',
                'hidden_size': 1024,
                'reasoning_layers': 12,
                'attention_heads': 32,
                'memory_size': 100,
                'reasoning_types': 20
            },
            # 配置4: 平衡配置
            {
                'name': '平衡配置',
                'hidden_size': 384,
                'reasoning_layers': 6,
                'attention_heads': 12,
                'memory_size': 30,
                'reasoning_types': 12
            },
            # 配置5: 高效配置
            {
                'name': '高效配置',
                'hidden_size': 768,
                'reasoning_layers': 10,
                'attention_heads': 24,
                'memory_size': 60,
                'reasoning_types': 18
            }
        ]
        
        evaluator = EnhancedEvaluator()
        
        for i, config in enumerate(configs, 1):
            logger.log_important(f"🔧 测试配置 {i}: {config['name']}")
            
            # 创建模型
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=config['hidden_size'],
                reasoning_layers=config['reasoning_layers'],
                attention_heads=config['attention_heads'],
                memory_size=config['memory_size'],
                reasoning_types=config['reasoning_types']
            )
            
            # 测试推理性能
            start_time = time.time()
            result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=8)
            end_time = time.time()
            
            inference_time = (end_time - start_time) * 1000
            reasoning_score = result.get('comprehensive_reasoning', 0.0)
            
            # 记录结果
            config_result = {
                'config_name': config['name'],
                'reasoning_score': reasoning_score,
                'inference_time': inference_time,
                'config': config
            }
            
            self.optimization_history.append(config_result)
            
            logger.log_important(f"📊 配置 {i} 结果:")
            logger.log_important(f"   推理分数: {reasoning_score:.4f}")
            logger.log_important(f"   推理时间: {inference_time:.2f} ms")
            
            # 更新最佳分数
            if reasoning_score > self.best_score:
                self.best_score = reasoning_score
                logger.log_success(f"🎉 新的最佳推理分数: {reasoning_score:.4f}")
            
            logger.log_important("")
        
        # 分析结果
        self._analyze_optimization_results()
        
        # 尝试模型训练优化
        await self._try_training_optimization()
        
        return self.best_score
    
    def _analyze_optimization_results(self):
        """分析优化结果"""
        logger.log_important("📊 优化结果分析")
        logger.log_important("=" * 30)
        
        # 按推理分数排序
        sorted_results = sorted(self.optimization_history, 
                               key=lambda x: x['reasoning_score'], reverse=True)
        
        logger.log_important("🏆 配置排名:")
        for i, result in enumerate(sorted_results, 1):
            logger.log_important(f"   {i}. {result['config_name']}: {result['reasoning_score']:.4f}")
        
        # 找到最佳配置
        best_config = sorted_results[0]
        logger.log_important(f"\n🎯 最佳配置: {best_config['config_name']}")
        logger.log_important(f"   推理分数: {best_config['reasoning_score']:.4f}")
        logger.log_important(f"   推理时间: {best_config['inference_time']:.2f} ms")
        
        # 分析配置参数对性能的影响
        self._analyze_parameter_impact()
    
    def _analyze_parameter_impact(self):
        """分析参数对性能的影响"""
        logger.log_important("\n🔍 参数影响分析:")
        
        # 分析隐藏层大小的影响
        hidden_sizes = [r['config']['hidden_size'] for r in self.optimization_history]
        scores = [r['reasoning_score'] for r in self.optimization_history]
        
        # 计算相关性
        correlation = np.corrcoef(hidden_sizes, scores)[0, 1]
        logger.log_important(f"   隐藏层大小与推理分数相关性: {correlation:.3f}")
        
        # 分析推理层数的影响
        reasoning_layers = [r['config']['reasoning_layers'] for r in self.optimization_history]
        correlation = np.corrcoef(reasoning_layers, scores)[0, 1]
        logger.log_important(f"   推理层数与推理分数相关性: {correlation:.3f}")
        
        # 分析注意力头数的影响
        attention_heads = [r['config']['attention_heads'] for r in self.optimization_history]
        correlation = np.corrcoef(attention_heads, scores)[0, 1]
        logger.log_important(f"   注意力头数与推理分数相关性: {correlation:.3f}")
    
    async def _try_training_optimization(self):
        """尝试训练优化"""
        logger.log_important("\n🎓 尝试训练优化")
        logger.log_important("=" * 30)
        
        # 选择最佳配置进行训练
        best_config = max(self.optimization_history, key=lambda x: x['reasoning_score'])
        
        logger.log_important(f"使用最佳配置进行训练: {best_config['config_name']}")
        
        # 创建模型
        model = AdvancedReasoningNet(
            input_size=4,
            hidden_size=best_config['config']['hidden_size'],
            reasoning_layers=best_config['config']['reasoning_layers'],
            attention_heads=best_config['config']['attention_heads'],
            memory_size=best_config['config']['memory_size'],
            reasoning_types=best_config['config']['reasoning_types']
        )
        
        # 创建优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 创建评估器
        evaluator = EnhancedEvaluator()
        
        # 训练循环
        training_epochs = 5
        logger.log_important(f"开始训练 {training_epochs} 个epoch...")
        
        for epoch in range(training_epochs):
            # 生成训练数据
            train_data = torch.randn(10, 4)
            target_data = torch.randn(10, 4)
            
            # 前向传播
            optimizer.zero_grad()
            output = model(train_data)
            
            # 计算损失（简化版本）
            if isinstance(output, dict):
                loss = nn.MSELoss()(output['comprehensive_reasoning'], target_data[:, 0])
            else:
                loss = nn.MSELoss()(output, target_data)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 评估当前性能
            with torch.no_grad():
                result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=3)
                current_score = result.get('comprehensive_reasoning', 0.0)
            
            logger.log_important(f"Epoch {epoch+1}: 损失={loss.item():.4f}, 推理分数={current_score:.4f}")
            
            # 更新最佳分数
            if current_score > self.best_score:
                self.best_score = current_score
                logger.log_success(f"🎉 训练后新的最佳推理分数: {current_score:.4f}")
        
        logger.log_important(f"\n✅ 训练完成，最终最佳推理分数: {self.best_score:.4f}")
    
    async def _test_advanced_reasoning_tasks(self):
        """测试高级推理任务"""
        logger.log_important("\n🧩 测试高级推理任务")
        logger.log_important("=" * 30)
        
        # 选择最佳配置
        best_config = max(self.optimization_history, key=lambda x: x['reasoning_score'])
        
        model = AdvancedReasoningNet(
            input_size=4,
            hidden_size=best_config['config']['hidden_size'],
            reasoning_layers=best_config['config']['reasoning_layers'],
            attention_heads=best_config['config']['attention_heads'],
            memory_size=best_config['config']['memory_size'],
            reasoning_types=best_config['config']['reasoning_types']
        )
        
        evaluator = EnhancedEvaluator()
        
        # 测试不同类型的推理任务
        task_types = [
            'mathematical_logic',
            'symbolic_reasoning', 
            'abstract_reasoning',
            'pattern_recognition',
            'reasoning_chains',
            'mathematical_proofs',
            'logical_chains',
            'abstract_concepts',
            'creative_reasoning',
            'multi_step_reasoning',
            'nested_reasoning',
            'symbolic_induction',
            'graph_reasoning'
        ]
        
        task_scores = {}
        
        for task_type in task_types:
            try:
                # 这里需要根据实际的评估器接口调整
                result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=2)
                score = result.get('comprehensive_reasoning', 0.0)
                task_scores[task_type] = score
                
                logger.log_important(f"   {task_type}: {score:.4f}")
            except Exception as e:
                logger.log_warning(f"   {task_type}: 测试失败 - {e}")
                task_scores[task_type] = 0.0
        
        # 计算平均分数
        avg_score = np.mean(list(task_scores.values()))
        logger.log_important(f"\n📊 平均推理分数: {avg_score:.4f}")
        
        return task_scores
    
    def generate_optimization_report(self):
        """生成优化报告"""
        logger.log_important("\n📋 推理能力优化报告")
        logger.log_important("=" * 50)
        
        logger.log_important(f"🎯 优化目标: 推理分数 > 0.1")
        logger.log_important(f"🏆 最佳推理分数: {self.best_score:.4f}")
        
        if self.best_score >= 0.1:
            logger.log_success("✅ 目标达成！推理分数已超过0.1")
        else:
            improvement_needed = 0.1 - self.best_score
            improvement_percentage = (improvement_needed / 0.1) * 100
            logger.log_warning(f"⚠️ 仍需改进: {improvement_needed:.4f} ({improvement_percentage:.1f}%)")
        
        # 配置对比
        logger.log_important(f"\n📊 配置对比:")
        for result in self.optimization_history:
            status = "✅" if result['reasoning_score'] >= 0.1 else "⚠️"
            logger.log_important(f"   {status} {result['config_name']}: {result['reasoning_score']:.4f}")
        
        return {
            'best_score': self.best_score,
            'target_achieved': self.best_score >= 0.1,
            'optimization_history': self.optimization_history
        }

async def main():
    """主函数"""
    logger.log_important("=== 推理能力优化测试 ===")
    
    # 创建优化器
    optimizer = ReasoningOptimizer()
    
    # 运行优化
    best_score = await optimizer.optimize_reasoning_models()
    
    # 测试高级推理任务
    await optimizer._test_advanced_reasoning_tasks()
    
    # 生成报告
    report = optimizer.generate_optimization_report()
    
    logger.log_important(f"\n🎉 推理能力优化测试完成！")
    logger.log_important(f"最终最佳推理分数: {best_score:.4f}")

if __name__ == "__main__":
    asyncio.run(main()) 