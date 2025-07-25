#!/usr/bin/env python3
"""
训练和优化脚本
解决输出键缺失问题并提升模型性能
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
from typing import Dict, List, Any, Tuple
from models.advanced_reasoning_net import AdvancedReasoningNet
from evaluators.enhanced_evaluator import EnhancedEvaluator
from evolution.advanced_evolution import AdvancedEvolution
from utils.visualization import EvolutionVisualizer
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

class ModelTrainer:
    """模型训练器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.evaluator = EnhancedEvaluator()
        self.visualizer = EvolutionVisualizer()
        self.training_history = []
        
    async def train_model(self, model: AdvancedReasoningNet, 
                         epochs: int = 50, 
                         learning_rate: float = 0.001) -> AdvancedReasoningNet:
        """训练模型"""
        logger.log_important("🚀 开始模型训练")
        
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # 生成训练数据
        train_data = self._generate_training_data(1000)
        
        best_score = 0.0
        best_model = None
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            batch_count = 0
            
            # 分批训练
            for i in range(0, len(train_data), 32):
                batch_data = train_data[i:i+32]
                batch_inputs = torch.stack([item[0] for item in batch_data]).to(self.device)
                batch_targets = torch.stack([item[1] for item in batch_data]).to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(batch_inputs)
                
                # 计算损失
                loss = self._calculate_training_loss(outputs, batch_targets)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
            
            # 评估模型
            model.eval()
            with torch.no_grad():
                evaluation_result = await self.evaluator.evaluate_enhanced_reasoning(
                    model=model, max_tasks=20
                )
                current_score = evaluation_result.get('comprehensive_reasoning', 0.0)
            
            # 学习率调度
            scheduler.step(total_loss / batch_count)
            
            # 记录训练历史
            self.training_history.append({
                'epoch': epoch,
                'loss': total_loss / batch_count,
                'score': current_score,
                'learning_rate': optimizer.param_groups[0]['lr']
            })
            
            logger.log_important(f"🔔 Epoch {epoch+1}/{epochs}: Loss={total_loss/batch_count:.4f}, Score={current_score:.4f}")
            
            # 保存最佳模型
            if current_score > best_score:
                best_score = current_score
                best_model = model.state_dict().copy()
                logger.log_success(f"✅ 新的最佳分数: {best_score:.4f}")
        
        # 加载最佳模型
        if best_model is not None:
            model.load_state_dict(best_model)
            logger.log_success(f"🎉 训练完成！最佳分数: {best_score:.4f}")
        
        return model
    
    def _generate_training_data(self, num_samples: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """生成训练数据"""
        data = []
        
        for _ in range(num_samples):
            # 生成随机输入
            inputs = torch.randn(4)
            
            # 生成目标输出（基于输入的模式）
            target = torch.zeros(11)  # 11个任务输出
            
            # 数学逻辑任务
            if inputs.sum() > 0:
                target[0] = 1.0
            
            # 符号推理任务
            if inputs[0] > inputs[1]:
                target[1] = 1.0
            
            # 抽象推理任务
            if inputs.std() > 0.5:
                target[2] = 1.0
            
            # 模式识别任务
            if torch.all(inputs > 0):
                target[3] = 1.0
            
            # 推理链任务
            if inputs.max() > 1.0:
                target[4] = 1.0
            
            # 数学证明任务
            if inputs.min() < -1.0:
                target[5] = 1.0
            
            # 逻辑链任务
            if inputs.mean() > 0:
                target[6] = 1.0
            
            # 抽象概念任务
            if inputs.var() > 0.5:
                target[7] = 1.0
            
            # 创造性推理任务
            if torch.abs(inputs).sum() > 2.0:
                target[8] = 1.0
            
            # 综合推理任务
            target[9] = target[:9].mean()
            
            # 符号表达式任务
            if inputs[0] * inputs[1] > 0:
                target[10] = 1.0
            
            data.append((inputs, target))
        
        return data
    
    def _calculate_training_loss(self, outputs: Dict[str, torch.Tensor], 
                                targets: torch.Tensor) -> torch.Tensor:
        """计算训练损失"""
        loss = 0.0
        
        # 定义输出键的顺序
        output_keys = [
            'mathematical_logic', 'symbolic_reasoning', 'abstract_reasoning',
            'pattern_recognition', 'reasoning_chain', 'mathematical_proof',
            'logical_chain', 'abstract_concepts', 'creative_reasoning',
            'comprehensive_reasoning', 'symbolic_expression'
        ]
        
        for i, key in enumerate(output_keys):
            if key in outputs:
                output = outputs[key]
                # 确保输出在0-1范围内
                output = torch.clamp(output, 0.0, 1.0)
                target = targets[:, i:i+1]
                loss += nn.BCELoss()(output, target)
        
        return loss
    
    async def optimize_model(self, model: AdvancedReasoningNet, 
                           generations: int = 10) -> AdvancedReasoningNet:
        """优化模型"""
        logger.log_important("🔄 开始模型优化")
        
        # 创建进化算法
        evolution = AdvancedEvolution(
            population_size=8,
            mutation_rate=0.15,
            crossover_rate=0.8,
            elite_size=2
        )
        
        # 创建初始种群
        population = [model]
        for _ in range(7):
            new_model = AdvancedReasoningNet(
                input_size=model.input_size,
                hidden_size=model.hidden_size,
                reasoning_layers=model.reasoning_layers,
                attention_heads=model.attention_heads,
                memory_size=model.memory_size,
                reasoning_types=model.reasoning_types
            )
            new_model.load_state_dict(model.state_dict())
            population.append(new_model)
        
        best_model = model
        best_score = 0.0
        
        for generation in range(generations):
            logger.log_important(f"🔄 第 {generation + 1} 代优化")
            
            # 进化
            evolved_population = evolution.evolve(
                population=population,
                evaluator=self.evaluator,
                generations=1
            )
            
            # 评估种群
            scores = []
            for i, model in enumerate(evolved_population):
                try:
                    evaluation_result = await self.evaluator.evaluate_enhanced_reasoning(
                        model=model, max_tasks=10
                    )
                    score = evaluation_result.get('comprehensive_reasoning', 0.0)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                        logger.log_success(f"✅ 新的最佳分数: {best_score:.4f}")
                        
                except Exception as e:
                    logger.log_warning(f"⚠️ 模型 {i} 评估失败: {e}")
                    scores.append(0.0)
            
            # 记录进化历史
            avg_score = np.mean(scores)
            logger.log_important(f"🔔 第 {generation + 1} 代平均分数: {avg_score:.4f}")
            
            # 更新种群
            population = evolved_population
        
        logger.log_success(f"🎉 优化完成！最终最佳分数: {best_score:.4f}")
        return best_model
    
    def test_model_outputs(self, model: AdvancedReasoningNet) -> Dict[str, Any]:
        """测试模型输出"""
        logger.log_important("🔍 测试模型输出结构")
        
        model.eval()
        test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
        
        with torch.no_grad():
            outputs = model(test_input)
        
        # 检查输出键
        expected_keys = [
            'comprehensive_reasoning', 'symbolic_expression', 'reasoning_chain',
            'mathematical_logic', 'symbolic_reasoning', 'abstract_reasoning',
            'pattern_recognition', 'mathematical_proof', 'logical_chain',
            'abstract_concepts', 'creative_reasoning'
        ]
        
        results = {
            'output_keys': list(outputs.keys()),
            'missing_keys': [],
            'present_keys': [],
            'output_values': {}
        }
        
        for key in expected_keys:
            if key in outputs:
                results['present_keys'].append(key)
                results['output_values'][key] = outputs[key].mean().item()
            else:
                results['missing_keys'].append(key)
        
        logger.log_important(f"📊 输出键统计:")
        logger.log_important(f"  总期望键数: {len(expected_keys)}")
        logger.log_important(f"  实际输出键数: {len(results['output_keys'])}")
        logger.log_important(f"  缺失键数: {len(results['missing_keys'])}")
        
        if results['missing_keys']:
            logger.log_warning(f"⚠️ 缺失的键: {results['missing_keys']}")
        else:
            logger.log_success("✅ 所有期望的输出键都存在")
        
        return results
    
    def generate_training_report(self) -> Dict[str, Any]:
        """生成训练报告"""
        if not self.training_history:
            return {}
        
        scores = [entry['score'] for entry in self.training_history]
        losses = [entry['loss'] for entry in self.training_history]
        
        report = {
            'total_epochs': len(self.training_history),
            'final_score': scores[-1] if scores else 0.0,
            'best_score': max(scores) if scores else 0.0,
            'final_loss': losses[-1] if losses else 0.0,
            'best_loss': min(losses) if losses else 0.0,
            'score_improvement': scores[-1] - scores[0] if len(scores) > 1 else 0.0,
            'loss_improvement': losses[0] - losses[-1] if len(losses) > 1 else 0.0,
            'training_history': self.training_history
        }
        
        logger.log_important("📋 训练报告:")
        logger.log_important(f"  总训练轮数: {report['total_epochs']}")
        logger.log_important(f"  最终分数: {report['final_score']:.4f}")
        logger.log_important(f"  最佳分数: {report['best_score']:.4f}")
        logger.log_important(f"  分数提升: {report['score_improvement']:.4f}")
        logger.log_important(f"  最终损失: {report['final_loss']:.4f}")
        logger.log_important(f"  最佳损失: {report['best_loss']:.4f}")
        logger.log_important(f"  损失改善: {report['loss_improvement']:.4f}")
        
        return report

async def main():
    """主函数"""
    trainer = ModelTrainer()
    
    # 创建模型
    model = AdvancedReasoningNet(
        input_size=4,
        hidden_size=256,
        reasoning_layers=5,
        attention_heads=8,
        memory_size=20,
        reasoning_types=10
    )
    
    # 测试初始输出
    logger.log_important("🔍 初始模型输出测试")
    initial_test = trainer.test_model_outputs(model)
    
    # 训练模型
    logger.log_important("🚀 开始训练")
    trained_model = await trainer.train_model(model, epochs=30, learning_rate=0.001)
    
    # 测试训练后输出
    logger.log_important("🔍 训练后模型输出测试")
    trained_test = trainer.test_model_outputs(trained_model)
    
    # 优化模型
    logger.log_important("🔄 开始优化")
    optimized_model = await trainer.optimize_model(trained_model, generations=5)
    
    # 测试优化后输出
    logger.log_important("🔍 优化后模型输出测试")
    optimized_test = trainer.test_model_outputs(optimized_model)
    
    # 生成训练报告
    training_report = trainer.generate_training_report()
    
    # 最终评估
    logger.log_important("📊 最终评估")
    final_evaluation = await trainer.evaluator.evaluate_enhanced_reasoning(
        model=optimized_model, max_tasks=20
    )
    
    logger.log_important("🎉 训练和优化完成！")
    logger.log_important(f"📊 最终综合推理分数: {final_evaluation.get('comprehensive_reasoning', 0.0):.4f}")

if __name__ == "__main__":
    asyncio.run(main()) 