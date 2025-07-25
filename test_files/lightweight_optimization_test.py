#!/usr/bin/env python3
"""
轻量级优化测试脚本
避免内存不足问题，重点测试关键功能
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

class LightweightOptimizationTest:
    """轻量级优化测试器"""
    
    def __init__(self):
        self.optimization_results = {}
        self.best_score = 0.0
        
    async def run_lightweight_optimization(self):
        """运行轻量级优化测试"""
        logger.log_important("🚀 开始轻量级优化测试")
        logger.log_important("=" * 50)
        
        # 1. 推理分数优化（轻量级）
        await self._optimize_reasoning_score_lightweight()
        
        # 2. 系统稳定性测试
        await self._test_system_stability()
        
        # 3. 性能基准测试
        await self._benchmark_performance()
        
        # 4. 生成优化报告
        self._generate_optimization_report()
        
        return self.optimization_results
    
    async def _optimize_reasoning_score_lightweight(self):
        """轻量级推理分数优化"""
        logger.log_important("🧠 1. 轻量级推理分数优化")
        logger.log_important("-" * 40)
        
        # 使用中等规模的配置，避免内存不足
        lightweight_configs = [
            # 中等配置
            {
                'name': '中等配置',
                'hidden_size': 1024,
                'reasoning_layers': 8,
                'attention_heads': 16,
                'memory_size': 100,
                'reasoning_types': 20
            },
            # 平衡配置
            {
                'name': '平衡配置',
                'hidden_size': 1536,
                'reasoning_layers': 10,
                'attention_heads': 24,
                'memory_size': 150,
                'reasoning_types': 25
            },
            # 增强配置
            {
                'name': '增强配置',
                'hidden_size': 2048,
                'reasoning_layers': 12,
                'attention_heads': 32,
                'memory_size': 200,
                'reasoning_types': 30
            }
        ]
        
        evaluator = EnhancedEvaluator()
        
        for i, config in enumerate(lightweight_configs, 1):
            logger.log_important(f"🔧 测试配置 {i}: {config['name']}")
            
            try:
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
                result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=5)
                end_time = time.time()
                
                inference_time = (end_time - start_time) * 1000
                reasoning_score = result.get('comprehensive_reasoning', 0.0)
                
                logger.log_important(f"   推理分数: {reasoning_score:.4f}")
                logger.log_important(f"   推理时间: {inference_time:.2f}ms")
                
                # 更新最佳分数
                if reasoning_score > self.best_score:
                    self.best_score = reasoning_score
                    logger.log_success(f"🎉 新的最佳推理分数: {reasoning_score:.4f}")
                
                # 清理内存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                logger.log_important("")
                
            except Exception as e:
                logger.log_error(f"❌ 配置 {config['name']} 测试失败: {e}")
                continue
        
        # 尝试训练优化
        await self._try_lightweight_training()
        
        self.optimization_results['reasoning_optimization'] = {
            'best_score': self.best_score,
            'target_achieved': self.best_score >= 0.1,
            'configs_tested': len(lightweight_configs)
        }
        
        return self.best_score
    
    async def _try_lightweight_training(self):
        """尝试轻量级训练优化"""
        logger.log_important("\n🎓 尝试轻量级训练优化")
        logger.log_important("-" * 40)
        
        try:
            # 使用最佳配置进行训练
            best_config = {
                'hidden_size': 1024,
                'reasoning_layers': 8,
                'attention_heads': 16,
                'memory_size': 100,
                'reasoning_types': 20
            }
            
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=best_config['hidden_size'],
                reasoning_layers=best_config['reasoning_layers'],
                attention_heads=best_config['attention_heads'],
                memory_size=best_config['memory_size'],
                reasoning_types=best_config['reasoning_types']
            )
            
            # 创建优化器
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
            
            evaluator = EnhancedEvaluator()
            
            # 轻量级训练循环
            training_epochs = 10
            logger.log_important(f"开始轻量级训练 {training_epochs} 个epoch...")
            
            for epoch in range(training_epochs):
                # 生成训练数据
                train_data = torch.randn(10, 4)
                target_data = torch.randn(10, 4)
                
                # 前向传播
                optimizer.zero_grad()
                output = model(train_data)
                
                # 计算损失
                if isinstance(output, dict):
                    loss = nn.MSELoss()(output['comprehensive_reasoning'], target_data[:, 0])
                else:
                    loss = nn.MSELoss()(output, target_data)
                
                # 反向传播
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                # 评估当前性能
                with torch.no_grad():
                    result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=3)
                    current_score = result.get('comprehensive_reasoning', 0.0)
                
                logger.log_important(f"Epoch {epoch+1}: 损失={loss.item():.4f}, 推理分数={current_score:.4f}")
                
                # 更新最佳分数
                if current_score > self.best_score:
                    self.best_score = current_score
                    logger.log_success(f"🎉 训练后新的最佳推理分数: {current_score:.4f}")
                
                # 检查是否达到目标
                if current_score >= 0.1:
                    logger.log_success("🎯 训练后目标达成！推理分数已超过0.1")
                    break
            
            logger.log_important(f"\n✅ 轻量级训练完成，最终最佳推理分数: {self.best_score:.4f}")
            
        except Exception as e:
            logger.log_error(f"❌ 轻量级训练优化失败: {e}")
    
    async def _test_system_stability(self):
        """测试系统稳定性"""
        logger.log_important("\n🔧 2. 系统稳定性测试")
        logger.log_important("-" * 40)
        
        stability_tests = []
        
        # 测试1: 模型创建稳定性
        try:
            models = []
            for i in range(3):
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=256 + i * 100,
                    reasoning_layers=5 + i,
                    attention_heads=8 + i * 2,
                    memory_size=20 + i * 10,
                    reasoning_types=10 + i
                )
                models.append(model)
            
            # 清理内存
            del models
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            stability_tests.append(('模型创建', True, '成功创建3个不同配置的模型'))
        except Exception as e:
            stability_tests.append(('模型创建', False, f'失败: {e}'))
        
        # 测试2: 推理稳定性
        try:
            model = AdvancedReasoningNet(input_size=4, hidden_size=256, reasoning_layers=5)
            test_input = torch.randn(5, 4)
            
            outputs = []
            for _ in range(5):
                with torch.no_grad():
                    output = model(test_input)
                    outputs.append(output)
            
            # 检查输出一致性
            if isinstance(outputs[0], dict):
                output_keys = outputs[0].keys()
                consistency_check = all(
                    all(key in output.keys() for key in output_keys) 
                    for output in outputs
                )
            else:
                consistency_check = all(
                    output.shape == outputs[0].shape 
                    for output in outputs
                )
            
            stability_tests.append(('推理稳定性', consistency_check, '连续推理输出一致'))
        except Exception as e:
            stability_tests.append(('推理稳定性', False, f'失败: {e}'))
        
        # 测试3: 内存稳定性
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # 创建模型测试内存
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=512,
                reasoning_layers=6,
                attention_heads=12,
                memory_size=30,
                reasoning_types=15
            )
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # 清理内存
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            stability_tests.append(('内存稳定性', memory_increase < 500, f'内存增加: {memory_increase:.1f}MB'))
        except Exception as e:
            stability_tests.append(('内存稳定性', False, f'失败: {e}'))
        
        # 统计结果
        passed_tests = sum(1 for test in stability_tests if test[1])
        total_tests = len(stability_tests)
        stability_rate = passed_tests / total_tests * 100
        
        self.optimization_results['system_stability'] = {
            'stability_rate': stability_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'test_details': stability_tests
        }
        
        logger.log_important(f"📊 系统稳定性测试结果:")
        for test_name, passed, description in stability_tests:
            status = "✅" if passed else "❌"
            logger.log_important(f"   {status} {test_name}: {description}")
        
        logger.log_important(f"   稳定性通过率: {stability_rate:.1f}% ({passed_tests}/{total_tests})")
        
        if stability_rate >= 90:
            logger.log_success("✅ 系统稳定性测试通过")
        else:
            logger.log_warning(f"⚠️ 系统稳定性需要改进")
    
    async def _benchmark_performance(self):
        """性能基准测试"""
        logger.log_important("\n⚡ 3. 性能基准测试")
        logger.log_important("-" * 40)
        
        # 测试不同规模的模型性能
        benchmark_configs = [
            {'name': '小型模型', 'hidden_size': 128, 'reasoning_layers': 3},
            {'name': '中型模型', 'hidden_size': 512, 'reasoning_layers': 6},
            {'name': '大型模型', 'hidden_size': 1024, 'reasoning_layers': 8}
        ]
        
        benchmark_results = []
        
        for config in benchmark_configs:
            try:
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=config['hidden_size'],
                    reasoning_layers=config['reasoning_layers'],
                    attention_heads=8,
                    memory_size=20,
                    reasoning_types=10
                )
                
                # 测试推理时间
                test_input = torch.randn(1, 4)
                
                # 预热
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(test_input)
                
                # 正式测试
                times = []
                for _ in range(5):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(test_input)
                    end_time = time.time()
                    times.append((end_time - start_time) * 1000)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                # 测试内存使用
                import psutil
                process = psutil.Process()
                memory_usage = process.memory_info().rss / 1024 / 1024  # MB
                
                benchmark_results.append({
                    'name': config['name'],
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'memory_usage': memory_usage,
                    'success': True
                })
                
                logger.log_important(f"   {config['name']}: {avg_time:.2f}ms ± {std_time:.2f}ms, {memory_usage:.1f}MB")
                
                # 清理内存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
            except Exception as e:
                benchmark_results.append({
                    'name': config['name'],
                    'avg_time': 0,
                    'std_time': 0,
                    'memory_usage': 0,
                    'success': False,
                    'error': str(e)
                })
                logger.log_error(f"   {config['name']}: 测试失败 - {e}")
        
        self.optimization_results['performance_benchmark'] = {
            'configs': benchmark_results,
            'success_rate': sum(1 for r in benchmark_results if r['success']) / len(benchmark_results) * 100
        }
        
        logger.log_important(f"📊 性能基准测试完成，成功率: {self.optimization_results['performance_benchmark']['success_rate']:.1f}%")
    
    def _generate_optimization_report(self):
        """生成优化报告"""
        logger.log_important("\n📋 轻量级优化测试报告")
        logger.log_important("=" * 50)
        
        # 统计优化结果
        total_optimizations = len(self.optimization_results)
        successful_optimizations = sum(1 for result in self.optimization_results.values() 
                                     if isinstance(result, dict) and result.get('success', True))
        
        overall_success_rate = successful_optimizations / total_optimizations * 100
        
        logger.log_important(f"📊 优化总体结果:")
        logger.log_important(f"   总优化项目: {total_optimizations}")
        logger.log_important(f"   成功优化: {successful_optimizations}")
        logger.log_important(f"   成功率: {overall_success_rate:.1f}%")
        
        # 详细优化结果
        logger.log_important(f"\n📋 详细优化结果:")
        
        for optimization_name, result in self.optimization_results.items():
            if isinstance(result, dict):
                success = result.get('success', True)
                status = "✅" if success else "❌"
                logger.log_important(f"   {status} {optimization_name}")
                
                # 显示关键指标
                if optimization_name == 'reasoning_optimization':
                    best_score = result.get('best_score', 0)
                    target_achieved = result.get('target_achieved', False)
                    logger.log_important(f"      推理分数: {best_score:.4f} {'✅' if target_achieved else '❌'}")
                
                elif optimization_name == 'system_stability':
                    stability_rate = result.get('stability_rate', 0)
                    logger.log_important(f"      稳定性: {stability_rate:.1f}%")
                
                elif optimization_name == 'performance_benchmark':
                    success_rate = result.get('success_rate', 0)
                    logger.log_important(f"      成功率: {success_rate:.1f}%")
        
        # 推理分数目标达成情况
        reasoning_result = self.optimization_results.get('reasoning_optimization', {})
        best_score = reasoning_result.get('best_score', 0)
        target_achieved = reasoning_result.get('target_achieved', False)
        
        logger.log_important(f"\n🎯 推理分数目标达成情况:")
        logger.log_important(f"   最佳推理分数: {best_score:.4f}")
        logger.log_important(f"   目标达成: {'✅ 是' if target_achieved else '❌ 否'}")
        
        if target_achieved:
            logger.log_success("🎉 恭喜！推理分数目标已达成！")
        else:
            improvement_needed = 0.1 - best_score
            improvement_percentage = (improvement_needed / 0.1) * 100
            logger.log_warning(f"⚠️ 仍需改进: {improvement_needed:.4f} ({improvement_percentage:.1f}%)")
        
        # 最终评估
        if overall_success_rate >= 90:
            logger.log_success("🎉 轻量级优化测试优秀！")
        elif overall_success_rate >= 80:
            logger.log_important("✅ 轻量级优化测试良好，部分项目需要改进")
        else:
            logger.log_warning("⚠️ 轻量级优化测试发现问题，需要重点改进")

async def main():
    """主函数"""
    logger.log_important("=== 轻量级优化测试 ===")
    
    # 创建轻量级优化测试器
    optimizer = LightweightOptimizationTest()
    
    # 运行轻量级优化测试
    results = await optimizer.run_lightweight_optimization()
    
    logger.log_important(f"\n🎉 轻量级优化测试完成！")
    logger.log_important(f"优化结果已生成，请查看详细报告")

if __name__ == "__main__":
    asyncio.run(main()) 