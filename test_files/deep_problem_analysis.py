#!/usr/bin/env python3
"""
深度问题分析脚本
深入分析推理分数低的原因和根本问题
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
import matplotlib.pyplot as plt

logger = setup_optimized_logging()

class DeepProblemAnalysis:
    """深度问题分析器"""
    
    def __init__(self):
        self.analysis_results = {}
        self.problems_found = []
        self.solutions_proposed = []
        
    async def run_deep_analysis(self):
        """运行深度问题分析"""
        logger.log_important("🔍 开始深度问题分析")
        logger.log_important("=" * 60)
        
        # 1. 模型架构分析
        await self._analyze_model_architecture()
        
        # 2. 评估器分析
        await self._analyze_evaluator()
        
        # 3. 训练策略分析
        await self._analyze_training_strategy()
        
        # 4. 数据流分析
        await self._analyze_data_flow()
        
        # 5. 性能瓶颈分析
        await self._analyze_performance_bottlenecks()
        
        # 6. 生成深度分析报告
        self._generate_deep_analysis_report()
        
        return self.analysis_results
    
    async def _analyze_model_architecture(self):
        """分析模型架构问题"""
        logger.log_important("🏗️ 1. 模型架构分析")
        logger.log_important("-" * 40)
        
        architecture_issues = []
        
        # 分析1: 检查模型输出结构
        try:
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=512,
                reasoning_layers=6,
                attention_heads=8,
                memory_size=50,
                reasoning_types=15
            )
            
            test_input = torch.randn(1, 4)
            with torch.no_grad():
                output = model(test_input)
            
            # 分析输出结构
            if isinstance(output, dict):
                output_keys = list(output.keys())
                logger.log_important(f"   模型输出键: {output_keys}")
                
                # 检查是否有comprehensive_reasoning键
                if 'comprehensive_reasoning' in output:
                    reasoning_output = output['comprehensive_reasoning']
                    logger.log_important(f"   推理输出形状: {reasoning_output.shape}")
                    logger.log_important(f"   推理输出值范围: [{reasoning_output.min():.4f}, {reasoning_output.max():.4f}]")
                    logger.log_important(f"   推理输出均值: {reasoning_output.mean():.4f}")
                    logger.log_important(f"   推理输出标准差: {reasoning_output.std():.4f}")
                    
                    # 检查输出是否过于集中
                    if reasoning_output.std() < 0.01:
                        architecture_issues.append("推理输出方差过小，模型可能欠拟合")
                    
                    if abs(reasoning_output.mean()) > 0.5:
                        architecture_issues.append("推理输出均值偏离0，可能存在偏差")
                else:
                    architecture_issues.append("模型输出缺少comprehensive_reasoning键")
            else:
                logger.log_important(f"   模型输出类型: {type(output)}")
                logger.log_important(f"   模型输出形状: {output.shape}")
                architecture_issues.append("模型输出不是字典格式，可能影响评估")
            
        except Exception as e:
            architecture_issues.append(f"模型架构分析失败: {e}")
        
        # 分析2: 检查模型参数
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            logger.log_important(f"   总参数数量: {total_params:,}")
            logger.log_important(f"   可训练参数: {trainable_params:,}")
            
            if total_params < 10000:
                architecture_issues.append("模型参数过少，可能容量不足")
            elif total_params > 1000000:
                architecture_issues.append("模型参数过多，可能存在过拟合风险")
                
        except Exception as e:
            architecture_issues.append(f"参数分析失败: {e}")
        
        # 分析3: 检查梯度流
        try:
            test_input = torch.randn(1, 4)
            output = model(test_input)
            
            if isinstance(output, dict):
                loss = nn.MSELoss()(output['comprehensive_reasoning'], torch.tensor([0.5]))
            else:
                loss = nn.MSELoss()(output, torch.randn_like(output))
            
            loss.backward()
            
            # 检查梯度
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)
                    if grad_norm < 1e-6:
                        logger.log_warning(f"   参数 {name} 梯度过小: {grad_norm:.2e}")
            
            if grad_norms:
                avg_grad_norm = np.mean(grad_norms)
                logger.log_important(f"   平均梯度范数: {avg_grad_norm:.2e}")
                
                if avg_grad_norm < 1e-4:
                    architecture_issues.append("梯度消失问题，模型可能无法有效学习")
                elif avg_grad_norm > 10:
                    architecture_issues.append("梯度爆炸问题，训练可能不稳定")
                    
        except Exception as e:
            architecture_issues.append(f"梯度分析失败: {e}")
        
        self.analysis_results['model_architecture'] = {
            'issues': architecture_issues,
            'total_issues': len(architecture_issues)
        }
        
        logger.log_important(f"📊 模型架构问题: {len(architecture_issues)} 个")
        for issue in architecture_issues:
            logger.log_warning(f"   ⚠️ {issue}")
    
    async def _analyze_evaluator(self):
        """分析评估器问题"""
        logger.log_important("\n📊 2. 评估器分析")
        logger.log_important("-" * 40)
        
        evaluator_issues = []
        
        try:
            evaluator = EnhancedEvaluator()
            
            # 分析1: 检查评估任务
            logger.log_important("   分析评估任务...")
            
            # 创建测试模型
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=4,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            # 测试单个任务
            start_time = time.time()
            result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=1)
            end_time = time.time()
            
            logger.log_important(f"   单个任务评估时间: {(end_time - start_time)*1000:.2f}ms")
            logger.log_important(f"   评估结果: {result}")
            
            # 分析结果结构
            if isinstance(result, dict):
                result_keys = list(result.keys())
                logger.log_important(f"   评估结果键: {result_keys}")
                
                # 检查comprehensive_reasoning
                if 'comprehensive_reasoning' in result:
                    reasoning_score = result['comprehensive_reasoning']
                    logger.log_important(f"   推理分数: {reasoning_score:.4f}")
                    
                    # 分析分数分布
                    if reasoning_score < 0.01:
                        evaluator_issues.append("推理分数过低，可能评估标准过于严格")
                    elif reasoning_score > 0.9:
                        evaluator_issues.append("推理分数过高，可能评估标准过于宽松")
                else:
                    evaluator_issues.append("评估结果缺少comprehensive_reasoning键")
            else:
                evaluator_issues.append("评估结果不是字典格式")
            
            # 分析2: 检查任务多样性
            logger.log_important("   分析任务多样性...")
            
            # 多次评估检查一致性
            scores = []
            for i in range(5):
                result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=1)
                if isinstance(result, dict) and 'comprehensive_reasoning' in result:
                    scores.append(result['comprehensive_reasoning'])
            
            if scores:
                score_std = np.std(scores)
                logger.log_important(f"   多次评估分数: {[f'{s:.4f}' for s in scores]}")
                logger.log_important(f"   分数标准差: {score_std:.4f}")
                
                if score_std < 0.001:
                    evaluator_issues.append("评估结果过于一致，可能任务缺乏多样性")
                elif score_std > 0.1:
                    evaluator_issues.append("评估结果波动过大，可能任务过于随机")
            
        except Exception as e:
            evaluator_issues.append(f"评估器分析失败: {e}")
        
        self.analysis_results['evaluator'] = {
            'issues': evaluator_issues,
            'total_issues': len(evaluator_issues)
        }
        
        logger.log_important(f"📊 评估器问题: {len(evaluator_issues)} 个")
        for issue in evaluator_issues:
            logger.log_warning(f"   ⚠️ {issue}")
    
    async def _analyze_training_strategy(self):
        """分析训练策略问题"""
        logger.log_important("\n🎓 3. 训练策略分析")
        logger.log_important("-" * 40)
        
        training_issues = []
        
        try:
            # 创建模型
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=4,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            # 分析1: 损失函数分析
            logger.log_important("   分析损失函数...")
            
            # 生成测试数据
            train_data = torch.randn(10, 4)
            target_data = torch.randn(10, 4)
            
            # 测试不同损失函数
            loss_functions = [
                ('MSE', nn.MSELoss()),
                ('MAE', nn.L1Loss()),
                ('Huber', nn.HuberLoss()),
                ('SmoothL1', nn.SmoothL1Loss())
            ]
            
            for loss_name, loss_fn in loss_functions:
                output = model(train_data)
                
                if isinstance(output, dict):
                    loss = loss_fn(output['comprehensive_reasoning'], target_data[:, 0])
                else:
                    loss = loss_fn(output, target_data)
                
                logger.log_important(f"   {loss_name} 损失: {loss.item():.4f}")
            
            # 分析2: 优化器分析
            logger.log_important("   分析优化器...")
            
            optimizers = [
                ('Adam', optim.Adam(model.parameters(), lr=0.001)),
                ('AdamW', optim.AdamW(model.parameters(), lr=0.001)),
                ('SGD', optim.SGD(model.parameters(), lr=0.01)),
                ('RMSprop', optim.RMSprop(model.parameters(), lr=0.001))
            ]
            
            for opt_name, optimizer in optimizers:
                optimizer.zero_grad()
                output = model(train_data)
                
                if isinstance(output, dict):
                    loss = nn.MSELoss()(output['comprehensive_reasoning'], target_data[:, 0])
                else:
                    loss = nn.MSELoss()(output, target_data)
                
                loss.backward()
                
                # 检查梯度
                total_grad_norm = 0
                param_count = 0
                for param in model.parameters():
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    avg_grad_norm = (total_grad_norm / param_count) ** 0.5
                    logger.log_important(f"   {opt_name} 平均梯度范数: {avg_grad_norm:.2e}")
                
                optimizer.step()
            
            # 分析3: 学习率分析
            logger.log_important("   分析学习率...")
            
            learning_rates = [0.0001, 0.001, 0.01, 0.1]
            
            for lr in learning_rates:
                optimizer = optim.Adam(model.parameters(), lr=lr)
                optimizer.zero_grad()
                
                output = model(train_data)
                if isinstance(output, dict):
                    loss = nn.MSELoss()(output['comprehensive_reasoning'], target_data[:, 0])
                else:
                    loss = nn.MSELoss()(output, target_data)
                
                loss.backward()
                optimizer.step()
                
                logger.log_important(f"   学习率 {lr}: 损失 {loss.item():.4f}")
                
                if loss.item() > 10:
                    training_issues.append(f"学习率 {lr} 可能导致训练不稳定")
            
        except Exception as e:
            training_issues.append(f"训练策略分析失败: {e}")
        
        self.analysis_results['training_strategy'] = {
            'issues': training_issues,
            'total_issues': len(training_issues)
        }
        
        logger.log_important(f"📊 训练策略问题: {len(training_issues)} 个")
        for issue in training_issues:
            logger.log_warning(f"   ⚠️ {issue}")
    
    async def _analyze_data_flow(self):
        """分析数据流问题"""
        logger.log_important("\n🔄 4. 数据流分析")
        logger.log_important("-" * 40)
        
        data_flow_issues = []
        
        try:
            # 分析1: 输入数据分布
            logger.log_important("   分析输入数据分布...")
            
            # 生成不同分布的输入数据
            data_distributions = [
                ('正态分布', torch.randn(100, 4)),
                ('均匀分布', torch.rand(100, 4)),
                ('偏态分布', torch.abs(torch.randn(100, 4))),
                ('稀疏分布', torch.sparse_coo_tensor(
                    torch.randint(0, 4, (2, 50)), 
                    torch.randn(50), 
                    (100, 4)
                ).to_dense())
            ]
            
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=4,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            for dist_name, data in data_distributions:
                logger.log_important(f"   {dist_name}:")
                logger.log_important(f"     均值: {data.mean():.4f}")
                logger.log_important(f"     标准差: {data.std():.4f}")
                logger.log_important(f"     范围: [{data.min():.4f}, {data.max():.4f}]")
                
                # 测试模型对不同数据分布的反应
                with torch.no_grad():
                    output = model(data)
                
                if isinstance(output, dict):
                    reasoning_output = output['comprehensive_reasoning']
                    logger.log_important(f"     推理输出均值: {reasoning_output.mean():.4f}")
                    logger.log_important(f"     推理输出标准差: {reasoning_output.std():.4f}")
                    
                    # 检查输出是否对输入敏感
                    if reasoning_output.std() < 0.001:
                        data_flow_issues.append(f"{dist_name}数据下模型输出缺乏变化")
                else:
                    logger.log_important(f"     输出均值: {output.mean():.4f}")
                    logger.log_important(f"     输出标准差: {output.std():.4f}")
            
            # 分析2: 数据预处理
            logger.log_important("   分析数据预处理...")
            
            # 测试标准化效果
            raw_data = torch.randn(100, 4)
            normalized_data = (raw_data - raw_data.mean()) / raw_data.std()
            
            with torch.no_grad():
                raw_output = model(raw_data)
                norm_output = model(normalized_data)
            
            if isinstance(raw_output, dict) and isinstance(norm_output, dict):
                raw_score = raw_output['comprehensive_reasoning'].mean()
                norm_score = norm_output['comprehensive_reasoning'].mean()
                
                logger.log_important(f"   原始数据推理分数: {raw_score:.4f}")
                logger.log_important(f"   标准化数据推理分数: {norm_score:.4f}")
                
                if abs(raw_score - norm_score) < 0.001:
                    data_flow_issues.append("数据标准化对模型输出影响很小")
            
        except Exception as e:
            data_flow_issues.append(f"数据流分析失败: {e}")
        
        self.analysis_results['data_flow'] = {
            'issues': data_flow_issues,
            'total_issues': len(data_flow_issues)
        }
        
        logger.log_important(f"📊 数据流问题: {len(data_flow_issues)} 个")
        for issue in data_flow_issues:
            logger.log_warning(f"   ⚠️ {issue}")
    
    async def _analyze_performance_bottlenecks(self):
        """分析性能瓶颈"""
        logger.log_important("\n⚡ 5. 性能瓶颈分析")
        logger.log_important("-" * 40)
        
        performance_issues = []
        
        try:
            # 分析1: 计算复杂度
            logger.log_important("   分析计算复杂度...")
            
            model_sizes = [128, 256, 512, 1024]
            inference_times = []
            
            for hidden_size in model_sizes:
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=hidden_size,
                    reasoning_layers=4,
                    attention_heads=8,
                    memory_size=20,
                    reasoning_types=10
                )
                
                test_input = torch.randn(1, 4)
                
                # 预热
                with torch.no_grad():
                    for _ in range(3):
                        _ = model(test_input)
                
                # 测试推理时间
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(10):
                        _ = model(test_input)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10 * 1000  # ms
                inference_times.append(avg_time)
                
                logger.log_important(f"   隐藏层大小 {hidden_size}: {avg_time:.2f}ms")
            
            # 分析时间增长趋势
            if len(inference_times) >= 2:
                time_growth = inference_times[-1] / inference_times[0]
                logger.log_important(f"   时间增长倍数: {time_growth:.2f}")
                
                if time_growth > 10:
                    performance_issues.append("模型规模增长导致推理时间急剧增加")
            
            # 分析2: 内存使用
            logger.log_important("   分析内存使用...")
            
            import psutil
            process = psutil.Process()
            
            for hidden_size in [256, 512, 1024]:
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=hidden_size,
                    reasoning_layers=4,
                    attention_heads=8,
                    memory_size=20,
                    reasoning_types=10
                )
                
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = final_memory - initial_memory
                
                logger.log_important(f"   隐藏层大小 {hidden_size}: 内存增加 {memory_increase:.1f}MB")
                
                if memory_increase > 500:
                    performance_issues.append(f"隐藏层大小 {hidden_size} 内存使用过高")
                
                # 清理内存
                del model
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 分析3: 并行性能
            logger.log_important("   分析并行性能...")
            
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=512,
                reasoning_layers=4,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            # 测试批量处理
            batch_sizes = [1, 4, 8, 16]
            
            for batch_size in batch_sizes:
                batch_input = torch.randn(batch_size, 4)
                
                start_time = time.time()
                with torch.no_grad():
                    _ = model(batch_input)
                end_time = time.time()
                
                total_time = (end_time - start_time) * 1000  # ms
                avg_time_per_sample = total_time / batch_size
                
                logger.log_important(f"   批量大小 {batch_size}: 总时间 {total_time:.2f}ms, 平均 {avg_time_per_sample:.2f}ms/样本")
                
                if batch_size > 1 and avg_time_per_sample > inference_times[2]:  # 512 hidden_size的时间
                    performance_issues.append(f"批量大小 {batch_size} 并行效率低")
            
        except Exception as e:
            performance_issues.append(f"性能瓶颈分析失败: {e}")
        
        self.analysis_results['performance_bottlenecks'] = {
            'issues': performance_issues,
            'total_issues': len(performance_issues)
        }
        
        logger.log_important(f"📊 性能瓶颈问题: {len(performance_issues)} 个")
        for issue in performance_issues:
            logger.log_warning(f"   ⚠️ {issue}")
    
    def _generate_deep_analysis_report(self):
        """生成深度分析报告"""
        logger.log_important("\n📋 深度问题分析报告")
        logger.log_important("=" * 60)
        
        # 统计所有问题
        all_issues = []
        total_issues = 0
        
        for category, result in self.analysis_results.items():
            if isinstance(result, dict) and 'issues' in result:
                issues = result['issues']
                all_issues.extend([(category, issue) for issue in issues])
                total_issues += len(issues)
        
        logger.log_important(f"📊 问题统计:")
        logger.log_important(f"   总问题数量: {total_issues}")
        
        # 按类别统计
        for category, result in self.analysis_results.items():
            if isinstance(result, dict) and 'issues' in result:
                issue_count = len(result['issues'])
                logger.log_important(f"   {category}: {issue_count} 个问题")
        
        # 分析关键问题
        logger.log_important(f"\n🔍 关键问题分析:")
        
        # 找出最严重的问题
        critical_issues = []
        for category, issue in all_issues:
            if any(keyword in issue.lower() for keyword in ['失败', '错误', '崩溃', '无法']):
                critical_issues.append((category, issue))
            elif any(keyword in issue.lower() for keyword in ['过低', '过小', '不足', '缺乏']):
                critical_issues.append((category, issue))
        
        logger.log_important(f"   严重问题: {len(critical_issues)} 个")
        for category, issue in critical_issues[:5]:  # 显示前5个
            logger.log_warning(f"   🔴 {category}: {issue}")
        
        # 生成解决方案建议
        logger.log_important(f"\n💡 解决方案建议:")
        
        solutions = []
        
        # 基于问题类型提出解决方案
        if any('推理分数过低' in issue for _, issue in all_issues):
            solutions.append("1. 优化模型架构，增加模型复杂度")
            solutions.append("2. 改进训练策略，使用更合适的损失函数")
            solutions.append("3. 增加训练数据量和多样性")
        
        if any('梯度' in issue for _, issue in all_issues):
            solutions.append("4. 解决梯度消失/爆炸问题，调整学习率")
            solutions.append("5. 使用更好的激活函数和初始化方法")
        
        if any('内存' in issue for _, issue in all_issues):
            solutions.append("6. 优化内存使用，实现梯度检查点")
            solutions.append("7. 使用模型量化和压缩技术")
        
        if any('评估' in issue for _, issue in all_issues):
            solutions.append("8. 重新设计评估标准，确保合理性")
            solutions.append("9. 增加评估任务的多样性")
        
        if any('数据' in issue for _, issue in all_issues):
            solutions.append("10. 改进数据预处理和增强")
            solutions.append("11. 使用更好的数据分布")
        
        for solution in solutions:
            logger.log_important(f"   💡 {solution}")
        
        # 优先级排序
        logger.log_important(f"\n🎯 优先级排序:")
        logger.log_important(f"   高优先级: 推理分数优化、模型架构改进")
        logger.log_important(f"   中优先级: 训练策略优化、评估标准调整")
        logger.log_important(f"   低优先级: 性能优化、内存管理")
        
        # 时间规划
        logger.log_important(f"\n⏰ 时间规划:")
        logger.log_important(f"   立即行动 (1-2天): 修复严重错误，调整评估标准")
        logger.log_important(f"   短期改进 (1周): 优化模型架构，改进训练策略")
        logger.log_important(f"   中期优化 (1个月): 全面重构，性能优化")
        
        self.analysis_results['summary'] = {
            'total_issues': total_issues,
            'critical_issues': len(critical_issues),
            'solutions': solutions
        }

async def main():
    """主函数"""
    logger.log_important("=== 深度问题分析 ===")
    
    # 创建深度问题分析器
    analyzer = DeepProblemAnalysis()
    
    # 运行深度问题分析
    results = await analyzer.run_deep_analysis()
    
    logger.log_important(f"\n🎉 深度问题分析完成！")
    logger.log_important(f"发现 {results.get('summary', {}).get('total_issues', 0)} 个问题")
    logger.log_important(f"其中 {results.get('summary', {}).get('critical_issues', 0)} 个严重问题")

if __name__ == "__main__":
    asyncio.run(main()) 