#!/usr/bin/env python3
"""
系统问题诊断和优化脚本
解决发现的关键问题
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import torch
import warnings
from models.advanced_reasoning_net import AdvancedReasoningNet
from evaluators.enhanced_evaluator import EnhancedEvaluator
from evolution.advanced_evolution import AdvancedEvolution
from config.optimized_logging import setup_optimized_logging
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logger = setup_optimized_logging()

class SystemIssuesDiagnosis:
    """系统问题诊断和优化器"""
    
    def __init__(self):
        self.issues_found = []
        self.optimizations_applied = []
        
    async def diagnose_all_issues(self):
        """诊断所有系统问题"""
        logger.log_important("🔍 开始系统问题诊断")
        logger.log_important("=" * 50)
        
        # 1. 诊断NaN值问题
        await self._diagnose_nan_issues()
        
        # 2. 诊断中文字体问题
        await self._diagnose_chinese_font_issues()
        
        # 3. 诊断推理性能问题
        await self._diagnose_reasoning_performance()
        
        # 4. 诊断多样性计算问题
        await self._diagnose_diversity_issues()
        
        # 5. 诊断异步支持问题
        await self._diagnose_async_issues()
        
        # 6. 生成诊断报告
        self._generate_diagnosis_report()
        
        return self.issues_found
    
    async def _diagnose_nan_issues(self):
        """诊断NaN值问题"""
        logger.log_important("🔍 诊断NaN值问题...")
        
        try:
            # 创建测试模型
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            # 生成测试数据
            test_input = torch.randn(1, 4)
            
            # 前向传播
            with torch.no_grad():
                output = model(test_input)
            
            # 检查输出中是否有NaN
            nan_found = False
            if isinstance(output, dict):
                for key, value in output.items():
                    if torch.isnan(value).any():
                        nan_found = True
                        logger.log_warning(f"发现NaN值在输出键 '{key}' 中")
            elif torch.isnan(output).any():
                nan_found = True
                logger.log_warning("发现NaN值在模型输出中")
            
            if not nan_found:
                logger.log_success("✅ 模型输出无NaN值")
            else:
                self.issues_found.append({
                    'type': 'NaN值问题',
                    'severity': '中等',
                    'description': '模型输出包含NaN值',
                    'solution': '检查模型参数初始化和激活函数'
                })
                
        except Exception as e:
            logger.log_error(f"❌ NaN诊断失败: {e}")
            self.issues_found.append({
                'type': 'NaN诊断错误',
                'severity': '高',
                'description': f'NaN诊断过程出错: {e}',
                'solution': '检查模型结构和参数'
            })
    
    async def _diagnose_chinese_font_issues(self):
        """诊断中文字体问题"""
        logger.log_important("🔍 诊断中文字体问题...")
        
        try:
            # 测试中文字体支持
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot([1, 2, 3], [1, 4, 2])
            ax.set_title('测试中文标题')
            ax.set_xlabel('横轴标签')
            ax.set_ylabel('纵轴标签')
            
            # 尝试保存图片
            test_file = "test_chinese_font.png"
            plt.savefig(test_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            # 检查文件是否生成
            if os.path.exists(test_file):
                os.remove(test_file)
                logger.log_success("✅ 中文字体支持正常")
            else:
                raise Exception("图片文件未生成")
                
        except Exception as e:
            logger.log_warning(f"⚠️ 中文字体问题: {e}")
            self.issues_found.append({
                'type': '中文字体问题',
                'severity': '低',
                'description': '中文字体显示异常',
                'solution': '安装中文字体或使用英文标签'
            })
    
    async def _diagnose_reasoning_performance(self):
        """诊断推理性能问题"""
        logger.log_important("🔍 诊断推理性能问题...")
        
        try:
            # 创建测试模型
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            # 创建评估器
            evaluator = EnhancedEvaluator()
            
            # 测试推理性能
            import time
            start_time = time.time()
            
            # 运行推理评估
            result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=3)
            
            end_time = time.time()
            inference_time = (end_time - start_time) * 1000  # 转换为毫秒
            
            # 检查推理分数
            reasoning_score = result.get('comprehensive_reasoning', 0.0)
            
            logger.log_important(f"📊 推理性能测试结果:")
            logger.log_important(f"   推理时间: {inference_time:.2f} ms")
            logger.log_important(f"   推理分数: {reasoning_score:.4f}")
            
            # 性能评估
            if inference_time > 10.0:
                self.issues_found.append({
                    'type': '推理性能问题',
                    'severity': '中等',
                    'description': f'推理时间过长: {inference_time:.2f}ms',
                    'solution': '优化模型结构或使用量化技术'
                })
            
            if reasoning_score < 0.1:
                self.issues_found.append({
                    'type': '推理分数问题',
                    'severity': '高',
                    'description': f'推理分数过低: {reasoning_score:.4f}',
                    'solution': '改进推理算法和训练策略'
                })
            
            if inference_time <= 10.0 and reasoning_score >= 0.1:
                logger.log_success("✅ 推理性能良好")
                
        except Exception as e:
            logger.log_error(f"❌ 推理性能诊断失败: {e}")
            self.issues_found.append({
                'type': '推理性能诊断错误',
                'severity': '高',
                'description': f'推理性能诊断失败: {e}',
                'solution': '检查模型和评估器配置'
            })
    
    async def _diagnose_diversity_issues(self):
        """诊断多样性计算问题"""
        logger.log_important("🔍 诊断多样性计算问题...")
        
        try:
            # 创建测试种群
            population = []
            for i in range(4):
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=256 + i * 32,
                    reasoning_layers=5 + i,
                    attention_heads=8 + i,
                    memory_size=20,
                    reasoning_types=10
                )
                population.append(model)
            
            # 创建进化算法
            evolution = AdvancedEvolution(
                population_size=4,
                mutation_rate=0.1,
                crossover_rate=0.8,
                elite_size=1
            )
            
            # 计算多样性
            diversity = evolution._calculate_diversity(population)
            
            logger.log_important(f"📊 多样性计算结果: {diversity}")
            
            if np.isnan(diversity):
                self.issues_found.append({
                    'type': '多样性计算NaN问题',
                    'severity': '中等',
                    'description': '多样性计算结果为NaN',
                    'solution': '修复多样性计算算法，添加NaN检查'
                })
            elif diversity == 0:
                self.issues_found.append({
                    'type': '多样性为零问题',
                    'severity': '低',
                    'description': '种群多样性为零',
                    'solution': '增加种群多样性或调整参数'
                })
            else:
                logger.log_success(f"✅ 多样性计算正常: {diversity:.4f}")
                
        except Exception as e:
            logger.log_error(f"❌ 多样性诊断失败: {e}")
            self.issues_found.append({
                'type': '多样性诊断错误',
                'severity': '中等',
                'description': f'多样性诊断失败: {e}',
                'solution': '检查多样性计算算法'
            })
    
    async def _diagnose_async_issues(self):
        """诊断异步支持问题"""
        logger.log_important("🔍 诊断异步支持问题...")
        
        try:
            # 测试异步评估
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            evaluator = EnhancedEvaluator()
            
            # 测试并发评估
            import asyncio
            import time
            
            start_time = time.time()
            
            # 创建多个并发任务
            tasks = []
            for i in range(3):
                task = evaluator.evaluate_enhanced_reasoning(model, max_tasks=2)
                tasks.append(task)
            
            # 并发执行
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = (end_time - start_time) * 1000
            
            # 检查结果
            success_count = sum(1 for r in results if not isinstance(r, Exception))
            
            logger.log_important(f"📊 异步支持测试结果:")
            logger.log_important(f"   总时间: {total_time:.2f} ms")
            logger.log_important(f"   成功任务: {success_count}/{len(tasks)}")
            
            if success_count < len(tasks):
                self.issues_found.append({
                    'type': '异步支持问题',
                    'severity': '中等',
                    'description': f'异步任务部分失败: {success_count}/{len(tasks)}',
                    'solution': '改进异步实现和错误处理'
                })
            else:
                logger.log_success("✅ 异步支持正常")
                
        except Exception as e:
            logger.log_error(f"❌ 异步诊断失败: {e}")
            self.issues_found.append({
                'type': '异步诊断错误',
                'severity': '中等',
                'description': f'异步诊断失败: {e}',
                'solution': '检查异步实现和事件循环'
            })
    
    def _generate_diagnosis_report(self):
        """生成诊断报告"""
        logger.log_important("📋 系统问题诊断报告")
        logger.log_important("=" * 50)
        
        if not self.issues_found:
            logger.log_success("🎉 未发现系统问题！")
            return
        
        # 按严重程度分类
        high_severity = [issue for issue in self.issues_found if issue['severity'] == '高']
        medium_severity = [issue for issue in self.issues_found if issue['severity'] == '中等']
        low_severity = [issue for issue in self.issues_found if issue['severity'] == '低']
        
        logger.log_important(f"🔴 高严重程度问题 ({len(high_severity)}个):")
        for issue in high_severity:
            logger.log_important(f"   - {issue['type']}: {issue['description']}")
            logger.log_important(f"     解决方案: {issue['solution']}")
        
        logger.log_important(f"🟡 中等严重程度问题 ({len(medium_severity)}个):")
        for issue in medium_severity:
            logger.log_important(f"   - {issue['type']}: {issue['description']}")
            logger.log_important(f"     解决方案: {issue['solution']}")
        
        logger.log_important(f"🟢 低严重程度问题 ({len(low_severity)}个):")
        for issue in low_severity:
            logger.log_important(f"   - {issue['type']}: {issue['description']}")
            logger.log_important(f"     解决方案: {issue['solution']}")
        
        logger.log_important(f"\n📊 问题统计:")
        logger.log_important(f"   总问题数: {len(self.issues_found)}")
        logger.log_important(f"   高严重程度: {len(high_severity)}")
        logger.log_important(f"   中等严重程度: {len(medium_severity)}")
        logger.log_important(f"   低严重程度: {len(low_severity)}")

async def main():
    """主函数"""
    logger.log_important("=== 系统问题诊断和优化 ===")
    
    # 创建诊断器
    diagnosis = SystemIssuesDiagnosis()
    
    # 运行诊断
    issues = await diagnosis.diagnose_all_issues()
    
    if issues:
        logger.log_important(f"\n🔧 发现 {len(issues)} 个问题，建议优先解决高严重程度问题")
    else:
        logger.log_success("🎉 系统状态良好，未发现重大问题")

if __name__ == "__main__":
    asyncio.run(main()) 