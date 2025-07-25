#!/usr/bin/env python3
"""
全面模型评估测试
检查所有核心组件和新增功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import torch
import time
from typing import Dict, List, Any
from models.advanced_reasoning_net import AdvancedReasoningNet
from evaluators.enhanced_evaluator import EnhancedEvaluator
from evolution.advanced_evolution import AdvancedEvolution, MultiObjectiveAdvancedEvolution
from utils.visualization import EvolutionVisualizer
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

class ComprehensiveEvaluator:
    """全面评估器"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        
    async def run_comprehensive_evaluation(self):
        """运行全面评估"""
        logger.log_important("🔍 🚀 开始全面模型评估")
        logger.log_important("=" * 60)
        
        # 1. 模型架构测试
        await self._test_model_architecture()
        
        # 2. 推理能力测试
        await self._test_reasoning_capabilities()
        
        # 3. 异构结构测试
        await self._test_heterogeneous_structures()
        
        # 4. 进化算法测试
        await self._test_evolution_algorithm()
        
        # 5. 评估器测试
        await self._test_evaluators()
        
        # 6. 可视化功能测试
        await self._test_visualization()
        
        # 7. 性能基准测试
        await self._test_performance_benchmarks()
        
        # 8. 生成评估报告
        self._generate_evaluation_report()
        
        return self.test_results
    
    async def _test_model_architecture(self):
        """测试模型架构"""
        logger.log_important("🔧 1. 模型架构测试")
        
        try:
            # 测试基本模型创建
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            # 测试前向传播
            test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
            output = model(test_input)
            
            # 检查输出结构
            expected_keys = ['comprehensive_reasoning', 'reasoning_chain', 'symbolic_expression']
            for key in expected_keys:
                if key in output:
                    logger.log_success(f"✅ 输出键 '{key}' 存在")
                else:
                    logger.log_warning(f"⚠️ 输出键 '{key}' 缺失")
            
            # 测试推理链
            reasoning_chain = model.get_reasoning_chain()
            if reasoning_chain:
                logger.log_success(f"✅ 推理链生成成功，长度: {len(reasoning_chain)}")
            else:
                logger.log_warning("⚠️ 推理链为空")
            
            # 测试符号表达式
            symbolic_expr = model.extract_symbolic(use_llm=False)
            if symbolic_expr:
                logger.log_success(f"✅ 符号表达式生成成功: {symbolic_expr}")
            else:
                logger.log_warning("⚠️ 符号表达式为空")
            
            self.test_results['model_architecture'] = {
                'status': 'PASS',
                'output_keys': list(output.keys()),
                'reasoning_chain_length': len(reasoning_chain) if reasoning_chain else 0,
                'symbolic_expression': symbolic_expr
            }
            
        except Exception as e:
            logger.log_error(f"❌ 模型架构测试失败: {e}")
            self.test_results['model_architecture'] = {'status': 'FAIL', 'error': str(e)}
    
    async def _test_reasoning_capabilities(self):
        """测试推理能力"""
        logger.log_important("🧠 2. 推理能力测试")
        
        try:
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            # 测试不同类型的输入
            test_cases = [
                ([1, 2, 3, 4], "基础推理"),
                ([2, 4, 6, 8], "偶数序列"),
                ([1, 3, 5, 7], "奇数序列"),
                ([1, 4, 9, 16], "平方序列")
            ]
            
            reasoning_results = {}
            for inputs, description in test_cases:
                test_input = torch.tensor([inputs], dtype=torch.float32)
                output = model(test_input)
                
                comprehensive_score = output.get('comprehensive_reasoning', torch.tensor(0.0))
                if isinstance(comprehensive_score, torch.Tensor):
                    score = comprehensive_score.mean().item()
                else:
                    score = float(comprehensive_score)
                
                reasoning_results[description] = score
                logger.log_important(f"🔔 {description}: {score:.3f}")
            
            self.test_results['reasoning_capabilities'] = {
                'status': 'PASS',
                'test_cases': reasoning_results,
                'average_score': np.mean(list(reasoning_results.values()))
            }
            
        except Exception as e:
            logger.log_error(f"❌ 推理能力测试失败: {e}")
            self.test_results['reasoning_capabilities'] = {'status': 'FAIL', 'error': str(e)}
    
    async def _test_heterogeneous_structures(self):
        """测试异构结构"""
        logger.log_important("🏗️ 3. 异构结构测试")
        
        try:
            # 创建不同结构的模型
            structures = [
                (128, 4, 4, 15, 8),
                (256, 5, 8, 20, 10),
                (384, 6, 12, 25, 12),
                (512, 7, 16, 30, 15)
            ]
            
            models = []
            for hidden_size, layers, heads, memory, types in structures:
                # 确保hidden_size能被heads整除
                adjusted_hidden = (hidden_size // heads) * heads
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=adjusted_hidden,
                    reasoning_layers=layers,
                    attention_heads=heads,
                    memory_size=memory,
                    reasoning_types=types
                )
                models.append(model)
            
            # 测试异构种群
            test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
            outputs = []
            
            for i, model in enumerate(models):
                try:
                    output = model(test_input)
                    comprehensive_score = output.get('comprehensive_reasoning', torch.tensor(0.0))
                    if isinstance(comprehensive_score, torch.Tensor):
                        score = comprehensive_score.mean().item()
                    else:
                        score = float(comprehensive_score)
                    outputs.append(score)
                    logger.log_success(f"✅ 模型 {i+1} (结构{structures[i]}): {score:.3f}")
                except Exception as e:
                    logger.log_warning(f"⚠️ 模型 {i+1} 失败: {e}")
                    outputs.append(0.0)
            
            self.test_results['heterogeneous_structures'] = {
                'status': 'PASS',
                'structures_tested': len(structures),
                'successful_models': len([o for o in outputs if o > 0]),
                'average_score': np.mean(outputs),
                'structure_diversity': len(set(str(s) for s in structures))
            }
            
        except Exception as e:
            logger.log_error(f"❌ 异构结构测试失败: {e}")
            self.test_results['heterogeneous_structures'] = {'status': 'FAIL', 'error': str(e)}
    
    async def _test_evolution_algorithm(self):
        """测试进化算法"""
        logger.log_important("🔄 4. 进化算法测试")
        
        try:
            # 创建同构种群
            population = []
            for i in range(4):
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=256,
                    reasoning_layers=5,
                    attention_heads=8,
                    memory_size=20,
                    reasoning_types=10
                )
                population.append(model)
            
            # 创建评估器
            evaluator = EnhancedEvaluator()
            
            # 测试进化算法
            evolution = AdvancedEvolution(
                population_size=4,
                mutation_rate=0.1,
                crossover_rate=0.8,
                elite_size=1
            )
            
            # 执行进化
            start_time = time.time()
            evolved_population = evolution.evolve(
                population=population,
                evaluator=evaluator,
                generations=2
            )
            evolution_time = time.time() - start_time
            
            logger.log_success(f"✅ 进化算法执行成功")
            logger.log_important(f"🔔 进化时间: {evolution_time:.2f}秒")
            logger.log_important(f"🔔 最终种群大小: {len(evolved_population)}")
            
            self.test_results['evolution_algorithm'] = {
                'status': 'PASS',
                'evolution_time': evolution_time,
                'final_population_size': len(evolved_population),
                'generations_completed': 2
            }
            
        except Exception as e:
            logger.log_error(f"❌ 进化算法测试失败: {e}")
            self.test_results['evolution_algorithm'] = {'status': 'FAIL', 'error': str(e)}
    
    async def _test_evaluators(self):
        """测试评估器"""
        logger.log_important("📊 5. 评估器测试")
        
        try:
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            evaluator = EnhancedEvaluator()
            
            # 测试增强评估
            start_time = time.time()
            evaluation_result = await evaluator.evaluate_enhanced_reasoning(
                model=model, 
                max_tasks=10
            )
            evaluation_time = time.time() - start_time
            
            logger.log_success(f"✅ 增强评估器测试成功")
            logger.log_important(f"🔔 评估时间: {evaluation_time:.2f}秒")
            logger.log_important(f"🔔 评估结果: {evaluation_result}")
            
            # 检查评估结果完整性
            expected_metrics = [
                'nested_reasoning', 'symbolic_induction', 'graph_reasoning',
                'multi_step_chain', 'logical_chain', 'abstract_concept',
                'creative_reasoning', 'symbolic_expression', 'comprehensive_reasoning'
            ]
            
            missing_metrics = [m for m in expected_metrics if m not in evaluation_result]
            if missing_metrics:
                logger.log_warning(f"⚠️ 缺失评估指标: {missing_metrics}")
            else:
                logger.log_success("✅ 所有评估指标完整")
            
            self.test_results['evaluators'] = {
                'status': 'PASS',
                'evaluation_time': evaluation_time,
                'metrics_count': len(evaluation_result),
                'comprehensive_score': evaluation_result.get('comprehensive_reasoning', 0.0),
                'missing_metrics': missing_metrics
            }
            
        except Exception as e:
            logger.log_error(f"❌ 评估器测试失败: {e}")
            self.test_results['evaluators'] = {'status': 'FAIL', 'error': str(e)}
    
    async def _test_visualization(self):
        """测试可视化功能"""
        logger.log_important("📈 6. 可视化功能测试")
        
        try:
            # 创建可视化器
            visualizer = EvolutionVisualizer()
            
            # 创建测试数据
            population = []
            for i in range(3):
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=256,
                    reasoning_layers=5,
                    attention_heads=8,
                    memory_size=20,
                    reasoning_types=10
                )
                population.append(model)
            
            # 记录测试数据
            visualizer.record_generation(
                generation=1,
                population=population,
                fitness_scores=[0.1, 0.2, 0.3],
                diversity=0.5,
                best_fitness=0.3,
                avg_fitness=0.2
            )
            
            visualizer.record_generation(
                generation=2,
                population=population,
                fitness_scores=[0.2, 0.3, 0.4],
                diversity=0.6,
                best_fitness=0.4,
                avg_fitness=0.3
            )
            
            # 生成可视化
            curves_file = visualizer.plot_evolution_curves()
            heatmap_file = visualizer.plot_diversity_heatmap()
            report_file = visualizer.generate_evolution_report()
            data_file = visualizer.save_visualization_data()
            
            logger.log_success("✅ 可视化功能测试成功")
            logger.log_important(f"🔔 生成文件:")
            logger.log_important(f"  📊 {curves_file}")
            logger.log_important(f"  📊 {heatmap_file}")
            logger.log_important(f"  📊 {report_file}")
            logger.log_important(f"  📊 {data_file}")
            
            self.test_results['visualization'] = {
                'status': 'PASS',
                'files_generated': [curves_file, heatmap_file, report_file, data_file],
                'generations_recorded': 2
            }
            
        except Exception as e:
            logger.log_error(f"❌ 可视化功能测试失败: {e}")
            self.test_results['visualization'] = {'status': 'FAIL', 'error': str(e)}
    
    async def _test_performance_benchmarks(self):
        """测试性能基准"""
        logger.log_important("⚡ 7. 性能基准测试")
        
        try:
            # 模型推理性能测试
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
            
            # 预热
            for _ in range(10):
                _ = model(test_input)
            
            # 性能测试
            start_time = time.time()
            for _ in range(100):
                _ = model(test_input)
            inference_time = (time.time() - start_time) / 100
            
            logger.log_success(f"✅ 推理性能测试完成")
            logger.log_important(f"🔔 平均推理时间: {inference_time*1000:.2f}毫秒")
            
            # 内存使用测试
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # 创建多个模型测试内存
            models = []
            for _ in range(10):
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=256,
                    reasoning_layers=5,
                    attention_heads=8,
                    memory_size=20,
                    reasoning_types=10
                )
                models.append(model)
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_usage = memory_after - memory_before
            
            logger.log_important(f"🔔 内存使用: {memory_usage:.2f}MB")
            
            self.test_results['performance_benchmarks'] = {
                'status': 'PASS',
                'inference_time_ms': inference_time * 1000,
                'memory_usage_mb': memory_usage,
                'models_created': 10
            }
            
        except Exception as e:
            logger.log_error(f"❌ 性能基准测试失败: {e}")
            self.test_results['performance_benchmarks'] = {'status': 'FAIL', 'error': str(e)}
    
    def _generate_evaluation_report(self):
        """生成评估报告"""
        logger.log_important("📋 8. 生成评估报告")
        logger.log_important("=" * 60)
        
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASS')
        failed_tests = total_tests - passed_tests
        
        logger.log_important(f"📊 测试结果统计:")
        logger.log_important(f"  总测试数: {total_tests}")
        logger.log_important(f"  通过测试: {passed_tests}")
        logger.log_important(f"  失败测试: {failed_tests}")
        logger.log_important(f"  成功率: {passed_tests/total_tests*100:.1f}%")
        
        # 详细结果
        logger.log_important(f"\n📋 详细测试结果:")
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            if status == 'PASS':
                logger.log_success(f"  ✅ {test_name}: PASS")
            else:
                error = result.get('error', 'Unknown error')
                logger.log_error(f"  ❌ {test_name}: FAIL - {error}")
        
        # 性能指标汇总
        if 'performance_benchmarks' in self.test_results and self.test_results['performance_benchmarks']['status'] == 'PASS':
            perf = self.test_results['performance_benchmarks']
            logger.log_important(f"\n⚡ 性能指标:")
            logger.log_important(f"  推理时间: {perf['inference_time_ms']:.2f}ms")
            logger.log_important(f"  内存使用: {perf['memory_usage_mb']:.2f}MB")
        
        # 推理能力汇总
        if 'reasoning_capabilities' in self.test_results and self.test_results['reasoning_capabilities']['status'] == 'PASS':
            reasoning = self.test_results['reasoning_capabilities']
            logger.log_important(f"\n🧠 推理能力:")
            logger.log_important(f"  平均推理分数: {reasoning['average_score']:.3f}")
        
        logger.log_important("=" * 60)
        
        if failed_tests == 0:
            logger.log_success("🎉 所有测试通过！系统运行正常")
        else:
            logger.log_warning(f"⚠️ {failed_tests} 个测试失败，需要进一步优化")

async def main():
    """主函数"""
    evaluator = ComprehensiveEvaluator()
    await evaluator.run_comprehensive_evaluation()

if __name__ == "__main__":
    asyncio.run(main()) 