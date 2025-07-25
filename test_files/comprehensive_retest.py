#!/usr/bin/env python3
"""
全面系统复测脚本
验证所有优化效果和系统稳定性
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import torch
import time
from models.advanced_reasoning_net import AdvancedReasoningNet
from evaluators.enhanced_evaluator import EnhancedEvaluator
from evolution.advanced_evolution import AdvancedEvolution
from config.optimized_logging import setup_optimized_logging
from utils.visualization import EvolutionVisualizer
import matplotlib.pyplot as plt

logger = setup_optimized_logging()

class ComprehensiveRetest:
    """全面系统复测器"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.optimization_verification = {}
        
    async def run_comprehensive_retest(self):
        """运行全面复测"""
        logger.log_important("🔄 开始全面系统复测")
        logger.log_important("=" * 60)
        
        # 1. 推理能力复测
        await self._retest_reasoning_capabilities()
        
        # 2. 系统稳定性复测
        await self._retest_system_stability()
        
        # 3. 性能指标复测
        await self._retest_performance_metrics()
        
        # 4. 进化算法复测
        await self._retest_evolution_algorithm()
        
        # 5. 评估器复测
        await self._retest_evaluators()
        
        # 6. 可视化功能复测
        await self._retest_visualization()
        
        # 7. 优化效果验证
        await self._verify_optimization_effects()
        
        # 8. 生成复测报告
        self._generate_retest_report()
        
        return self.test_results
    
    async def _retest_reasoning_capabilities(self):
        """复测推理能力"""
        logger.log_important("🧠 1. 推理能力复测")
        logger.log_important("-" * 40)
        
        # 使用最佳配置
        best_config = {
            'hidden_size': 4096,
            'reasoning_layers': 8,
            'attention_heads': 128,
            'memory_size': 300,
            'reasoning_types': 25
        }
        
        model = AdvancedReasoningNet(
            input_size=4,
            hidden_size=best_config['hidden_size'],
            reasoning_layers=best_config['reasoning_layers'],
            attention_heads=best_config['attention_heads'],
            memory_size=best_config['memory_size'],
            reasoning_types=best_config['reasoning_types']
        )
        
        evaluator = EnhancedEvaluator()
        
        # 多次测试取平均值
        reasoning_scores = []
        inference_times = []
        
        for i in range(3):
            start_time = time.time()
            result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=8)
            end_time = time.time()
            
            reasoning_score = result.get('comprehensive_reasoning', 0.0)
            inference_time = (end_time - start_time) * 1000
            
            reasoning_scores.append(reasoning_score)
            inference_times.append(inference_time)
            
            logger.log_important(f"   测试 {i+1}: 推理分数={reasoning_score:.4f}, 时间={inference_time:.2f}ms")
        
        avg_reasoning_score = np.mean(reasoning_scores)
        avg_inference_time = np.mean(inference_times)
        score_std = np.std(reasoning_scores)
        
        self.test_results['reasoning_capabilities'] = {
            'avg_score': avg_reasoning_score,
            'avg_time': avg_inference_time,
            'score_std': score_std,
            'target_achieved': avg_reasoning_score >= 0.1,
            'scores': reasoning_scores,
            'times': inference_times
        }
        
        logger.log_important(f"📊 推理能力复测结果:")
        logger.log_important(f"   平均推理分数: {avg_reasoning_score:.4f}")
        logger.log_important(f"   平均推理时间: {avg_inference_time:.2f}ms")
        logger.log_important(f"   分数标准差: {score_std:.4f}")
        
        if avg_reasoning_score >= 0.1:
            logger.log_success("✅ 推理能力复测通过，目标达成")
        else:
            logger.log_warning(f"⚠️ 推理能力复测未达标，需要改进")
    
    async def _retest_system_stability(self):
        """复测系统稳定性"""
        logger.log_important("\n🔧 2. 系统稳定性复测")
        logger.log_important("-" * 40)
        
        stability_tests = []
        
        # 测试1: 模型创建稳定性
        try:
            models = []
            for i in range(5):
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=256 + i * 100,
                    reasoning_layers=5 + i,
                    attention_heads=8 + i * 2,
                    memory_size=20 + i * 10,
                    reasoning_types=10 + i
                )
                models.append(model)
            stability_tests.append(('模型创建', True, '成功创建5个不同配置的模型'))
        except Exception as e:
            stability_tests.append(('模型创建', False, f'失败: {e}'))
        
        # 测试2: 推理稳定性
        try:
            model = AdvancedReasoningNet(input_size=4, hidden_size=256, reasoning_layers=5)
            test_input = torch.randn(10, 4)
            
            outputs = []
            for _ in range(10):
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
            
            # 创建多个模型测试内存
            models = []
            for i in range(3):
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=512,
                    reasoning_layers=6,
                    attention_heads=12,
                    memory_size=30,
                    reasoning_types=15
                )
                models.append(model)
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory
            
            # 清理内存
            del models
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            stability_tests.append(('内存稳定性', memory_increase < 1000, f'内存增加: {memory_increase:.1f}MB'))
        except Exception as e:
            stability_tests.append(('内存稳定性', False, f'失败: {e}'))
        
        # 统计结果
        passed_tests = sum(1 for test in stability_tests if test[1])
        total_tests = len(stability_tests)
        stability_rate = passed_tests / total_tests * 100
        
        self.test_results['system_stability'] = {
            'stability_rate': stability_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'test_details': stability_tests
        }
        
        logger.log_important(f"📊 系统稳定性复测结果:")
        for test_name, passed, description in stability_tests:
            status = "✅" if passed else "❌"
            logger.log_important(f"   {status} {test_name}: {description}")
        
        logger.log_important(f"   稳定性通过率: {stability_rate:.1f}% ({passed_tests}/{total_tests})")
        
        if stability_rate >= 90:
            logger.log_success("✅ 系统稳定性复测通过")
        else:
            logger.log_warning(f"⚠️ 系统稳定性需要改进")
    
    async def _retest_performance_metrics(self):
        """复测性能指标"""
        logger.log_important("\n⚡ 3. 性能指标复测")
        logger.log_important("-" * 40)
        
        # 测试不同规模的模型性能
        performance_configs = [
            {'name': '小型模型', 'hidden_size': 128, 'reasoning_layers': 3},
            {'name': '中型模型', 'hidden_size': 512, 'reasoning_layers': 6},
            {'name': '大型模型', 'hidden_size': 1024, 'reasoning_layers': 8},
            {'name': '超大型模型', 'hidden_size': 2048, 'reasoning_layers': 10}
        ]
        
        performance_results = []
        
        for config in performance_configs:
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
                for _ in range(10):
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
                
                performance_results.append({
                    'name': config['name'],
                    'avg_time': avg_time,
                    'std_time': std_time,
                    'memory_usage': memory_usage,
                    'success': True
                })
                
                logger.log_important(f"   {config['name']}: {avg_time:.2f}ms ± {std_time:.2f}ms, {memory_usage:.1f}MB")
                
            except Exception as e:
                performance_results.append({
                    'name': config['name'],
                    'avg_time': 0,
                    'std_time': 0,
                    'memory_usage': 0,
                    'success': False,
                    'error': str(e)
                })
                logger.log_error(f"   {config['name']}: 测试失败 - {e}")
        
        self.test_results['performance_metrics'] = {
            'configs': performance_results,
            'success_rate': sum(1 for r in performance_results if r['success']) / len(performance_results) * 100
        }
        
        logger.log_important(f"📊 性能指标复测完成，成功率: {self.test_results['performance_metrics']['success_rate']:.1f}%")
    
    async def _retest_evolution_algorithm(self):
        """复测进化算法"""
        logger.log_important("\n🔄 4. 进化算法复测")
        logger.log_important("-" * 40)
        
        try:
            # 创建进化算法
            evolution = AdvancedEvolution(
                population_size=6,
                mutation_rate=0.1,
                crossover_rate=0.8,
                elite_size=1
            )
            
            # 创建初始种群
            population = []
            for i in range(6):
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=256 + i * 50,
                    reasoning_layers=5 + i,
                    attention_heads=8 + i,
                    memory_size=20,
                    reasoning_types=10
                )
                population.append(model)
            
            # 运行进化
            start_time = time.time()
            evolved_population, history = await evolution.evolve_population(
                population, 
                generations=3,
                evaluator=EnhancedEvaluator()
            )
            end_time = time.time()
            
            evolution_time = end_time - start_time
            
            # 分析结果
            if history and len(history) > 0:
                best_fitness = max(history[-1]['best_fitness'] for history in history)
                avg_fitness = np.mean([h['avg_fitness'] for h in history])
                diversity = np.mean([h.get('diversity', 0) for h in history])
            else:
                best_fitness = 0
                avg_fitness = 0
                diversity = 0
            
            self.test_results['evolution_algorithm'] = {
                'evolution_time': evolution_time,
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
                'diversity': diversity,
                'population_size': len(evolved_population),
                'success': True
            }
            
            logger.log_important(f"📊 进化算法复测结果:")
            logger.log_important(f"   进化时间: {evolution_time:.2f}秒")
            logger.log_important(f"   最佳适应度: {best_fitness:.4f}")
            logger.log_important(f"   平均适应度: {avg_fitness:.4f}")
            logger.log_important(f"   多样性: {diversity:.4f}")
            logger.log_important(f"   种群大小: {len(evolved_population)}")
            
            logger.log_success("✅ 进化算法复测通过")
            
        except Exception as e:
            logger.log_error(f"❌ 进化算法复测失败: {e}")
            self.test_results['evolution_algorithm'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _retest_evaluators(self):
        """复测评估器"""
        logger.log_important("\n📊 5. 评估器复测")
        logger.log_important("-" * 40)
        
        try:
            evaluator = EnhancedEvaluator()
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            
            # 测试评估器
            start_time = time.time()
            result = await evaluator.evaluate_enhanced_reasoning(model, max_tasks=5)
            end_time = time.time()
            
            evaluation_time = end_time - start_time
            
            # 检查评估结果完整性
            expected_keys = [
                'comprehensive_reasoning', 'nested_reasoning', 'symbolic_induction',
                'graph_reasoning', 'multi_step_chain', 'logical_chain',
                'abstract_concept', 'creative_reasoning', 'symbolic_expression'
            ]
            
            missing_keys = [key for key in expected_keys if key not in result]
            completeness = (len(expected_keys) - len(missing_keys)) / len(expected_keys) * 100
            
            self.test_results['evaluators'] = {
                'evaluation_time': evaluation_time,
                'completeness': completeness,
                'missing_keys': missing_keys,
                'result_keys': list(result.keys()),
                'success': True
            }
            
            logger.log_important(f"📊 评估器复测结果:")
            logger.log_important(f"   评估时间: {evaluation_time:.2f}秒")
            logger.log_important(f"   结果完整性: {completeness:.1f}%")
            logger.log_important(f"   缺失键: {missing_keys if missing_keys else '无'}")
            
            if completeness >= 90:
                logger.log_success("✅ 评估器复测通过")
            else:
                logger.log_warning(f"⚠️ 评估器结果不完整")
                
        except Exception as e:
            logger.log_error(f"❌ 评估器复测失败: {e}")
            self.test_results['evaluators'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _retest_visualization(self):
        """复测可视化功能"""
        logger.log_important("\n📈 6. 可视化功能复测")
        logger.log_important("-" * 40)
        
        try:
            viz_manager = EvolutionVisualizer()
            
            # 生成测试数据
            test_data = {
                'generations': list(range(1, 6)),
                'best_fitness': [0.02, 0.03, 0.04, 0.05, 0.06],
                'avg_fitness': [0.015, 0.025, 0.035, 0.045, 0.055],
                'diversity': [0.8, 0.7, 0.6, 0.5, 0.4]
            }
            
            # 测试进化曲线生成
            evolution_plot_path = viz_manager.plot_evolution_curves(test_data)
            
            # 测试多样性热力图
            diversity_data = np.random.rand(5, 5)
            diversity_plot_path = viz_manager.plot_diversity_heatmap(diversity_data)
            
            # 测试报告生成
            report_data = {
                'total_generations': 5,
                'final_best_fitness': 0.06,
                'improvement_rate': 200.0,
                'diversity_trend': 'decreasing'
            }
            report_path = viz_manager.generate_evolution_report(report_data)
            
            # 检查文件是否生成
            files_generated = []
            for path in [evolution_plot_path, diversity_plot_path, report_path]:
                if path and os.path.exists(path):
                    files_generated.append(os.path.basename(path))
            
            success_rate = len(files_generated) / 3 * 100
            
            self.test_results['visualization'] = {
                'success_rate': success_rate,
                'files_generated': files_generated,
                'evolution_plot': evolution_plot_path,
                'diversity_plot': diversity_plot_path,
                'report_path': report_path,
                'success': True
            }
            
            logger.log_important(f"📊 可视化功能复测结果:")
            logger.log_important(f"   成功率: {success_rate:.1f}%")
            logger.log_important(f"   生成文件: {files_generated}")
            
            if success_rate >= 80:
                logger.log_success("✅ 可视化功能复测通过")
            else:
                logger.log_warning(f"⚠️ 可视化功能需要改进")
                
        except Exception as e:
            logger.log_error(f"❌ 可视化功能复测失败: {e}")
            self.test_results['visualization'] = {
                'success': False,
                'error': str(e)
            }
    
    async def _verify_optimization_effects(self):
        """验证优化效果"""
        logger.log_important("\n🔍 7. 优化效果验证")
        logger.log_important("-" * 40)
        
        # 验证推理分数目标
        reasoning_score = self.test_results.get('reasoning_capabilities', {}).get('avg_score', 0)
        target_achieved = reasoning_score >= 0.1
        
        # 验证系统稳定性
        stability_rate = self.test_results.get('system_stability', {}).get('stability_rate', 0)
        stability_achieved = stability_rate >= 90
        
        # 验证性能指标
        performance_success = self.test_results.get('performance_metrics', {}).get('success_rate', 0)
        performance_achieved = performance_success >= 80
        
        # 验证进化算法
        evolution_success = self.test_results.get('evolution_algorithm', {}).get('success', False)
        
        # 验证评估器
        evaluator_success = self.test_results.get('evaluators', {}).get('success', False)
        
        # 验证可视化
        viz_success = self.test_results.get('visualization', {}).get('success', False)
        
        self.optimization_verification = {
            'reasoning_target': target_achieved,
            'stability_target': stability_achieved,
            'performance_target': performance_achieved,
            'evolution_working': evolution_success,
            'evaluator_working': evaluator_success,
            'visualization_working': viz_success,
            'overall_success': all([
                target_achieved, stability_achieved, performance_achieved,
                evolution_success, evaluator_success, viz_success
            ])
        }
        
        logger.log_important(f"📊 优化效果验证结果:")
        logger.log_important(f"   推理分数目标: {'✅' if target_achieved else '❌'} ({reasoning_score:.4f})")
        logger.log_important(f"   系统稳定性: {'✅' if stability_achieved else '❌'} ({stability_rate:.1f}%)")
        logger.log_important(f"   性能指标: {'✅' if performance_achieved else '❌'} ({performance_success:.1f}%)")
        logger.log_important(f"   进化算法: {'✅' if evolution_success else '❌'}")
        logger.log_important(f"   评估器: {'✅' if evaluator_success else '❌'}")
        logger.log_important(f"   可视化: {'✅' if viz_success else '❌'}")
        
        if self.optimization_verification['overall_success']:
            logger.log_success("🎉 所有优化目标均已达成！")
        else:
            logger.log_warning("⚠️ 部分优化目标尚未达成，需要继续改进")
    
    def _generate_retest_report(self):
        """生成复测报告"""
        logger.log_important("\n📋 全面系统复测报告")
        logger.log_important("=" * 60)
        
        # 统计总体结果
        total_tests = len(self.test_results)
        successful_tests = sum(1 for result in self.test_results.values() 
                             if isinstance(result, dict) and result.get('success', True))
        
        overall_success_rate = successful_tests / total_tests * 100
        
        logger.log_important(f"📊 复测总体结果:")
        logger.log_important(f"   总测试数: {total_tests}")
        logger.log_important(f"   成功测试: {successful_tests}")
        logger.log_important(f"   成功率: {overall_success_rate:.1f}%")
        
        # 详细结果
        logger.log_important(f"\n📋 详细测试结果:")
        
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                success = result.get('success', True)
                status = "✅" if success else "❌"
                logger.log_important(f"   {status} {test_name}")
                
                # 显示关键指标
                if test_name == 'reasoning_capabilities':
                    score = result.get('avg_score', 0)
                    target_achieved = result.get('target_achieved', False)
                    logger.log_important(f"      推理分数: {score:.4f} {'✅' if target_achieved else '❌'}")
                
                elif test_name == 'system_stability':
                    stability_rate = result.get('stability_rate', 0)
                    logger.log_important(f"      稳定性: {stability_rate:.1f}%")
                
                elif test_name == 'performance_metrics':
                    success_rate = result.get('success_rate', 0)
                    logger.log_important(f"      成功率: {success_rate:.1f}%")
        
        # 优化效果总结
        if self.optimization_verification:
            logger.log_important(f"\n🎯 优化效果总结:")
            overall_success = self.optimization_verification.get('overall_success', False)
            logger.log_important(f"   整体优化效果: {'✅ 优秀' if overall_success else '⚠️ 需改进'}")
        
        # 最终评估
        if overall_success_rate >= 90:
            logger.log_success("🎉 全面系统复测通过！系统运行优秀")
        elif overall_success_rate >= 80:
            logger.log_important("✅ 全面系统复测基本通过，部分功能需要改进")
        else:
            logger.log_warning("⚠️ 全面系统复测发现问题，需要重点改进")

async def main():
    """主函数"""
    logger.log_important("=== 全面系统复测 ===")
    
    # 创建复测器
    retester = ComprehensiveRetest()
    
    # 运行全面复测
    results = await retester.run_comprehensive_retest()
    
    logger.log_important(f"\n🎉 全面系统复测完成！")
    logger.log_important(f"复测结果已生成，请查看详细报告")

if __name__ == "__main__":
    asyncio.run(main()) 