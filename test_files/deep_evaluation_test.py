#!/usr/bin/env python3
"""
深度评估测试
进行更全面的系统测试和评估
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import numpy as np
import torch
import time
import json
from typing import Dict, List, Any, Tuple
from models.advanced_reasoning_net import AdvancedReasoningNet
from evaluators.enhanced_evaluator import EnhancedEvaluator
from evolution.advanced_evolution import AdvancedEvolution, MultiObjectiveAdvancedEvolution
from utils.visualization import EvolutionVisualizer
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

class DeepEvaluator:
    """深度评估器"""
    
    def __init__(self):
        self.evaluator = EnhancedEvaluator()
        self.visualizer = EvolutionVisualizer()
        self.test_results = {}
        
    async def test_model_scalability(self) -> Dict[str, Any]:
        """测试模型可扩展性"""
        logger.log_important("🔍 测试模型可扩展性")
        
        scalability_results = {
            'small_models': [],
            'medium_models': [],
            'large_models': [],
            'performance_metrics': {}
        }
        
        # 测试不同规模的模型
        model_configs = [
            # 小模型
            {'hidden_size': 64, 'reasoning_layers': 2, 'attention_heads': 4, 'memory_size': 10, 'reasoning_types': 5},
            {'hidden_size': 128, 'reasoning_layers': 3, 'attention_heads': 4, 'memory_size': 15, 'reasoning_types': 8},
            # 中等模型
            {'hidden_size': 256, 'reasoning_layers': 4, 'attention_heads': 8, 'memory_size': 20, 'reasoning_types': 10},
            {'hidden_size': 384, 'reasoning_layers': 5, 'attention_heads': 12, 'memory_size': 25, 'reasoning_types': 12},
            # 大模型
            {'hidden_size': 512, 'reasoning_layers': 6, 'attention_heads': 16, 'memory_size': 30, 'reasoning_types': 15},
            {'hidden_size': 768, 'reasoning_layers': 8, 'attention_heads': 24, 'memory_size': 40, 'reasoning_types': 20}
        ]
        
        for i, config in enumerate(model_configs):
            try:
                # 确保hidden_size能被attention_heads整除
                adjusted_hidden = (config['hidden_size'] // config['attention_heads']) * config['attention_heads']
                
                model = AdvancedReasoningNet(
                    input_size=4,
                    hidden_size=adjusted_hidden,
                    reasoning_layers=config['reasoning_layers'],
                    attention_heads=config['attention_heads'],
                    memory_size=config['memory_size'],
                    reasoning_types=config['reasoning_types']
                )
                
                # 测试推理性能
                start_time = time.time()
                test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
                with torch.no_grad():
                    output = model(test_input)
                inference_time = (time.time() - start_time) * 1000  # 毫秒
                
                # 计算模型参数数量
                total_params = sum(p.numel() for p in model.parameters())
                
                # 评估推理能力
                reasoning_score = await self.evaluator.evaluate_enhanced_reasoning(model, max_tasks=5)
                
                result = {
                    'config': config,
                    'adjusted_hidden_size': adjusted_hidden,
                    'total_params': total_params,
                    'inference_time_ms': inference_time,
                    'reasoning_score': reasoning_score.get('comprehensive_reasoning', 0.0),
                    'success': True
                }
                
                if i < 2:
                    scalability_results['small_models'].append(result)
                elif i < 4:
                    scalability_results['medium_models'].append(result)
                else:
                    scalability_results['large_models'].append(result)
                    
                logger.log_success(f"✅ 模型 {i+1} 测试成功: {total_params:,} 参数, {inference_time:.2f}ms")
                
            except Exception as e:
                logger.log_error(f"❌ 模型 {i+1} 测试失败: {e}")
                result = {'config': config, 'success': False, 'error': str(e)}
                if i < 2:
                    scalability_results['small_models'].append(result)
                elif i < 4:
                    scalability_results['medium_models'].append(result)
                else:
                    scalability_results['large_models'].append(result)
        
        # 计算性能指标
        all_successful = [r for r in scalability_results['small_models'] + 
                         scalability_results['medium_models'] + 
                         scalability_results['large_models'] if r.get('success', False)]
        
        if all_successful:
            scalability_results['performance_metrics'] = {
                'avg_inference_time': np.mean([r['inference_time_ms'] for r in all_successful]),
                'avg_reasoning_score': np.mean([r['reasoning_score'] for r in all_successful]),
                'total_params_range': (min([r['total_params'] for r in all_successful]), 
                                     max([r['total_params'] for r in all_successful]))
            }
        
        return scalability_results
    
    def test_evolution_convergence(self) -> Dict[str, Any]:
        """测试进化收敛性"""
        logger.log_important("🔍 测试进化收敛性")
        
        # 创建初始种群
        population = []
        for i in range(8):
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            population.append(model)
        
        # 创建进化算法
        evolution = AdvancedEvolution(
            population_size=8,
            mutation_rate=0.15,
            crossover_rate=0.8,
            elite_size=2
        )
        
        convergence_data = {
            'generations': [],
            'best_fitness': [],
            'avg_fitness': [],
            'diversity_scores': [],
            'convergence_metrics': {}
        }
        
        # 运行多代进化
        for gen in range(10):
            logger.log_important(f"🔄 第 {gen+1} 代进化")
            
            # 计算适应度
            fitness_scores = []
            for model in population:
                score = evolution._calculate_fitness(model, self.evaluator)
                fitness_scores.append(score)
            
            # 计算多样性
            diversity = evolution._calculate_diversity(population)
            
            # 记录数据
            convergence_data['generations'].append(gen + 1)
            convergence_data['best_fitness'].append(max(fitness_scores))
            convergence_data['avg_fitness'].append(np.mean(fitness_scores))
            convergence_data['diversity_scores'].append(diversity)
            
            logger.log_important(f"  最佳适应度: {max(fitness_scores):.4f}")
            logger.log_important(f"  平均适应度: {np.mean(fitness_scores):.4f}")
            logger.log_important(f"  多样性: {diversity:.4f}")
            
            # 进化到下一代
            population = evolution.evolve(population, self.evaluator, generations=1)
        
        # 计算收敛指标
        best_fitness = convergence_data['best_fitness']
        convergence_data['convergence_metrics'] = {
            'final_best_fitness': best_fitness[-1],
            'fitness_improvement': best_fitness[-1] - best_fitness[0],
            'convergence_rate': (best_fitness[-1] - best_fitness[0]) / len(best_fitness),
            'stability': np.std(best_fitness[-3:]) if len(best_fitness) >= 3 else 0
        }
        
        return convergence_data
    
    async def test_multi_objective_optimization(self) -> Dict[str, Any]:
        """测试多目标优化"""
        logger.log_important("🔍 测试多目标优化")
        
        # 创建种群
        population = []
        for i in range(10):
            model = AdvancedReasoningNet(
                input_size=4,
                hidden_size=256,
                reasoning_layers=5,
                attention_heads=8,
                memory_size=20,
                reasoning_types=10
            )
            population.append(model)
        
        # 创建多目标进化算法
        multi_evolution = MultiObjectiveAdvancedEvolution(population_size=10)
        
        # 计算多目标
        objectives = {
            'reasoning_ability': [],
            'efficiency': [],
            'complexity': []
        }
        
        for model in population:
            # 推理能力
            reasoning_score = await self.evaluator.evaluate_enhanced_reasoning(model, max_tasks=5)
            objectives['reasoning_ability'].append(reasoning_score.get('comprehensive_reasoning', 0.0))
            
            # 效率（推理时间）
            start_time = time.time()
            test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
            with torch.no_grad():
                model(test_input)
            inference_time = time.time() - start_time
            objectives['efficiency'].append(1.0 / (1.0 + inference_time * 1000))  # 转换为0-1范围
            
            # 复杂度（参数数量）
            total_params = sum(p.numel() for p in model.parameters())
            objectives['complexity'].append(1.0 / (1.0 + total_params / 1000000))  # 转换为0-1范围
        
        # 运行多目标进化
        evolved_population = await multi_evolution.evolve_multi_objective(population, objectives)
        
        # 评估最终结果
        final_objectives = {
            'reasoning_ability': [],
            'efficiency': [],
            'complexity': []
        }
        
        for model in evolved_population:
            reasoning_score = await self.evaluator.evaluate_enhanced_reasoning(model, max_tasks=5)
            final_objectives['reasoning_ability'].append(reasoning_score.get('comprehensive_reasoning', 0.0))
            
            start_time = time.time()
            test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
            with torch.no_grad():
                model(test_input)
            inference_time = time.time() - start_time
            final_objectives['efficiency'].append(1.0 / (1.0 + inference_time * 1000))
            
            total_params = sum(p.numel() for p in model.parameters())
            final_objectives['complexity'].append(1.0 / (1.0 + total_params / 1000000))
        
        return {
            'initial_objectives': objectives,
            'final_objectives': final_objectives,
            'population_size': len(evolved_population),
            'improvement_metrics': {
                'reasoning_improvement': np.mean(final_objectives['reasoning_ability']) - np.mean(objectives['reasoning_ability']),
                'efficiency_improvement': np.mean(final_objectives['efficiency']) - np.mean(objectives['efficiency']),
                'complexity_improvement': np.mean(final_objectives['complexity']) - np.mean(objectives['complexity'])
            }
        }
    
    def test_robustness_and_stability(self) -> Dict[str, Any]:
        """测试鲁棒性和稳定性"""
        logger.log_important("🔍 测试鲁棒性和稳定性")
        
        robustness_results = {
            'error_handling': [],
            'memory_usage': [],
            'performance_consistency': [],
            'stress_tests': []
        }
        
        # 1. 错误处理测试
        logger.log_important("🔍 错误处理测试")
        try:
            # 测试无效输入
            model = AdvancedReasoningNet()
            invalid_input = torch.tensor([[1, 2, 3]], dtype=torch.float32)  # 维度不匹配
            with torch.no_grad():
                output = model(invalid_input)
            robustness_results['error_handling'].append({
                'test': 'invalid_input_dimension',
                'success': True,
                'message': '正确处理了维度不匹配'
            })
        except Exception as e:
            robustness_results['error_handling'].append({
                'test': 'invalid_input_dimension',
                'success': False,
                'error': str(e)
            })
        
        # 2. 内存使用测试
        logger.log_important("🔍 内存使用测试")
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        models = []
        for i in range(5):
            model = AdvancedReasoningNet()
            models.append(model)
            
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory
            
            robustness_results['memory_usage'].append({
                'model_count': i + 1,
                'memory_mb': current_memory,
                'memory_increase_mb': memory_increase
            })
        
        # 3. 性能一致性测试
        logger.log_important("🔍 性能一致性测试")
        model = AdvancedReasoningNet()
        inference_times = []
        
        for i in range(10):
            start_time = time.time()
            test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
            with torch.no_grad():
                model(test_input)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
        
        robustness_results['performance_consistency'] = {
            'mean_inference_time': np.mean(inference_times),
            'std_inference_time': np.std(inference_times),
            'cv_inference_time': np.std(inference_times) / np.mean(inference_times),  # 变异系数
            'all_times': inference_times
        }
        
        # 4. 压力测试
        logger.log_important("🔍 压力测试")
        try:
            # 连续推理测试
            model = AdvancedReasoningNet()
            start_time = time.time()
            
            for i in range(100):
                test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
                with torch.no_grad():
                    model(test_input)
            
            total_time = time.time() - start_time
            avg_time = total_time / 100 * 1000  # 毫秒
            
            robustness_results['stress_tests'].append({
                'test': 'continuous_inference',
                'iterations': 100,
                'total_time_seconds': total_time,
                'avg_time_ms': avg_time,
                'success': True
            })
            
        except Exception as e:
            robustness_results['stress_tests'].append({
                'test': 'continuous_inference',
                'success': False,
                'error': str(e)
            })
        
        return robustness_results
    
    async def generate_comprehensive_report(self) -> Dict[str, Any]:
        """生成综合评估报告"""
        logger.log_important("📋 生成综合评估报告")
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'test_summary': {},
            'detailed_results': {},
            'recommendations': []
        }
        
        # 运行所有测试
        logger.log_important("🔍 运行可扩展性测试")
        scalability_results = await self.test_model_scalability()
        report['detailed_results']['scalability'] = scalability_results
        
        logger.log_important("🔍 运行收敛性测试")
        convergence_results = self.test_evolution_convergence()
        report['detailed_results']['convergence'] = convergence_results
        
        logger.log_important("🔍 运行多目标优化测试")
        multi_objective_results = await self.test_multi_objective_optimization()
        report['detailed_results']['multi_objective'] = multi_objective_results
        
        logger.log_important("🔍 运行鲁棒性测试")
        robustness_results = self.test_robustness_and_stability()
        report['detailed_results']['robustness'] = robustness_results
        
        # 生成测试总结
        total_tests = 0
        passed_tests = 0
        
        # 可扩展性测试统计
        scalability_success = sum(1 for r in scalability_results['small_models'] + 
                                scalability_results['medium_models'] + 
                                scalability_results['large_models'] if r.get('success', False))
        total_scalability = len(scalability_results['small_models'] + 
                              scalability_results['medium_models'] + 
                              scalability_results['large_models'])
        total_tests += total_scalability
        passed_tests += scalability_success
        
        # 收敛性测试统计
        if convergence_results['convergence_metrics']['final_best_fitness'] > 0:
            passed_tests += 1
        total_tests += 1
        
        # 多目标优化测试统计
        if multi_objective_results['improvement_metrics']['reasoning_improvement'] > 0:
            passed_tests += 1
        total_tests += 1
        
        # 鲁棒性测试统计
        robustness_success = sum(1 for r in robustness_results['error_handling'] if r.get('success', False))
        robustness_success += sum(1 for r in robustness_results['stress_tests'] if r.get('success', False))
        total_robustness = len(robustness_results['error_handling']) + len(robustness_results['stress_tests'])
        total_tests += total_robustness
        passed_tests += robustness_success
        
        report['test_summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'scalability_success_rate': (scalability_success / total_scalability * 100) if total_scalability > 0 else 0,
            'robustness_success_rate': (robustness_success / total_robustness * 100) if total_robustness > 0 else 0
        }
        
        # 生成建议
        if report['test_summary']['success_rate'] < 90:
            report['recommendations'].append("需要进一步优化系统稳定性")
        
        if scalability_results['performance_metrics'].get('avg_inference_time', 0) > 10:
            report['recommendations'].append("推理性能需要优化")
        
        if convergence_results['convergence_metrics']['stability'] > 0.1:
            report['recommendations'].append("进化收敛性需要改进")
        
        return report

async def main():
    """主函数"""
    evaluator = DeepEvaluator()
    
    logger.log_important("🚀 开始深度评估测试")
    logger.log_important("=" * 60)
    
    # 运行深度评估
    report = await evaluator.generate_comprehensive_report()
    
    # 输出结果
    logger.log_important("📋 深度评估报告")
    logger.log_important("=" * 60)
    
    summary = report['test_summary']
    logger.log_important(f"📊 测试总结:")
    logger.log_important(f"  总测试数: {summary['total_tests']}")
    logger.log_important(f"  通过测试: {summary['passed_tests']}")
    logger.log_important(f"  成功率: {summary['success_rate']:.1f}%")
    logger.log_important(f"  可扩展性成功率: {summary['scalability_success_rate']:.1f}%")
    logger.log_important(f"  鲁棒性成功率: {summary['robustness_success_rate']:.1f}%")
    
    if report['recommendations']:
        logger.log_important(f"💡 建议:")
        for rec in report['recommendations']:
            logger.log_important(f"  - {rec}")
    
    # 保存报告
    report_file = f"evaluation_report_{int(time.time())}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.log_important(f"📄 详细报告已保存: {report_file}")
    
    if summary['success_rate'] >= 90:
        logger.log_success("🎉 深度评估测试成功！系统表现优秀")
    elif summary['success_rate'] >= 70:
        logger.log_success("✅ 深度评估测试通过！系统表现良好")
    else:
        logger.log_warning("⚠️ 深度评估测试部分通过，需要进一步优化")

if __name__ == "__main__":
    asyncio.run(main()) 