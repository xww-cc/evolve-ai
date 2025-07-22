#!/usr/bin/env python3
"""
专业模型评估测试报告生成器
"""

import asyncio
import time
import json
import statistics
import numpy as np
from datetime import datetime
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evolution.nsga2 import evolve_population_nsga2
from config.logging_setup import setup_logging
from config.global_constants import POPULATION_SIZE, NUM_GENERATIONS, LEVEL_DESCRIPTIONS

logger = setup_logging('model_evaluation_report.log')

class ModelEvaluationReport:
    """模型评估报告生成器"""
    
    def __init__(self):
        self.report_data = {
            'test_info': {},
            'performance_metrics': {},
            'evaluation_results': {},
            'evolution_analysis': {},
            'diversity_metrics': {},
            'recommendations': {}
        }
        
    async def run_comprehensive_evaluation(self):
        """运行全面评估"""
        print("🔬 开始专业模型评估测试...")
        start_time = time.time()
        
        # 1. 基础性能测试
        await self._test_basic_performance()
        
        # 2. 评估器性能测试
        await self._test_evaluator_performance()
        
        # 3. 进化算法测试
        await self._test_evolution_performance()
        
        # 4. 多样性分析
        await self._test_diversity_analysis()
        
        # 5. 稳定性测试
        await self._test_stability()
        
        # 6. 可扩展性测试
        await self._test_scalability()
        
        total_time = time.time() - start_time
        self.report_data['test_info']['total_test_time'] = total_time
        self.report_data['test_info']['test_date'] = datetime.now().isoformat()
        
        print(f"✅ 评估测试完成 - 总耗时: {total_time:.2f}秒")
        
    async def _test_basic_performance(self):
        """测试基础性能"""
        print("📊 测试基础性能...")
        
        # 种群创建性能
        creation_times = []
        for i in range(5):
            start_time = time.time()
            population = create_initial_population(10)
            creation_time = time.time() - start_time
            creation_times.append(creation_time)
            
        avg_creation_time = statistics.mean(creation_times)
        std_creation_time = statistics.stdev(creation_times)
        
        self.report_data['performance_metrics']['population_creation'] = {
            'average_time': avg_creation_time,
            'std_deviation': std_creation_time,
            'min_time': min(creation_times),
            'max_time': max(creation_times),
            'individuals_per_second': 10 / avg_creation_time
        }
        
        print(f"✅ 种群创建: 平均{avg_creation_time:.3f}s ± {std_creation_time:.3f}s")
        
    async def _test_evaluator_performance(self):
        """测试评估器性能"""
        print("🔧 测试评估器性能...")
        
        population = create_initial_population(15)
        symbolic_evaluator = SymbolicEvaluator()
        realworld_evaluator = RealWorldEvaluator()
        
        # 符号评估性能
        symbolic_times = []
        symbolic_scores = []
        for individual in population:
            start_time = time.time()
            score = await symbolic_evaluator.evaluate(individual)
            eval_time = time.time() - start_time
            symbolic_times.append(eval_time)
            symbolic_scores.append(score)
            
        # 真实世界评估性能
        realworld_times = []
        realworld_scores = []
        for individual in population:
            start_time = time.time()
            score = await realworld_evaluator.evaluate(individual)
            eval_time = time.time() - start_time
            realworld_times.append(eval_time)
            realworld_scores.append(score)
            
        self.report_data['evaluation_results']['symbolic'] = {
            'average_score': statistics.mean(symbolic_scores),
            'std_score': statistics.stdev(symbolic_scores),
            'min_score': min(symbolic_scores),
            'max_score': max(symbolic_scores),
            'average_time': statistics.mean(symbolic_times),
            'std_time': statistics.stdev(symbolic_times),
            'individuals_per_second': len(population) / sum(symbolic_times)
        }
        
        self.report_data['evaluation_results']['realworld'] = {
            'average_score': statistics.mean(realworld_scores),
            'std_score': statistics.stdev(realworld_scores),
            'min_score': min(realworld_scores),
            'max_score': max(realworld_scores),
            'average_time': statistics.mean(realworld_times),
            'std_time': statistics.stdev(realworld_times),
            'individuals_per_second': len(population) / sum(realworld_times)
        }
        
        print(f"✅ 符号评估: 平均得分{statistics.mean(symbolic_scores):.3f}, 速度{len(population)/sum(symbolic_times):.1f}个体/秒")
        print(f"✅ 真实世界评估: 平均得分{statistics.mean(realworld_scores):.3f}, 速度{len(population)/sum(realworld_times):.1f}个体/秒")
        
    async def _test_evolution_performance(self):
        """测试进化算法性能"""
        print("🔄 测试进化算法性能...")
        
        evolution_times = []
        diversity_scores = []
        
        for i in range(3):
            population = create_initial_population(12)
            fitness_scores = [(0.8, 0.6), (0.9, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 0.8),
                             (0.8, 0.7), (0.9, 0.6), (0.7, 0.9), (0.8, 0.8), (0.9, 0.7),
                             (0.8, 0.8), (0.9, 0.6)]
            
            start_time = time.time()
            evolved_population = evolve_population_nsga2(population, fitness_scores)
            evolution_time = time.time() - start_time
            evolution_times.append(evolution_time)
            
            # 计算多样性
            diversity = self._calculate_diversity(evolved_population)
            diversity_scores.append(diversity)
            
        self.report_data['evolution_analysis'] = {
            'average_evolution_time': statistics.mean(evolution_times),
            'std_evolution_time': statistics.stdev(evolution_times),
            'generations_per_second': 1 / statistics.mean(evolution_times),
            'average_diversity': statistics.mean(diversity_scores),
            'std_diversity': statistics.stdev(diversity_scores)
        }
        
        print(f"✅ 进化算法: 平均{statistics.mean(evolution_times):.3f}s, 多样性{statistics.mean(diversity_scores):.3f}")
        
    def _calculate_diversity(self, population):
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
            
        # 简化的多样性计算
        config_hashes = [hash(str(ind.modules_config)) for ind in population]
        unique_configs = len(set(config_hashes))
        return unique_configs / len(population)
        
    async def _test_diversity_analysis(self):
        """测试多样性分析"""
        print("🌊 测试多样性分析...")
        
        diversity_metrics = []
        
        for i in range(5):
            population = create_initial_population(10)
            initial_diversity = self._calculate_diversity(population)
            
            fitness_scores = [(0.8, 0.6)] * len(population)
            evolved_population = evolve_population_nsga2(population, fitness_scores)
            final_diversity = self._calculate_diversity(evolved_population)
            
            diversity_metrics.append({
                'initial': initial_diversity,
                'final': final_diversity,
                'change': final_diversity - initial_diversity
            })
            
        self.report_data['diversity_metrics'] = {
            'average_initial_diversity': statistics.mean([m['initial'] for m in diversity_metrics]),
            'average_final_diversity': statistics.mean([m['final'] for m in diversity_metrics]),
            'average_diversity_change': statistics.mean([m['change'] for m in diversity_metrics]),
            'diversity_maintenance_rate': sum(1 for m in diversity_metrics if m['final'] >= 0.5) / len(diversity_metrics)
        }
        
        print(f"✅ 多样性分析: 初始{statistics.mean([m['initial'] for m in diversity_metrics]):.3f}, 最终{statistics.mean([m['final'] for m in diversity_metrics]):.3f}")
        
    async def _test_stability(self):
        """测试稳定性"""
        print("🔒 测试系统稳定性...")
        
        stability_results = []
        
        for i in range(3):
            try:
                start_time = time.time()
                
                # 完整流程测试
                population = create_initial_population(8)
                symbolic_evaluator = SymbolicEvaluator()
                realworld_evaluator = RealWorldEvaluator()
                
                fitness_scores = []
                for individual in population:
                    symbolic_score = await symbolic_evaluator.evaluate(individual)
                    realworld_score = await realworld_evaluator.evaluate(individual)
                    fitness_scores.append((symbolic_score, realworld_score))
                    
                evolved_population = evolve_population_nsga2(population, fitness_scores)
                
                total_time = time.time() - start_time
                stability_results.append({
                    'success': True,
                    'time': total_time,
                    'population_size': len(evolved_population)
                })
                
            except Exception as e:
                stability_results.append({
                    'success': False,
                    'error': str(e)
                })
                
        success_rate = sum(1 for r in stability_results if r['success']) / len(stability_results)
        avg_time = statistics.mean([r['time'] for r in stability_results if r['success']])
        
        self.report_data['performance_metrics']['stability'] = {
            'success_rate': success_rate,
            'average_execution_time': avg_time,
            'total_tests': len(stability_results)
        }
        
        print(f"✅ 稳定性测试: 成功率{success_rate*100:.1f}%, 平均执行时间{avg_time:.3f}s")
        
    async def _test_scalability(self):
        """测试可扩展性"""
        print("📈 测试可扩展性...")
        
        scalability_results = {}
        
        for population_size in [5, 10, 15, 20]:
            start_time = time.time()
            population = create_initial_population(population_size)
            creation_time = time.time() - start_time
            
            # 评估时间
            evaluator = SymbolicEvaluator()
            eval_start = time.time()
            for individual in population:
                await evaluator.evaluate(individual)
            eval_time = time.time() - eval_start
            
            scalability_results[population_size] = {
                'creation_time': creation_time,
                'evaluation_time': eval_time,
                'total_time': creation_time + eval_time,
                'time_per_individual': (creation_time + eval_time) / population_size
            }
            
        self.report_data['performance_metrics']['scalability'] = scalability_results
        
        print(f"✅ 可扩展性测试: 测试了{len(scalability_results)}种种群大小")
        
    def generate_recommendations(self):
        """生成改进建议"""
        print("💡 生成改进建议...")
        
        recommendations = []
        
        # 基于性能指标的建议
        symbolic_perf = self.report_data['evaluation_results']['symbolic']
        realworld_perf = self.report_data['evaluation_results']['realworld']
        
        if symbolic_perf['average_score'] < 0.8:
            recommendations.append("建议优化符号评估器以提高基础推理能力")
            
        if realworld_perf['average_score'] < 0.7:
            recommendations.append("建议增强真实世界评估的复杂性和多样性")
            
        # 基于多样性的建议
        diversity_metrics = self.report_data['diversity_metrics']
        if diversity_metrics['diversity_maintenance_rate'] < 0.8:
            recommendations.append("建议改进多样性维护机制以防止过早收敛")
            
        # 基于稳定性的建议
        stability = self.report_data['performance_metrics']['stability']
        if stability['success_rate'] < 1.0:
            recommendations.append("建议加强错误处理和异常恢复机制")
            
        # 基于可扩展性的建议
        scalability = self.report_data['performance_metrics']['scalability']
        if len(scalability) > 0:
            largest_pop = max(scalability.keys())
            if scalability[largest_pop]['time_per_individual'] > 0.1:
                recommendations.append("建议优化大规模种群的处理效率")
                
        self.report_data['recommendations'] = recommendations
        
        print(f"✅ 生成了{len(recommendations)}条改进建议")
        
    def save_report(self, filename='model_evaluation_report.json'):
        """保存报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, indent=2, ensure_ascii=False)
        print(f"📄 报告已保存到: {filename}")
        
    def print_summary(self):
        """打印报告摘要"""
        print("\n" + "="*60)
        print("📋 模型评估测试报告摘要")
        print("="*60)
        
        # 测试信息
        test_info = self.report_data['test_info']
        print(f"📅 测试日期: {test_info['test_date']}")
        print(f"⏱️  总测试时间: {test_info['total_test_time']:.2f}秒")
        
        # 性能指标
        perf_metrics = self.report_data['performance_metrics']
        if 'population_creation' in perf_metrics:
            creation = perf_metrics['population_creation']
            print(f"📊 种群创建: {creation['individuals_per_second']:.1f}个体/秒")
            
        if 'stability' in perf_metrics:
            stability = perf_metrics['stability']
            print(f"🔒 稳定性: {stability['success_rate']*100:.1f}%成功率")
            
        # 评估结果
        eval_results = self.report_data['evaluation_results']
        if 'symbolic' in eval_results:
            symbolic = eval_results['symbolic']
            print(f"🧮 符号评估: 平均得分{symbolic['average_score']:.3f} ± {symbolic['std_score']:.3f}")
            
        if 'realworld' in eval_results:
            realworld = eval_results['realworld']
            print(f"🌍 真实世界评估: 平均得分{realworld['average_score']:.3f} ± {realworld['std_score']:.3f}")
            
        # 进化分析
        evolution = self.report_data['evolution_analysis']
        if evolution:
            print(f"🔄 进化性能: {evolution['generations_per_second']:.1f}代/秒")
            print(f"🌊 多样性: {evolution['average_diversity']:.3f} ± {evolution['std_diversity']:.3f}")
            
        # 建议
        recommendations = self.report_data['recommendations']
        if recommendations:
            print(f"\n💡 改进建议 ({len(recommendations)}条):")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
                
        print("="*60)

async def main():
    """主函数"""
    evaluator = ModelEvaluationReport()
    
    # 运行全面评估
    await evaluator.run_comprehensive_evaluation()
    
    # 生成建议
    evaluator.generate_recommendations()
    
    # 保存报告
    evaluator.save_report()
    
    # 打印摘要
    evaluator.print_summary()

if __name__ == "__main__":
    asyncio.run(main()) 