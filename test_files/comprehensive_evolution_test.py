#!/usr/bin/env python3
"""
全面AI自主进化测试 - 验证系统有效性
"""

import asyncio
import time
import statistics
import torch
from evolution.population import create_initial_population
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evolution.nsga2 import evolve_population_nsga2
from config.logging_setup import setup_logging

logger = setup_logging()

class ComprehensiveEvolutionTest:
    """全面进化测试"""
    
    def __init__(self):
        self.test_results = {}
        
    async def run_comprehensive_test(self):
        """运行全面测试"""
        print("🧬 开始全面AI自主进化测试...")
        start_time = time.time()
        
        # 1. 基础功能测试
        await self._test_basic_functionality()
        
        # 2. 进化能力测试
        await self._test_evolution_capability()
        
        # 3. 性能测试
        await self._test_performance()
        
        # 4. 稳定性测试
        await self._test_stability()
        
        # 5. 多样性测试
        await self._test_diversity()
        
        # 6. 生成综合报告
        total_time = time.time() - start_time
        self._generate_comprehensive_report(total_time)
        
        return self._is_system_valid()
    
    async def _test_basic_functionality(self):
        """测试基础功能"""
        print("🔧 测试基础功能...")
        
        try:
            # 种群创建
            population = create_initial_population(10)
            assert len(population) == 10, "种群大小不正确"
            
            # 评估器初始化
            realworld_evaluator = RealWorldEvaluator()
            symbolic_evaluator = SymbolicEvaluator()
            
            # 评估功能
            scores = []
            for individual in population:
                symbolic_score = await symbolic_evaluator.evaluate(individual, level=0)
                realworld_score = await realworld_evaluator.evaluate(individual)
                scores.append((symbolic_score, realworld_score))
                assert 0 <= symbolic_score <= 1, "符号评估分数超出范围"
                assert 0 <= realworld_score <= 1, "真实世界评估分数超出范围"
            
            self.test_results['basic_functionality'] = {
                'status': 'PASS',
                'population_creation': True,
                'evaluator_initialization': True,
                'evaluation_functionality': True,
                'score_range_valid': True
            }
            print("✅ 基础功能测试通过")
            
        except Exception as e:
            self.test_results['basic_functionality'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ 基础功能测试失败: {e}")
    
    async def _test_evolution_capability(self):
        """测试进化能力"""
        print("🔄 测试进化能力...")
        
        try:
            # 创建初始种群
            population = create_initial_population(8)
            
            # 初始评估
            realworld_evaluator = RealWorldEvaluator()
            symbolic_evaluator = SymbolicEvaluator()
            
            initial_scores = []
            for individual in population:
                symbolic_score = await symbolic_evaluator.evaluate(individual, level=0)
                realworld_score = await realworld_evaluator.evaluate(individual)
                initial_scores.append((symbolic_score, realworld_score))
            
            avg_initial_symbolic = statistics.mean(score[0] for score in initial_scores)
            avg_initial_realworld = statistics.mean(score[1] for score in initial_scores)
            
            # 执行进化
            evolved_population, _, _ = await evolve_population_nsga2(population, 3, 0)
            
            # 进化后评估
            evolved_scores = []
            for individual in evolved_population:
                symbolic_score = await symbolic_evaluator.evaluate(individual, level=0)
                realworld_score = await realworld_evaluator.evaluate(individual)
                evolved_scores.append((symbolic_score, realworld_score))
            
            avg_evolved_symbolic = statistics.mean(score[0] for score in evolved_scores)
            avg_evolved_realworld = statistics.mean(score[1] for score in evolved_scores)
            
            # 计算改进
            symbolic_improvement = avg_evolved_symbolic - avg_initial_symbolic
            realworld_improvement = avg_evolved_realworld - avg_initial_realworld
            total_improvement = symbolic_improvement + realworld_improvement
            
            self.test_results['evolution_capability'] = {
                'status': 'PASS' if total_improvement > 0 else 'PARTIAL',
                'initial_symbolic': avg_initial_symbolic,
                'initial_realworld': avg_initial_realworld,
                'evolved_symbolic': avg_evolved_symbolic,
                'evolved_realworld': avg_evolved_realworld,
                'symbolic_improvement': symbolic_improvement,
                'realworld_improvement': realworld_improvement,
                'total_improvement': total_improvement,
                'evolution_effective': total_improvement > 0
            }
            
            print(f"✅ 进化能力测试完成 - 总改进: {total_improvement:+.3f}")
            
        except Exception as e:
            self.test_results['evolution_capability'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ 进化能力测试失败: {e}")
    
    async def _test_performance(self):
        """测试性能"""
        print("⚡ 测试性能...")
        
        try:
            # 种群创建性能
            creation_times = []
            for _ in range(5):
                start_time = time.time()
                population = create_initial_population(10)
                creation_time = time.time() - start_time
                creation_times.append(creation_time)
            
            avg_creation_time = statistics.mean(creation_times)
            
            # 评估性能
            population = create_initial_population(10)
            realworld_evaluator = RealWorldEvaluator()
            symbolic_evaluator = SymbolicEvaluator()
            
            evaluation_times = []
            for individual in population:
                start_time = time.time()
                await symbolic_evaluator.evaluate(individual, level=0)
                await realworld_evaluator.evaluate(individual)
                eval_time = time.time() - start_time
                evaluation_times.append(eval_time)
            
            avg_evaluation_time = statistics.mean(evaluation_times)
            
            # 进化性能
            evolution_times = []
            for _ in range(3):
                population = create_initial_population(8)
                start_time = time.time()
                await evolve_population_nsga2(population, 2, 0)
                evolution_time = time.time() - start_time
                evolution_times.append(evolution_time)
            
            avg_evolution_time = statistics.mean(evolution_times)
            
            self.test_results['performance'] = {
                'status': 'PASS',
                'avg_creation_time': avg_creation_time,
                'avg_evaluation_time': avg_evaluation_time,
                'avg_evolution_time': avg_evolution_time,
                'creation_performance': 'EXCELLENT' if avg_creation_time < 0.1 else 'GOOD',
                'evaluation_performance': 'EXCELLENT' if avg_evaluation_time < 1.0 else 'GOOD',
                'evolution_performance': 'EXCELLENT' if avg_evolution_time < 10.0 else 'GOOD'
            }
            
            print(f"✅ 性能测试完成 - 创建:{avg_creation_time:.3f}s, 评估:{avg_evaluation_time:.3f}s, 进化:{avg_evolution_time:.3f}s")
            
        except Exception as e:
            self.test_results['performance'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ 性能测试失败: {e}")
    
    async def _test_stability(self):
        """测试稳定性"""
        print("🔒 测试稳定性...")
        
        try:
            stability_results = []
            
            for i in range(5):
                try:
                    # 完整流程测试
                    population = create_initial_population(6)
                    realworld_evaluator = RealWorldEvaluator()
                    symbolic_evaluator = SymbolicEvaluator()
                    
                    # 评估
                    for individual in population:
                        await symbolic_evaluator.evaluate(individual, level=0)
                        await realworld_evaluator.evaluate(individual)
                    
                    # 进化
                    await evolve_population_nsga2(population, 2, 0)
                    
                    stability_results.append(True)
                    
                except Exception as e:
                    stability_results.append(False)
            
            success_rate = sum(stability_results) / len(stability_results)
            
            self.test_results['stability'] = {
                'status': 'PASS' if success_rate >= 0.8 else 'PARTIAL',
                'success_rate': success_rate,
                'total_tests': len(stability_results),
                'successful_tests': sum(stability_results)
            }
            
            print(f"✅ 稳定性测试完成 - 成功率: {success_rate*100:.1f}%")
            
        except Exception as e:
            self.test_results['stability'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ 稳定性测试失败: {e}")
    
    async def _test_diversity(self):
        """测试多样性"""
        print("🌐 测试多样性...")
        
        try:
            diversity_metrics = []
            
            for _ in range(5):
                # 初始种群
                population = create_initial_population(10)
                initial_diversity = self._calculate_diversity(population)
                
                # 进化
                evolved_population, _, _ = await evolve_population_nsga2(population, 2, 0)
                final_diversity = self._calculate_diversity(evolved_population)
                
                diversity_metrics.append({
                    'initial': initial_diversity,
                    'final': final_diversity,
                    'change': final_diversity - initial_diversity
                })
            
            avg_initial_diversity = statistics.mean(m['initial'] for m in diversity_metrics)
            avg_final_diversity = statistics.mean(m['final'] for m in diversity_metrics)
            avg_diversity_change = statistics.mean(m['change'] for m in diversity_metrics)
            
            self.test_results['diversity'] = {
                'status': 'PASS' if avg_final_diversity >= 0.5 else 'PARTIAL',
                'avg_initial_diversity': avg_initial_diversity,
                'avg_final_diversity': avg_final_diversity,
                'avg_diversity_change': avg_diversity_change,
                'diversity_maintained': avg_final_diversity >= 0.5
            }
            
            print(f"✅ 多样性测试完成 - 初始:{avg_initial_diversity:.3f}, 最终:{avg_final_diversity:.3f}")
            
        except Exception as e:
            self.test_results['diversity'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            print(f"❌ 多样性测试失败: {e}")
    
    def _calculate_diversity(self, population):
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0
        
        config_hashes = [hash(str(ind.modules_config)) for ind in population]
        unique_configs = len(set(config_hashes))
        return unique_configs / len(population)
    
    def _generate_comprehensive_report(self, total_time):
        """生成综合报告"""
        print(f"\n📊 全面AI自主进化测试报告")
        print(f"=" * 50)
        
        # 统计测试结果
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PASS')
        partial_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'PARTIAL')
        failed_tests = sum(1 for result in self.test_results.values() if result.get('status') == 'FAIL')
        
        print(f"📈 测试统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   通过测试: {passed_tests}")
        print(f"   部分通过: {partial_tests}")
        print(f"   失败测试: {failed_tests}")
        print(f"   成功率: {(passed_tests + partial_tests * 0.5) / total_tests * 100:.1f}%")
        
        # 详细结果
        print(f"\n📋 详细结果:")
        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            status_icon = '✅' if status == 'PASS' else '⚠️' if status == 'PARTIAL' else '❌'
            print(f"   {status_icon} {test_name}: {status}")
            
            if test_name == 'evolution_capability' and 'total_improvement' in result:
                improvement = result['total_improvement']
                print(f"      进化改进: {improvement:+.3f}")
            
            if test_name == 'performance' and 'avg_evolution_time' in result:
                evolution_time = result['avg_evolution_time']
                print(f"      平均进化时间: {evolution_time:.3f}s")
        
        print(f"\n⏱️ 总耗时: {total_time:.2f}秒")
        
        # 系统有效性评估
        system_valid = self._is_system_valid()
        print(f"\n🎯 系统有效性评估:")
        if system_valid:
            print(f"   ✅ AI自主进化系统有效")
            print(f"   🎉 系统可以正常运行和进化")
        else:
            print(f"   ⚠️ AI自主进化系统需要优化")
            print(f"   🔧 建议检查失败的功能模块")
    
    def _is_system_valid(self):
        """判断系统是否有效"""
        # 检查关键功能是否通过
        critical_tests = ['basic_functionality', 'evolution_capability']
        critical_passed = all(
            self.test_results.get(test, {}).get('status') in ['PASS', 'PARTIAL']
            for test in critical_tests
        )
        
        # 检查进化是否有效
        evolution_effective = self.test_results.get('evolution_capability', {}).get('evolution_effective', False)
        
        return critical_passed and evolution_effective

async def main():
    """主函数"""
    test = ComprehensiveEvolutionTest()
    success = await test.run_comprehensive_test()
    
    if success:
        print("\n🎉 AI自主进化系统全面验证成功！")
        print("✅ 系统具备有效的自主进化能力")
    else:
        print("\n⚠️ AI自主进化系统需要进一步优化")
        print("🔧 请检查相关功能模块")

if __name__ == "__main__":
    asyncio.run(main()) 