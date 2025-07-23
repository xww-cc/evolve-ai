#!/usr/bin/env python3
"""
系统核心功能测试脚本
用于测试系统各模块的正确性和完整性
"""

import asyncio
import time
import torch
from typing import Dict, Any, List
from config.logging_setup import setup_logging

# 导入核心模块
from models.modular_net import ModularMathReasoningNet
from evolution.population import create_initial_population
from evolution.nsga2 import evolve_population_nsga2_simple
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evaluators.realworld_evaluator import RealWorldEvaluator
from data.generator import RealWorldDataGenerator

logger = setup_logging()

class SystemCoreTester:
    """系统核心功能测试器"""
    
    def __init__(self):
        self.logger = logger
        self.test_results = {}
        
    async def test_model_creation(self) -> Dict[str, Any]:
        """测试模型创建功能"""
        try:
            self.logger.info("测试模型创建功能...")
            
            # 测试空配置模型
            empty_model = ModularMathReasoningNet(modules_config=[])
            assert isinstance(empty_model, ModularMathReasoningNet)
            
            # 测试带配置模型
            config = [{
                'type': 'linear',
                'input_dim': 10,
                'output_dim': 5,
                'widths': [10, 8, 5],
                'activation_fn_name': 'relu',
                'use_batchnorm': False,
                'module_type': 'linear'
            }]
            configured_model = ModularMathReasoningNet(modules_config=config)
            assert isinstance(configured_model, ModularMathReasoningNet)
            
            # 测试前向传播
            x = torch.randn(3, 4)
            output = empty_model(x)
            assert output.shape[0] == 3
            
            output = configured_model(x)
            assert output.shape[0] == 3
            
            self.logger.info("✅ 模型创建功能测试通过")
            return {"status": "passed", "message": "模型创建功能正常"}
            
        except Exception as e:
            self.logger.error(f"❌ 模型创建功能测试失败: {e}")
            return {"status": "failed", "message": str(e)}
    
    async def test_population_creation(self) -> Dict[str, Any]:
        """测试种群创建功能"""
        try:
            self.logger.info("测试种群创建功能...")
            
            # 测试小种群
            small_population = create_initial_population(5)
            assert len(small_population) == 5
            assert all(isinstance(ind, ModularMathReasoningNet) for ind in small_population)
            
            # 测试大种群
            large_population = create_initial_population(20)
            assert len(large_population) == 20
            assert all(isinstance(ind, ModularMathReasoningNet) for ind in large_population)
            
            # 测试种群多样性
            configs = [str(ind.modules_config) for ind in small_population]
            unique_configs = set(configs)
            diversity_ratio = len(unique_configs) / len(small_population)
            assert diversity_ratio > 0.3  # 至少30%的多样性
            
            self.logger.info("✅ 种群创建功能测试通过")
            return {"status": "passed", "message": "种群创建功能正常"}
            
        except Exception as e:
            self.logger.error(f"❌ 种群创建功能测试失败: {e}")
            return {"status": "failed", "message": str(e)}
    
    async def test_evaluators(self) -> Dict[str, Any]:
        """测试评估器功能"""
        try:
            self.logger.info("测试评估器功能...")
            
            # 创建测试模型
            config = [{
                'type': 'linear',
                'input_dim': 10,
                'output_dim': 5,
                'widths': [10, 8, 5],
                'activation_fn_name': 'relu',
                'use_batchnorm': False,
                'module_type': 'linear'
            }]
            model = ModularMathReasoningNet(modules_config=config)
            
            # 测试符号评估器
            symbolic_evaluator = SymbolicEvaluator()
            symbolic_score = await symbolic_evaluator.evaluate(model, level=0)
            assert isinstance(symbolic_score, float)
            assert 0 <= symbolic_score <= 1
            
            # 测试真实世界评估器
            realworld_evaluator = RealWorldEvaluator()
            realworld_score = await realworld_evaluator.evaluate(model)
            assert isinstance(realworld_score, float)
            assert 0 <= realworld_score <= 1
            
            # 测试任务生成
            tasks = await realworld_evaluator._generate_tasks()
            assert isinstance(tasks, list)
            assert len(tasks) > 0
            
            # 测试任务解决
            for task in tasks:
                score = await realworld_evaluator._solve_task(model, task)
                assert isinstance(score, float)
                assert 0 <= score <= 1
            
            self.logger.info("✅ 评估器功能测试通过")
            return {"status": "passed", "message": "评估器功能正常"}
            
        except Exception as e:
            self.logger.error(f"❌ 评估器功能测试失败: {e}")
            return {"status": "failed", "message": str(e)}
    
    async def test_evolution_algorithm(self) -> Dict[str, Any]:
        """测试进化算法功能"""
        try:
            self.logger.info("测试进化算法功能...")
            
            # 创建测试种群
            population = create_initial_population(10)
            fitness_scores = [(0.8, 0.7)] * len(population)
            
            # 测试进化
            evolved_population = evolve_population_nsga2_simple(
                population,
                fitness_scores,
                mutation_rate=0.8,
                crossover_rate=0.8
            )
            
            assert len(evolved_population) == len(population)
            assert all(isinstance(ind, ModularMathReasoningNet) for ind in evolved_population)
            
            self.logger.info("✅ 进化算法功能测试通过")
            return {"status": "passed", "message": "进化算法功能正常"}
            
        except Exception as e:
            self.logger.error(f"❌ 进化算法功能测试失败: {e}")
            return {"status": "failed", "message": str(e)}
    
    async def test_data_generation(self) -> Dict[str, Any]:
        """测试数据生成功能"""
        try:
            self.logger.info("测试数据生成功能...")
            
            generator = RealWorldDataGenerator()
            
            # 测试测试数据生成
            test_data = generator.generate_test_data(num_samples=10)
            assert isinstance(test_data, dict)
            assert 'x' in test_data
            assert 'y' in test_data
            assert 'num_samples' in test_data
            assert test_data['num_samples'] == 10
            
            # 测试数据形状
            assert test_data['x'].shape[0] == 10
            assert test_data['y'].shape[0] == 10
            
            self.logger.info("✅ 数据生成功能测试通过")
            return {"status": "passed", "message": "数据生成功能正常"}
            
        except Exception as e:
            self.logger.error(f"❌ 数据生成功能测试失败: {e}")
            return {"status": "failed", "message": str(e)}
    
    async def test_system_integration(self) -> Dict[str, Any]:
        """测试系统集成功能"""
        try:
            self.logger.info("测试系统集成功能...")
            
            # 创建完整进化流程
            population = create_initial_population(6)
            
            # 创建评估器
            symbolic_evaluator = SymbolicEvaluator()
            realworld_evaluator = RealWorldEvaluator()
            
            # 评估种群
            fitness_scores = []
            for individual in population:
                symbolic_score = await symbolic_evaluator.evaluate(individual, level=0)
                realworld_score = await realworld_evaluator.evaluate(individual)
                fitness_scores.append((symbolic_score, realworld_score))
            
            # 执行进化
            evolved_population = evolve_population_nsga2_simple(
                population,
                fitness_scores,
                mutation_rate=0.8,
                crossover_rate=0.8
            )
            
            assert len(evolved_population) == len(population)
            
            self.logger.info("✅ 系统集成功能测试通过")
            return {"status": "passed", "message": "系统集成功能正常"}
            
        except Exception as e:
            self.logger.error(f"❌ 系统集成功能测试失败: {e}")
            return {"status": "failed", "message": str(e)}
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """运行所有系统测试"""
        self.logger.info("🧪 开始系统核心功能测试...")
        
        tests = [
            ("模型创建", self.test_model_creation),
            ("种群创建", self.test_population_creation),
            ("评估器", self.test_evaluators),
            ("进化算法", self.test_evolution_algorithm),
            ("数据生成", self.test_data_generation),
            ("系统集成", self.test_system_integration)
        ]
        
        results = {}
        passed = 0
        failed = 0
        
        for test_name, test_func in tests:
            self.logger.info(f"测试: {test_name}")
            result = await test_func()
            results[test_name] = result
            
            if result["status"] == "passed":
                passed += 1
            else:
                failed += 1
        
        # 生成测试报告
        total_tests = len(tests)
        success_rate = (passed / total_tests) * 100
        
        self.logger.info(f"📊 测试结果: {passed}/{total_tests} 通过 ({success_rate:.1f}%)")
        
        if failed == 0:
            self.logger.info("🎉 所有系统核心功能测试通过！")
        else:
            self.logger.warning(f"⚠️  {failed} 个测试失败")
        
        return {
            "total_tests": total_tests,
            "passed": passed,
            "failed": failed,
            "success_rate": success_rate,
            "results": results
        }

async def main():
    """主函数"""
    tester = SystemCoreTester()
    results = await tester.run_all_tests()
    
    print(f"\n📊 系统核心功能测试完成")
    print(f"总测试数: {results['total_tests']}")
    print(f"通过: {results['passed']}")
    print(f"失败: {results['failed']}")
    print(f"成功率: {results['success_rate']:.1f}%")
    
    if results['failed'] == 0:
        print("🎉 系统核心功能完全正常！")
    else:
        print("⚠️  发现系统问题，请检查失败项")

if __name__ == "__main__":
    asyncio.run(main()) 