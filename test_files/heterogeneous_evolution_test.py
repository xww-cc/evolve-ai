#!/usr/bin/env python3
"""
异构结构进化测试
专门测试异构结构维度不匹配问题的解决方案
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
from evolution.advanced_evolution import AdvancedEvolution
from config.optimized_logging import setup_optimized_logging
import random

logger = setup_optimized_logging()

class HeterogeneousEvolutionTester:
    """异构结构进化测试器"""
    
    def __init__(self):
        self.evaluator = EnhancedEvaluator()
        
    def create_heterogeneous_population(self) -> List[AdvancedReasoningNet]:
        """创建异构种群"""
        logger.log_important("🏗️ 创建异构种群")
        
        # 定义不同的结构配置
        structures = [
            (128, 4, 4, 15, 8),   # 小模型
            (256, 5, 8, 20, 10),  # 中等模型
            (384, 6, 12, 25, 12), # 大模型
            (512, 7, 16, 30, 15)  # 超大模型
        ]
        
        population = []
        for i, (hidden_size, layers, heads, memory, types) in enumerate(structures):
            try:
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
                
                population.append(model)
                logger.log_success(f"✅ 模型 {i+1} 创建成功: hidden_size={adjusted_hidden}, layers={layers}, heads={heads}")
                
            except Exception as e:
                logger.log_error(f"❌ 模型 {i+1} 创建失败: {e}")
        
        logger.log_important(f"📊 异构种群创建完成: {len(population)} 个模型")
        return population
    
    def test_model_forward_pass(self, model: AdvancedReasoningNet, model_name: str) -> bool:
        """测试模型前向传播"""
        try:
            model.eval()
            test_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.float32)
            
            with torch.no_grad():
                output = model(test_input)
            
            # 检查输出键
            expected_keys = ['comprehensive_reasoning', 'symbolic_expression']
            missing_keys = [key for key in expected_keys if key not in output]
            
            if missing_keys:
                logger.log_warning(f"⚠️ {model_name} 缺失输出键: {missing_keys}")
                return False
            else:
                logger.log_success(f"✅ {model_name} 前向传播成功")
                return True
                
        except Exception as e:
            logger.log_error(f"❌ {model_name} 前向传播失败: {e}")
            return False
    
    def test_evolution_operations(self, population: List[AdvancedReasoningNet]) -> bool:
        """测试进化操作"""
        logger.log_important("🔄 测试进化操作")
        
        try:
            # 创建进化算法
            evolution = AdvancedEvolution(
                population_size=4,
                mutation_rate=0.1,
                crossover_rate=0.8,
                elite_size=1
            )
            
            # 测试选择操作
            logger.log_important("🔍 测试选择操作")
            fitness_scores = [0.1, 0.2, 0.3, 0.4]
            selected = evolution._multi_objective_selection(population, fitness_scores)
            logger.log_success(f"✅ 选择操作成功，选择了 {len(selected)} 个个体")
            
            # 测试交叉操作 - 直接调用同步版本
            logger.log_important("🔍 测试交叉操作")
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    parent1 = selected[i]
                    parent2 = selected[i + 1]
                    
                    if random.random() < evolution.adaptive_crossover_rate:
                        # 执行高级交叉
                        child1, child2 = evolution._advanced_parameter_crossover(parent1, parent2)
                        offspring.extend([child1, child2])
                    else:
                        # 直接复制
                        offspring.extend([parent1, parent2])
                else:
                    offspring.append(selected[i])
            
            logger.log_success(f"✅ 交叉操作成功，生成了 {len(offspring)} 个后代")
            
            # 测试变异操作
            logger.log_important("🔍 测试变异操作")
            mutated = evolution._intelligent_mutation(offspring)
            logger.log_success(f"✅ 变异操作成功，处理了 {len(mutated)} 个个体")
            
            # 测试精英保留
            logger.log_important("🔍 测试精英保留")
            new_population = evolution._elitism_with_diversity(population, fitness_scores, mutated)
            logger.log_success(f"✅ 精英保留成功，新种群大小: {len(new_population)}")
            
            return True
            
        except Exception as e:
            logger.log_error(f"❌ 进化操作测试失败: {e}")
            return False
    
    async def test_full_evolution_cycle(self, population: List[AdvancedReasoningNet]) -> bool:
        """测试完整进化周期"""
        logger.log_important("🔄 测试完整进化周期")
        
        try:
            # 创建进化算法
            evolution = AdvancedEvolution(
                population_size=4,
                mutation_rate=0.1,
                crossover_rate=0.8,
                elite_size=1
            )
            
            # 执行进化
            evolved_population = evolution.evolve(
                population=population,
                evaluator=self.evaluator,
                generations=2
            )
            
            logger.log_success(f"✅ 完整进化周期成功，最终种群大小: {len(evolved_population)}")
            return True
            
        except Exception as e:
            logger.log_error(f"❌ 完整进化周期失败: {e}")
            return False
    
    def analyze_population_diversity(self, population: List[AdvancedReasoningNet]) -> Dict[str, Any]:
        """分析种群多样性"""
        logger.log_important("📊 分析种群多样性")
        
        diversity_info = {
            'total_models': len(population),
            'unique_hidden_sizes': set(),
            'unique_layers': set(),
            'unique_heads': set(),
            'unique_memory_sizes': set(),
            'unique_types': set()
        }
        
        for model in population:
            diversity_info['unique_hidden_sizes'].add(model.hidden_size)
            diversity_info['unique_layers'].add(model.reasoning_layers)
            diversity_info['unique_heads'].add(model.attention_heads)
            diversity_info['unique_memory_sizes'].add(model.memory_size)
            diversity_info['unique_types'].add(model.reasoning_types)
        
        # 转换为列表以便JSON序列化
        for key in diversity_info:
            if isinstance(diversity_info[key], set):
                diversity_info[key] = list(diversity_info[key])
        
        logger.log_important(f"📊 多样性统计:")
        logger.log_important(f"  隐藏层大小: {diversity_info['unique_hidden_sizes']}")
        logger.log_important(f"  推理层数: {diversity_info['unique_layers']}")
        logger.log_important(f"  注意力头数: {diversity_info['unique_heads']}")
        logger.log_important(f"  记忆大小: {diversity_info['unique_memory_sizes']}")
        logger.log_important(f"  推理类型: {diversity_info['unique_types']}")
        
        return diversity_info

async def main():
    """主函数"""
    tester = HeterogeneousEvolutionTester()
    
    logger.log_important("🚀 开始异构结构进化测试")
    logger.log_important("=" * 60)
    
    # 1. 创建异构种群
    population = tester.create_heterogeneous_population()
    
    if not population:
        logger.log_error("❌ 无法创建异构种群，测试终止")
        return
    
    # 2. 测试每个模型的前向传播
    logger.log_important("🔍 测试模型前向传播")
    forward_pass_success = 0
    for i, model in enumerate(population):
        if tester.test_model_forward_pass(model, f"模型{i+1}"):
            forward_pass_success += 1
    
    logger.log_important(f"📊 前向传播测试: {forward_pass_success}/{len(population)} 成功")
    
    # 3. 分析种群多样性
    diversity_info = tester.analyze_population_diversity(population)
    
    # 4. 测试进化操作
    evolution_ops_success = tester.test_evolution_operations(population)
    
    # 5. 测试完整进化周期
    full_evolution_success = await tester.test_full_evolution_cycle(population)
    
    # 6. 生成测试报告
    logger.log_important("📋 测试报告")
    logger.log_important("=" * 60)
    
    total_tests = 3
    passed_tests = 0
    
    if forward_pass_success == len(population):
        logger.log_success("✅ 前向传播测试: PASS")
        passed_tests += 1
    else:
        logger.log_error(f"❌ 前向传播测试: FAIL ({forward_pass_success}/{len(population)})")
    
    if evolution_ops_success:
        logger.log_success("✅ 进化操作测试: PASS")
        passed_tests += 1
    else:
        logger.log_error("❌ 进化操作测试: FAIL")
    
    if full_evolution_success:
        logger.log_success("✅ 完整进化周期测试: PASS")
        passed_tests += 1
    else:
        logger.log_error("❌ 完整进化周期测试: FAIL")
    
    logger.log_important(f"📊 总体结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        logger.log_success("🎉 异构结构进化测试全部通过！")
    else:
        logger.log_warning(f"⚠️ {total_tests - passed_tests} 个测试失败，需要进一步优化")

if __name__ == "__main__":
    asyncio.run(main()) 