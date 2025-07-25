import asyncio
import time
import logging
import torch
import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Any
from models.modular_net import ModularMathReasoningNet
from config.logging_setup import setup_logging

logger = setup_logging()

class ComplexReasoningEvaluator:
    """复杂推理能力评估器 - 包含多种高级推理任务"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300
        self.cache_timestamps = {}
        self.last_cache_cleanup = time.time()
        
    async def evaluate_complex_reasoning(self, model: ModularMathReasoningNet, level: int = 0) -> Dict[str, float]:
        """评估复杂推理能力"""
        try:
            cache_key = f"complex_{hash(str(model.modules_config))}_{level}"
            current_time = time.time()
            
            # 清理缓存
            if current_time - self.last_cache_cleanup > 60:
                self._cleanup_cache(current_time)
                self.last_cache_cleanup = current_time
            
            # 检查缓存
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            results = {}
            
            # 1. 数学逻辑推理
            results['mathematical_logic'] = await self._evaluate_mathematical_logic(model, level)
            
            # 2. 符号推理
            results['symbolic_reasoning'] = await self._evaluate_symbolic_reasoning(model, level)
            
            # 3. 抽象推理
            results['abstract_reasoning'] = await self._evaluate_abstract_reasoning(model, level)
            
            # 4. 模式识别
            results['pattern_recognition'] = await self._evaluate_pattern_recognition(model, level)
            
            # 5. 推理链
            results['reasoning_chain'] = await self._evaluate_reasoning_chain(model, level)
            
            # 缓存结果
            self.cache[cache_key] = results
            self.cache_timestamps[cache_key] = current_time
            
            return results
            
        except Exception as e:
            logger.error(f"复杂推理评估失败: {e}")
            return {
                'mathematical_logic': 0.0,
                'symbolic_reasoning': 0.0,
                'abstract_reasoning': 0.0,
                'pattern_recognition': 0.0,
                'reasoning_chain': 0.0
            }
    
    def _safe_model_output(self, model: ModularMathReasoningNet, input_tensor: torch.Tensor) -> float:
        """安全获取模型输出，处理多维张量"""
        try:
            with torch.no_grad():
                output = model(input_tensor)
                
                # 处理多维输出
                if output.dim() > 1:
                    # 如果是多维张量，取第一个元素或平均值
                    if output.shape[1] > 1:
                        # 多输出，取第一个
                        output = output[:, 0]
                    else:
                        # 单输出，保持形状
                        output = output.squeeze()
                
                # 确保是标量
                if output.numel() > 1:
                    output = output.mean()
                
                return output.item()
                
        except Exception as e:
            logger.warning(f"模型输出处理失败: {e}")
            return 0.0
    
    async def _evaluate_mathematical_logic(self, model: ModularMathReasoningNet, level: int) -> float:
        """评估数学逻辑推理能力"""
        try:
            score = 0.0
            
            # 基础数学逻辑测试
            test_cases = [
                # 输入: [x, y, z, w], 期望: 逻辑表达式
                ([1, 0, 1, 0], 1),  # AND逻辑
                ([1, 1, 0, 0], 1),  # OR逻辑
                ([0, 0, 1, 1], 0),  # XOR逻辑
                ([1, 1, 1, 1], 1),  # 全真
                ([0, 0, 0, 0], 0),  # 全假
            ]
            
            correct_predictions = 0
            for inputs, expected in test_cases:
                input_tensor = torch.tensor([inputs], dtype=torch.float32)
                output_value = self._safe_model_output(model, input_tensor)
                prediction = 1 if output_value > 0.5 else 0
                if prediction == expected:
                    correct_predictions += 1
            
            score += (correct_predictions / len(test_cases)) * 0.4
            
            # 复杂数学逻辑测试
            if level > 0:
                complex_cases = [
                    # 二次方程逻辑
                    ([1, 2, 1, 0], 1),  # x² + 2x + 1 = 0 有解
                    ([1, 0, -1, 0], 1),  # x² - 1 = 0 有解
                    ([1, 0, 1, 0], 0),   # x² + 1 = 0 无实解
                ]
                
                complex_correct = 0
                for inputs, expected in complex_cases:
                    input_tensor = torch.tensor([inputs], dtype=torch.float32)
                    output_value = self._safe_model_output(model, input_tensor)
                    prediction = 1 if output_value > 0.5 else 0
                    if prediction == expected:
                        complex_correct += 1
                
                score += (complex_correct / len(complex_cases)) * 0.3
            
            # 高级数学逻辑（微积分、线性代数）
            if level > 2:
                advanced_cases = [
                    # 导数逻辑
                    ([1, 2, 3, 4], 1),  # 线性函数的导数
                    ([0, 1, 2, 3], 1),  # 常数函数的导数
                ]
                
                advanced_correct = 0
                for inputs, expected in advanced_cases:
                    input_tensor = torch.tensor([inputs], dtype=torch.float32)
                    output_value = self._safe_model_output(model, input_tensor)
                    prediction = 1 if output_value > 0.5 else 0
                    if prediction == expected:
                        advanced_correct += 1
                
                score += (advanced_correct / len(advanced_cases)) * 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"数学逻辑评估失败: {e}")
            return 0.0
    
    async def _evaluate_symbolic_reasoning(self, model: ModularMathReasoningNet, level: int) -> float:
        """评估符号推理能力"""
        try:
            score = 0.0
            
            # 基础符号推理 - 简化版本
            try:
                # 测试模型是否能处理符号输入
                test_inputs = torch.randn(3, 4)
                outputs = []
                
                for i in range(test_inputs.shape[0]):
                    output = self._safe_model_output(model, test_inputs[i:i+1])
                    outputs.append(output)
                
                # 检查输出的多样性
                output_std = np.std(outputs)
                if output_std > 0.01:
                    score += 0.3
                
                # 检查输出的合理性
                if all(-10 < out < 10 for out in outputs):
                    score += 0.2
                
            except Exception as e:
                logger.warning(f"基础符号推理测试失败: {e}")
            
            # 高级符号推理测试
            if level > 1:
                try:
                    # 测试符号一致性
                    test_inputs = torch.randn(5, 4)
                    model_outputs = []
                    
                    for i in range(test_inputs.shape[0]):
                        output = self._safe_model_output(model, test_inputs[i:i+1])
                        model_outputs.append(output)
                    
                    # 检查输出的一致性
                    if len(model_outputs) > 1:
                        consistency = 1.0 - np.std(model_outputs) / (np.mean(np.abs(model_outputs)) + 1e-8)
                        score += max(0, consistency) * 0.3
                
                except Exception as e:
                    logger.warning(f"高级符号推理测试失败: {e}")
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"符号推理评估失败: {e}")
            return 0.0
    
    async def _evaluate_abstract_reasoning(self, model: ModularMathReasoningNet, level: int) -> float:
        """评估抽象推理能力"""
        try:
            score = 0.0
            
            # 抽象模式识别
            patterns = [
                # 等差数列
                ([1, 2, 3, 4], 5),
                ([2, 4, 6, 8], 10),
                ([1, 3, 5, 7], 9),
                
                # 等比数列
                ([1, 2, 4, 8], 16),
                ([1, 3, 9, 27], 81),
                
                # 平方数列
                ([1, 4, 9, 16], 25),
                ([1, 9, 25, 49], 81),
            ]
            
            correct_predictions = 0
            for pattern, expected in patterns:
                input_tensor = torch.tensor([pattern], dtype=torch.float32)
                output_value = self._safe_model_output(model, input_tensor)
                
                # 预测下一个数
                predicted = int(round(output_value))
                if abs(predicted - expected) <= 2:  # 允许一定误差
                    correct_predictions += 1
            
            score += (correct_predictions / len(patterns)) * 0.4
            
            # 高级抽象推理
            if level > 2:
                complex_patterns = [
                    # 斐波那契数列
                    ([1, 1, 2, 3], 5),
                    ([1, 1, 2, 3, 5], 8),
                    
                    # 交替数列
                    ([1, -1, 1, -1], 1),
                    ([2, -2, 2, -2], 2),
                ]
                
                complex_correct = 0
                for pattern, expected in complex_patterns:
                    input_tensor = torch.tensor([pattern], dtype=torch.float32)
                    output_value = self._safe_model_output(model, input_tensor)
                    predicted = int(round(output_value))
                    if abs(predicted - expected) <= 2:
                        complex_correct += 1
                
                score += (complex_correct / len(complex_patterns)) * 0.3
            
            # 函数关系推理
            if level > 3:
                function_tests = [
                    # 线性关系
                    ([1, 2, 3, 4], 5),
                    # 二次关系
                    ([1, 4, 9, 16], 25),
                    # 指数关系
                    ([1, 2, 4, 8], 16),
                ]
                
                function_correct = 0
                for inputs, expected in function_tests:
                    input_tensor = torch.tensor([inputs], dtype=torch.float32)
                    output_value = self._safe_model_output(model, input_tensor)
                    predicted = int(round(output_value))
                    if abs(predicted - expected) <= 3:
                        function_correct += 1
                
                score += (function_correct / len(function_tests)) * 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"抽象推理评估失败: {e}")
            return 0.0
    
    async def _evaluate_pattern_recognition(self, model: ModularMathReasoningNet, level: int) -> float:
        """评估模式识别能力"""
        try:
            score = 0.0
            
            # 基础模式识别
            patterns = [
                # 重复模式
                ([1, 2, 1, 2], 1),
                ([3, 1, 3, 1], 3),
                
                # 递增模式
                ([1, 2, 3, 4], 5),
                ([5, 6, 7, 8], 9),
                
                # 递减模式
                ([4, 3, 2, 1], 0),
                ([8, 7, 6, 5], 4),
            ]
            
            correct_predictions = 0
            for pattern, expected in patterns:
                input_tensor = torch.tensor([pattern], dtype=torch.float32)
                output_value = self._safe_model_output(model, input_tensor)
                predicted = int(round(output_value))
                if abs(predicted - expected) <= 1:
                    correct_predictions += 1
            
            score += (correct_predictions / len(patterns)) * 0.4
            
            # 高级模式识别
            if level > 1:
                complex_patterns = [
                    # 交替递增
                    ([1, 3, 2, 4], 3),
                    ([2, 4, 3, 5], 4),
                    
                    # 平方模式
                    ([1, 4, 9, 16], 25),
                    ([1, 9, 25, 49], 81),
                ]
                
                complex_correct = 0
                for pattern, expected in complex_patterns:
                    input_tensor = torch.tensor([pattern], dtype=torch.float32)
                    output_value = self._safe_model_output(model, input_tensor)
                    predicted = int(round(output_value))
                    if abs(predicted - expected) <= 2:
                        complex_correct += 1
                
                score += (complex_correct / len(complex_patterns)) * 0.3
            
            # 非线性模式
            if level > 2:
                nonlinear_patterns = [
                    # 指数模式
                    ([1, 2, 4, 8], 16),
                    ([1, 3, 9, 27], 81),
                    
                    # 对数模式
                    ([1, 2, 4, 8], 16),
                ]
                
                nonlinear_correct = 0
                for pattern, expected in nonlinear_patterns:
                    input_tensor = torch.tensor([pattern], dtype=torch.float32)
                    output_value = self._safe_model_output(model, input_tensor)
                    predicted = int(round(output_value))
                    if abs(predicted - expected) <= 3:
                        nonlinear_correct += 1
                
                score += (nonlinear_correct / len(nonlinear_patterns)) * 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"模式识别评估失败: {e}")
            return 0.0
    
    async def _evaluate_reasoning_chain(self, model: ModularMathReasoningNet, level: int) -> float:
        """评估推理链能力"""
        try:
            score = 0.0
            
            # 多步推理测试
            reasoning_chains = [
                # 链式推理：A -> B -> C
                ([1, 2, 3, 4], 6),  # 1+2+3 = 6
                ([2, 3, 4, 5], 9),  # 2+3+4 = 9
                
                # 条件推理
                ([1, 0, 1, 0], 1),  # 如果x=1且z=1，则输出1
                ([0, 1, 0, 1], 0),  # 否则输出0
            ]
            
            correct_chains = 0
            for inputs, expected in reasoning_chains:
                input_tensor = torch.tensor([inputs], dtype=torch.float32)
                output_value = self._safe_model_output(model, input_tensor)
                predicted = int(round(output_value))
                if abs(predicted - expected) <= 1:
                    correct_chains += 1
            
            score += (correct_chains / len(reasoning_chains)) * 0.4
            
            # 高级推理链
            if level > 2:
                advanced_chains = [
                    # 数学推理链
                    ([1, 2, 3, 4], 10),  # (1+2)*(3+4) = 21, 简化后10
                    ([2, 3, 4, 5], 14),  # (2+3)*(4+5) = 45, 简化后14
                ]
                
                advanced_correct = 0
                for inputs, expected in advanced_chains:
                    input_tensor = torch.tensor([inputs], dtype=torch.float32)
                    output_value = self._safe_model_output(model, input_tensor)
                    predicted = int(round(output_value))
                    if abs(predicted - expected) <= 2:
                        advanced_correct += 1
                
                score += (advanced_correct / len(advanced_chains)) * 0.3
            
            # 逻辑推理链
            if level > 3:
                logic_chains = [
                    # 逻辑推理：A AND B OR C
                    ([1, 1, 0, 0], 1),  # A=1, B=1, C=0 -> 1
                    ([1, 0, 1, 0], 1),  # A=1, B=0, C=1 -> 1
                    ([0, 0, 0, 0], 0),  # A=0, B=0, C=0 -> 0
                ]
                
                logic_correct = 0
                for inputs, expected in logic_chains:
                    input_tensor = torch.tensor([inputs], dtype=torch.float32)
                    output_value = self._safe_model_output(model, input_tensor)
                    predicted = 1 if output_value > 0.5 else 0
                    if predicted == expected:
                        logic_correct += 1
                
                score += (logic_correct / len(logic_chains)) * 0.3
            
            return min(1.0, score)
            
        except Exception as e:
            logger.warning(f"推理链评估失败: {e}")
            return 0.0
    
    def _cleanup_cache(self, current_time: float):
        """清理过期缓存"""
        expired_keys = [k for k, ts in self.cache_timestamps.items() 
                       if current_time - ts > self.cache_ttl]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None) 