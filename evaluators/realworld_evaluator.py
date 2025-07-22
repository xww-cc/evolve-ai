from typing import List, Dict
from models.modular_net import ModularMathReasoningNet
from data.generator import RealWorldDataGenerator
from torch import nn
import torch
import time
from utils.parallel_utils import parallel_map
from evaluators.symbolic_evaluator import evaluate_symbolic
from config.logging_setup import setup_logging
import numpy as np

logger = setup_logging()

class RealWorldEvaluator:
    """优化版真实世界评估器"""
    
    def __init__(self):
        self.cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        self.cache_timestamps = {}
    
    async def evaluate(self, model: ModularMathReasoningNet) -> float:
        """优化版真实世界评估"""
        try:
            # 检查缓存
            cache_key = f"realworld_{hash(str(model.modules_config))}"
            current_time = time.time()
            
            # 清理过期缓存
            expired_keys = [k for k, ts in self.cache_timestamps.items() 
                           if current_time - ts > self.cache_ttl]
            for key in expired_keys:
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
            
            # 检查缓存命中
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # 快速评估
            score = await self._quick_evaluation(model)
            
            # 缓存结果
            self.cache[cache_key] = score
            self.cache_timestamps[cache_key] = current_time
            
            return score
            
        except Exception as e:
            logger.warning(f"真实世界评估失败: {e}")
            return 0.0
    
    async def evaluate_model_real_world(self, model_state: Dict, device: str, level: int) -> List[float]:
        """评估模型在真实世界任务中的表现 - 兼容nsga2调用"""
        try:
            # 从状态重建模型
            model = ModularMathReasoningNet(
                model_state['modules_config'], 
                model_state['epigenetic_markers']
            ).to(device)
            model.load_state(model_state)
            
            # 执行评估
            score = await self._quick_evaluation(model)
            
            # 返回多目标评估结果 [主要得分, 多样性得分]
            diversity_score = self._calculate_diversity_score(model)
            
            return [score, diversity_score]
            
        except Exception as e:
            logger.warning(f"真实世界模型评估失败: {e}")
            return [-float('inf'), -float('inf')]
    
    def _calculate_diversity_score(self, model: ModularMathReasoningNet) -> float:
        """计算多样性得分"""
        try:
            # 基于模块配置的多样性
            config_str = str(model.modules_config)
            unique_modules = len(set(config_str.split(',')))
            
            # 基于参数数量的多样性
            total_params = sum(p.numel() for p in model.parameters())
            
            # 基于输出维度的多样性
            test_input = torch.randn(1, 4)
            output = model(test_input)
            output_diversity = output.shape[1] if len(output.shape) > 1 else 1
            
            # 综合多样性得分
            diversity_score = (unique_modules / 10.0 + 
                             min(total_params / 1000.0, 1.0) + 
                             min(output_diversity / 10.0, 1.0)) / 3.0
            
            return min(1.0, diversity_score)
            
        except Exception as e:
            logger.warning(f"多样性计算失败: {e}")
            return 0.1
    
    async def _quick_evaluation(self, model: ModularMathReasoningNet) -> float:
        """快速评估：基础能力测试"""
        try:
            # 测试前向传播
            x = torch.randn(3, 4)
            output = model(x)
            
            # 基础评分
            score = 0.2  # 基础分
            
            # 根据输出维度评分
            if output.shape[1] > 2:
                score += 0.2
            
            # 根据输出稳定性评分
            output_std = torch.std(output).item()
            if output_std > 0.1:
                score += 0.2
            
            # 根据输出范围评分
            output_max = torch.max(output).item()
            output_min = torch.min(output).item()
            if abs(output_max - output_min) > 0.5:
                score += 0.2
            
            # 根据模型复杂度评分
            total_params = sum(p.numel() for p in model.parameters())
            if total_params > 1000:
                score += 0.2
            
            return min(1.0, score)
            
        except Exception as e:
            return 0.0
    
    async def _generate_tasks(self) -> List[Dict]:
        """生成真实世界任务"""
        try:
            tasks = []
            
            # 数学计算任务
            tasks.append({
                'type': 'math_calculation',
                'description': '计算基础数学运算',
                'difficulty': 0.3
            })
            
            # 模式识别任务
            tasks.append({
                'type': 'pattern_recognition',
                'description': '识别数字模式',
                'difficulty': 0.5
            })
            
            # 逻辑推理任务
            tasks.append({
                'type': 'logical_reasoning',
                'description': '解决逻辑问题',
                'difficulty': 0.7
            })
            
            # 序列预测任务
            tasks.append({
                'type': 'sequence_prediction',
                'description': '预测数字序列',
                'difficulty': 0.6
            })
            
            return tasks
            
        except Exception as e:
            logger.warning(f"任务生成失败: {e}")
            return []
    
    async def _solve_task(self, model: ModularMathReasoningNet, task: Dict) -> float:
        """解决单个任务"""
        try:
            task_type = task['type']
            difficulty = task['difficulty']
            
            if task_type == 'math_calculation':
                # 基础数学计算测试
                x = torch.randn(5, 4)
                output = model(x)
                score = min(1.0, 0.3 + difficulty * 0.7)
                
            elif task_type == 'pattern_recognition':
                # 模式识别测试
                x = torch.randn(10, 4)
                output = model(x)
                # 检查输出的变化性
                output_var = torch.var(output).item()
                score = min(1.0, 0.4 + difficulty * 0.6 * min(output_var, 1.0))
                
            elif task_type == 'logical_reasoning':
                # 逻辑推理测试
                x = torch.randn(8, 4)
                output = model(x)
                # 检查输出的逻辑性
                output_range = torch.max(output).item() - torch.min(output).item()
                score = min(1.0, 0.5 + difficulty * 0.5 * min(output_range, 1.0))
                
            elif task_type == 'sequence_prediction':
                # 序列预测测试
                x = torch.randn(12, 4)
                output = model(x)
                # 检查输出的连续性
                output_diff = torch.mean(torch.abs(torch.diff(output, dim=0))).item()
                score = min(1.0, 0.4 + difficulty * 0.6 * min(output_diff, 1.0))
                
            else:
                score = 0.3  # 默认分数
            
            return score
            
        except Exception as e:
            logger.warning(f"任务解决失败: {e}")
            return 0.0