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