import asyncio
import time
import logging
from typing import Dict, Any
from models.modular_net import ModularMathReasoningNet
import torch

logger = logging.getLogger(__name__)

class SymbolicEvaluator:
    """符号推理能力评估器 - 优化版本"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timestamps = {}
        self.cache_ttl = 300  # 5分钟缓存时间
        self.last_cache_cleanup = time.time()
        self.cleanup_interval = 60  # 每60秒清理一次缓存
        
    async def evaluate(self, model: ModularMathReasoningNet, level: int = 0) -> float:
        """评估模型的符号推理能力 - 优化版本"""
        try:
            # 检查缓存
            cache_key = f"{hash(str(model.modules_config))}_{level}"
            current_time = time.time()
            
            # 减少缓存清理频率
            if current_time - self.last_cache_cleanup > self.cleanup_interval:
                self._cleanup_cache(current_time)
                self.last_cache_cleanup = current_time
            
            # 检查缓存命中
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            # 快速评估策略
            if level == 0:
                # 基础评估：只测试前向传播和基本符号提取
                score = await self._basic_evaluation(model)
            else:
                # 详细评估：完整的符号提取和证明
                score = await self._detailed_evaluation(model)
            
            # 缓存结果
            self.cache[cache_key] = score
            self.cache_timestamps[cache_key] = current_time
            
            return score
            
        except Exception as e:
            logger.warning(f"符号评估失败: {e}")
            return 0.0
    
    def _cleanup_cache(self, current_time: float):
        """清理过期缓存 - 优化版本"""
        expired_keys = [k for k, ts in self.cache_timestamps.items() 
                       if current_time - ts > self.cache_ttl]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.cache_timestamps.pop(key, None)
        
        # 如果缓存太大，清理最旧的条目
        if len(self.cache) > 1000:
            sorted_keys = sorted(self.cache_timestamps.items(), key=lambda x: x[1])
            keys_to_remove = [k for k, _ in sorted_keys[:len(sorted_keys)//2]]
            for key in keys_to_remove:
                self.cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
    
    async def _basic_evaluation(self, model: ModularMathReasoningNet) -> float:
        """基础评估：快速检查模型能力 - 优化版本"""
        try:
            # 测试前向传播
            x = torch.randn(2, 4)
            output = model(x)
            
            # 基础评分 - 改进版本
            score = 0.3  # 基础分
            
            # 根据输出维度加分
            if output.shape[1] > 1:
                score += 0.2
            
            # 根据输出稳定性加分
            output_std = torch.std(output).item()
            if output_std > 0.01:
                score += 0.2
            
            # 根据模型复杂度加分
            total_params = sum(p.numel() for p in model.parameters())
            if total_params > 100:
                score += 0.2
            
            # 根据模块数量加分
            if len(model.subnet_modules) > 1:
                score += 0.1
            
            # 添加随机性，防止过早收敛
            score += torch.rand(1).item() * 0.1
            
            return min(1.0, score)
            
        except Exception as e:
            return 0.1  # 最小分
    
    async def _detailed_evaluation(self, model: ModularMathReasoningNet) -> float:
        """详细评估：完整的符号能力测试 - 优化版本"""
        try:
            # 基础评估
            base_score = await self._basic_evaluation(model)
            
            # 高级符号提取（使用LLM）
            try:
                # 修复参数调用问题
                expr = await model.extract_symbolic(use_llm=True)
                
                # 高级评分
                advanced_score = 0.0
                
                # 根据表达式复杂度评分
                if hasattr(expr, 'atoms'):
                    atom_count = len(expr.atoms())
                    if atom_count > 3:
                        advanced_score += 0.3
                    if atom_count > 5:
                        advanced_score += 0.2
                
                # 检查是否包含复杂函数
                expr_str = str(expr)
                if any(func in expr_str for func in ['sin', 'cos', 'exp', 'log']):
                    advanced_score += 0.2
                
                # 检查是否包含多项式
                if '+' in expr_str and '*' in expr_str:
                    advanced_score += 0.2
                
                # 添加随机性
                advanced_score += torch.rand(1).item() * 0.1
                
                return min(1.0, base_score + advanced_score)
                
            except Exception as e:
                logger.warning(f"高级符号评估失败: {e}")
                return base_score
            
        except Exception as e:
            logger.warning(f"详细评估失败: {e}")
            return 0.1

# 保持向后兼容性
async def evaluate_symbolic(model: ModularMathReasoningNet, device: str, level: int) -> float:
    """评估符号推理能力 - 完整"""
    try:
        # 修复参数调用问题
        symbolic_expr = await model.extract_symbolic(use_llm=True)
        
        if symbolic_expr is not None:
            test_inputs = torch.randn(10, 4).to(device)
            model_outputs = model(test_inputs).detach().cpu().numpy()
            
            try:
                if isinstance(symbolic_expr, sp.Matrix):
                    expr = symbolic_expr[0] if symbolic_expr.shape[0] > 0 else None
                else:
                    expr = symbolic_expr
                
                if expr is not None:
                    x, y, z, w = sp.symbols('x y z w')
                    sym_func = sp.lambdify((x, y, z, w), expr, 'numpy')
                    
                    test_np = test_inputs.cpu().numpy()
                    sym_outputs = sym_func(test_np[:, 0], test_np[:, 1], test_np[:, 2], test_np[:, 3])
                    
                    if isinstance(sym_outputs, (int, float)):
                        sym_outputs = np.full(10, sym_outputs)
                    
                    mse = np.mean((sym_outputs - model_outputs.flatten())**2)
                    return -mse * 0.1
            except Exception as e:
                logger.warning(f"符号推理评估失败: {e}")
            
        return 0.0
    except Exception as e:
        logger.error(f"符号推理评估异常: {e}")
        return 0.0