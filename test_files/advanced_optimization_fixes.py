#!/usr/bin/env python3
"""
高级优化修复脚本 - 解决量化失败和JIT编译失败问题
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from models.advanced_reasoning_net import AdvancedReasoningNet
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

class OptimizedModel(nn.Module):
    """优化后的模型，解决量化和JIT编译问题"""
    
    def __init__(self, hidden_size=256, reasoning_layers=7, attention_heads=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.reasoning_layers = reasoning_layers
        self.attention_heads = attention_heads
        
        # 简化的推理层，避免复杂操作
        self.input_projection = nn.Linear(4, hidden_size)
        self.reasoning_layers_stack = nn.ModuleList([
            self._create_reasoning_layer() for _ in range(reasoning_layers)
        ])
        self.output_projection = nn.Linear(hidden_size, 13)  # 13种推理类型
        
        # 初始化权重
        self._initialize_weights()
    
    def _create_reasoning_layer(self):
        """创建简化的推理层"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.hidden_size)
        )
    
    def _initialize_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """前向传播，返回简化的输出格式"""
        # 输入投影
        x = self.input_projection(x)
        
        # 推理层处理
        for layer in self.reasoning_layers_stack:
            residual = x
            x = layer(x)
            x = x + residual  # 残差连接
        
        # 输出投影
        output = self.output_projection(x)
        
        # 返回简化的输出格式，避免复杂字典
        return {
            'reasoning_scores': torch.sigmoid(output),
            'confidence': torch.softmax(output, dim=-1).max(dim=-1)[0].unsqueeze(-1)
        }

def create_quantization_friendly_model():
    """创建量化友好的模型"""
    logger.log_important("[创建量化友好模型]")
    
    model = OptimizedModel()
    
    # 确保模型处于评估模式
    model.eval()
    
    # 准备量化
    try:
        # 使用动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model, 
            {nn.Linear}, 
            dtype=torch.qint8
        )
        logger.log_success("动态量化成功")
        return quantized_model
    except Exception as e:
        logger.log_warning(f"动态量化失败: {e}")
        logger.log_important("使用原始模型")
        return model

def create_jit_friendly_model():
    """创建JIT友好的模型"""
    logger.log_important("[创建JIT友好模型]")
    
    model = OptimizedModel()
    model.eval()
    
    # 创建示例输入
    example_input = torch.randn(1, 4)
    
    try:
        # 使用trace而不是script
        traced_model = torch.jit.trace(model, example_input)
        logger.log_success("JIT Trace成功")
        return traced_model
    except Exception as e:
        logger.log_warning(f"JIT Trace失败: {e}")
        logger.log_important("使用原始模型")
        return model

def test_model_performance(model, model_name):
    """测试模型性能"""
    logger.log_important(f"[测试{model_name}性能]")
    
    # 准备测试数据
    test_input = torch.randn(1, 4)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)
    
    # 性能测试
    times = []
    scores = []
    
    with torch.no_grad():
        for _ in range(20):
            import time
            start_time = time.time()
            
            output = model(test_input)
            
            end_time = time.time()
            elapsed = (end_time - start_time) * 1000
            
            times.append(elapsed)
            
            # 计算分数
            if isinstance(output, dict):
                score = output['reasoning_scores'].mean().item()
            else:
                score = output.mean().item()
            scores.append(score)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    avg_score = np.mean(scores)
    
    logger.log_important(f"{model_name}: 平均推理时间={avg_time:.2f}ms, 波动={std_time:.2f}ms, 分数={avg_score:.4f}")
    
    return avg_time, std_time, avg_score

def test_batch_performance(model, model_name):
    """测试批量性能"""
    logger.log_important(f"[测试{model_name}批量性能]")
    
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        test_input = torch.randn(batch_size, 4)
        
        with torch.no_grad():
            import time
            start_time = time.time()
            
            output = model(test_input)
            
            end_time = time.time()
            elapsed = (end_time - start_time) * 1000
            
            # 计算分数
            if isinstance(output, dict):
                score = output['reasoning_scores'].mean().item()
            else:
                score = output.mean().item()
            
            logger.log_important(f"batch={batch_size}: 推理时间={elapsed:.2f}ms, 分数={score:.4f}")

def main():
    logger.log_important("=== 高级优化修复测试 ===")
    
    # 1. 测试原始模型
    original_model = OptimizedModel()
    test_model_performance(original_model, "原始优化模型")
    
    # 2. 测试量化模型
    quantized_model = create_quantization_friendly_model()
    test_model_performance(quantized_model, "量化模型")
    
    # 3. 测试JIT模型
    jit_model = create_jit_friendly_model()
    test_model_performance(jit_model, "JIT模型")
    
    # 4. 测试批量性能
    test_batch_performance(original_model, "原始优化模型")
    
    logger.log_success("高级优化修复测试完成！")

if __name__ == "__main__":
    main() 