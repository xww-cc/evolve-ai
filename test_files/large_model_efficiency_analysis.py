#!/usr/bin/env python3
"""
大模型推理效率专项优化与详细分析
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import torch
import torch.nn as nn
import numpy as np
from models.advanced_reasoning_net import AdvancedReasoningNet
from config.optimized_logging import setup_optimized_logging

logger = setup_optimized_logging()

def measure_inference_time(model, input_tensor, repeat=10, use_amp=False):
    times = []
    scores = []
    for _ in range(repeat):
        start = time.time()
        with torch.no_grad():
            if use_amp and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    output = model(input_tensor)
            else:
                output = model(input_tensor)
        elapsed = (time.time() - start) * 1000
        times.append(elapsed)
        # 取推理分数均值
        score = np.mean([v.mean().item() for v in output.values() if isinstance(v, torch.Tensor)])
        scores.append(score)
    return np.mean(times), np.std(times), np.mean(scores)

def get_quantized_model(model):
    try:
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized
    except Exception as e:
        logger.log_warning(f"量化失败: {e}")
        return model

def get_jit_model(model, input_tensor):
    try:
        scripted = torch.jit.trace(model, input_tensor)
        return scripted
    except Exception as e:
        logger.log_warning(f"JIT编译失败: {e}")
        return model

def batch_inference_test(model, batch_sizes=[1, 4, 8, 16]):
    logger.log_important("[批量推理效率测试]")
    for bs in batch_sizes:
        input_tensor = torch.randn(bs, 4)
        t, std, score = measure_inference_time(model, input_tensor, repeat=5)
        logger.log_important(f"batch={bs}: 平均推理时间={t:.2f}ms, 分数={score:.4f}")

def main():
    logger.log_important("=== 大模型推理效率专项优化与详细分析 ===")
    input_tensor = torch.randn(1, 4)
    base_model = AdvancedReasoningNet(hidden_size=512, reasoning_layers=10, attention_heads=8)
    
    # 原始模型
    logger.log_important("[原始模型]")
    t, std, score = measure_inference_time(base_model, input_tensor)
    logger.log_important(f"原始模型: 平均推理时间={t:.2f}ms, 波动={std:.2f}ms, 分数={score:.4f}")
    
    # 量化模型
    logger.log_important("[量化模型]")
    quant_model = get_quantized_model(base_model)
    t, std, score = measure_inference_time(quant_model, input_tensor)
    logger.log_important(f"量化模型: 平均推理时间={t:.2f}ms, 波动={std:.2f}ms, 分数={score:.4f}")
    
    # JIT模型
    logger.log_important("[JIT编译模型]")
    jit_model = get_jit_model(base_model, input_tensor)
    t, std, score = measure_inference_time(jit_model, input_tensor)
    logger.log_important(f"JIT模型: 平均推理时间={t:.2f}ms, 波动={std:.2f}ms, 分数={score:.4f}")
    
    # 混合精度模型（如支持GPU）
    logger.log_important("[混合精度推理]")
    t, std, score = measure_inference_time(base_model, input_tensor, use_amp=True)
    logger.log_important(f"混合精度: 平均推理时间={t:.2f}ms, 波动={std:.2f}ms, 分数={score:.4f}")
    
    # 批量推理效率
    batch_inference_test(base_model)
    
    logger.log_success("大模型推理效率专项分析完成！")

if __name__ == "__main__":
    main() 