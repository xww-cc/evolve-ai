#!/usr/bin/env python3
"""
视觉模块集成测试
测试视觉模块与现有系统的集成
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List

# 导入视觉模块
from models.vision import VisionEncoder, VisualReasoning, SpatialUnderstanding, VisionEvolution
from evaluators.vision_evaluator import VisionEvaluator

def test_vision_module_integration():
    """测试视觉模块集成"""
    print("🧪 测试视觉模块集成...")
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    test_images = torch.randn(batch_size, channels, height, width)
    
    # 创建视觉模块
    vision_encoder = VisionEncoder(hidden_dim=256)
    visual_reasoning = VisualReasoning(hidden_dim=256)
    spatial_understanding = SpatialUnderstanding(hidden_dim=256)
    vision_evolution = VisionEvolution(hidden_dim=256)
    vision_evaluator = VisionEvaluator(hidden_dim=256)
    
    print("✅ 视觉模块创建成功")
    
    # 测试视觉编码
    print("🔍 测试视觉编码...")
    encoder_outputs = vision_encoder(test_images)
    print(f"   编码输出形状: {encoder_outputs['features'].shape}")
    
    # 测试视觉推理
    print("🔍 测试视觉推理...")
    reasoning_outputs = visual_reasoning(encoder_outputs['features'])
    print(f"   推理输出形状: {reasoning_outputs['reasoning_output'].shape}")
    
    # 测试空间理解
    print("🔍 测试空间理解...")
    spatial_outputs = spatial_understanding(encoder_outputs['features'])
    print(f"   空间理解形状: {spatial_outputs['spatial_understanding'].shape}")
    
    # 测试视觉进化
    print("🔍 测试视觉进化...")
    evolution_outputs = vision_evolution(encoder_outputs['features'])
    print(f"   进化输出形状: {evolution_outputs['encoder_evolution'].shape}")
    
    # 测试视觉评估
    print("🔍 测试视觉评估...")
    evaluation_outputs = vision_evaluator(encoder_outputs['features'])
    print(f"   评估输出: {evaluation_outputs['overall_score'].mean().item():.4f}")
    
    print("✅ 所有视觉模块测试通过!")
    
    return {
        'encoder_outputs': encoder_outputs,
        'reasoning_outputs': reasoning_outputs,
        'spatial_outputs': spatial_outputs,
        'evolution_outputs': evolution_outputs,
        'evaluation_outputs': evaluation_outputs
    }

def test_vision_capabilities():
    """测试视觉能力"""
    print("\n🧪 测试视觉能力...")
    
    # 创建测试数据
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    
    test_images = torch.randn(batch_size, channels, height, width)
    
    # 创建视觉编码器
    vision_encoder = VisionEncoder(hidden_dim=256)
    
    # 测试不同批量大小
    for batch_size in [1, 2, 4, 8]:
        test_batch = torch.randn(batch_size, channels, height, width)
        
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        outputs = vision_encoder(test_batch)
        
        if end_time:
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
        else:
            elapsed_time = 0
        
        print(f"   批量大小 {batch_size}: 输出形状 {outputs['features'].shape}, 时间 {elapsed_time:.2f}ms")
    
    print("✅ 视觉能力测试通过!")

def test_vision_evaluation_metrics():
    """测试视觉评估指标"""
    print("\n🧪 测试视觉评估指标...")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    hidden_dim = 256
    
    test_features = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建视觉评估器
    vision_evaluator = VisionEvaluator(hidden_dim=hidden_dim)
    
    # 评估
    evaluation_results = vision_evaluator(test_features)
    
    print("📊 评估指标:")
    print(f"   理解评分: {evaluation_results['understanding_score'].mean().item():.4f}")
    print(f"   推理评分: {evaluation_results['reasoning_score'].mean().item():.4f}")
    print(f"   创造评分: {evaluation_results['creation_score'].mean().item():.4f}")
    print(f"   空间评分: {evaluation_results['spatial_score'].mean().item():.4f}")
    print(f"   综合评分: {evaluation_results['comprehensive_score'].mean().item():.4f}")
    print(f"   总体评分: {evaluation_results['overall_score'].mean().item():.4f}")
    
    print("✅ 视觉评估指标测试通过!")

def main():
    """主测试函数"""
    print("🚀 开始视觉模块集成测试")
    print("=" * 50)
    
    try:
        # 测试视觉模块集成
        test_vision_module_integration()
        
        # 测试视觉能力
        test_vision_capabilities()
        
        # 测试视觉评估指标
        test_vision_evaluation_metrics()
        
        print("\n🎉 所有视觉模块集成测试通过!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("=" * 50)
        raise

if __name__ == "__main__":
    main() 