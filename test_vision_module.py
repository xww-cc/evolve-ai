#!/usr/bin/env python3
"""
视觉模块测试脚本
测试视觉编码器、推理、空间理解和进化功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import time

# 导入视觉模块
from models.vision.vision_encoder import VisionEncoder, VisionAttention, VisionMemory
from models.vision.visual_reasoning import VisualReasoning
from models.vision.spatial_understanding import SpatialUnderstanding
from models.vision.vision_evolution import VisionEvolution
from evaluators.vision_evaluator import VisionEvaluator

def test_vision_encoder():
    """测试视觉编码器"""
    print("🧪 测试视觉编码器...")
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    # 模拟图像数据
    test_images = torch.randn(batch_size, channels, height, width)
    
    # 创建视觉编码器
    vision_encoder = VisionEncoder(
        input_channels=channels,
        hidden_dim=256,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    # 前向传播
    start_time = time.time()
    outputs = vision_encoder(test_images)
    end_time = time.time()
    
    print(f"✅ 视觉编码器测试通过")
    print(f"   输入形状: {test_images.shape}")
    print(f"   输出特征形状: {outputs['features'].shape}")
    print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
    
    return outputs

def test_visual_reasoning():
    """测试视觉推理"""
    print("\n🧪 测试视觉推理...")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    hidden_dim = 256
    
    test_features = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建视觉推理模块
    visual_reasoning = VisualReasoning(
        hidden_dim=hidden_dim,
        num_reasoning_layers=4,
        num_heads=8,
        dropout=0.1
    )
    
    # 前向传播
    start_time = time.time()
    outputs = visual_reasoning(test_features)
    end_time = time.time()
    
    print(f"✅ 视觉推理测试通过")
    print(f"   输入形状: {test_features.shape}")
    print(f"   推理输出形状: {outputs['reasoning_output'].shape}")
    print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
    
    return outputs

def test_spatial_understanding():
    """测试空间理解"""
    print("\n🧪 测试空间理解...")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    hidden_dim = 256
    
    test_features = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建空间理解模块
    spatial_understanding = SpatialUnderstanding(
        hidden_dim=hidden_dim,
        num_spatial_relations=8,
        num_geometric_shapes=10,
        dropout=0.1
    )
    
    # 前向传播
    start_time = time.time()
    outputs = spatial_understanding(test_features)
    end_time = time.time()
    
    print(f"✅ 空间理解测试通过")
    print(f"   输入形状: {test_features.shape}")
    print(f"   空间理解输出形状: {outputs['spatial_understanding'].shape}")
    print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
    
    return outputs

def test_vision_evolution():
    """测试视觉进化"""
    print("\n🧪 测试视觉进化...")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    hidden_dim = 256
    
    test_features = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建视觉进化模块
    vision_evolution = VisionEvolution(
        hidden_dim=hidden_dim,
        evolution_rate=0.01,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # 前向传播
    start_time = time.time()
    outputs = vision_evolution(test_features)
    end_time = time.time()
    
    print(f"✅ 视觉进化测试通过")
    print(f"   输入形状: {test_features.shape}")
    print(f"   进化输出形状: {outputs['encoder_evolution'].shape}")
    print(f"   推理时间: {(end_time - start_time)*1000:.2f}ms")
    
    # 测试进化操作
    population = [torch.randn(seq_len, hidden_dim) for _ in range(5)]
    evolved_population = vision_evolution.evolve(population)
    
    print(f"   种群进化: {len(population)} -> {len(evolved_population)}")
    
    return outputs

def test_vision_evaluator():
    """测试视觉评估器"""
    print("\n🧪 测试视觉评估器...")
    
    # 创建测试数据
    batch_size = 2
    seq_len = 10
    hidden_dim = 256
    
    test_features = torch.randn(batch_size, seq_len, hidden_dim)
    
    # 创建视觉评估器
    vision_evaluator = VisionEvaluator(
        hidden_dim=hidden_dim,
        num_classes=10,
        evaluation_dimensions=5
    )
    
    # 前向传播
    start_time = time.time()
    outputs = vision_evaluator(test_features)
    end_time = time.time()
    
    print(f"✅ 视觉评估器测试通过")
    print(f"   输入形状: {test_features.shape}")
    print(f"   理解评分: {outputs['understanding_score'].mean().item():.4f}")
    print(f"   推理评分: {outputs['reasoning_score'].mean().item():.4f}")
    print(f"   创造评分: {outputs['creation_score'].mean().item():.4f}")
    print(f"   空间评分: {outputs['spatial_score'].mean().item():.4f}")
    print(f"   综合评分: {outputs['comprehensive_score'].mean().item():.4f}")
    print(f"   总体评分: {outputs['overall_score'].mean().item():.4f}")
    print(f"   评估时间: {(end_time - start_time)*1000:.2f}ms")
    
    return outputs

def test_integrated_vision_system():
    """测试集成视觉系统"""
    print("\n🧪 测试集成视觉系统...")
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    test_images = torch.randn(batch_size, channels, height, width)
    
    # 创建视觉系统组件
    vision_encoder = VisionEncoder(hidden_dim=256)
    visual_reasoning = VisualReasoning(hidden_dim=256)
    spatial_understanding = SpatialUnderstanding(hidden_dim=256)
    vision_evolution = VisionEvolution(hidden_dim=256)
    vision_evaluator = VisionEvaluator(hidden_dim=256)
    
    # 集成处理流程
    start_time = time.time()
    
    # 1. 视觉编码
    encoder_outputs = vision_encoder(test_images)
    visual_features = encoder_outputs['features']
    
    # 2. 视觉推理
    reasoning_outputs = visual_reasoning(visual_features)
    
    # 3. 空间理解
    spatial_outputs = spatial_understanding(visual_features)
    
    # 4. 视觉进化
    evolution_outputs = vision_evolution(visual_features)
    
    # 5. 视觉评估
    evaluation_outputs = vision_evaluator(visual_features)
    
    end_time = time.time()
    
    print(f"✅ 集成视觉系统测试通过")
    print(f"   图像输入形状: {test_images.shape}")
    print(f"   视觉特征形状: {visual_features.shape}")
    print(f"   推理输出形状: {reasoning_outputs['reasoning_output'].shape}")
    print(f"   空间理解形状: {spatial_outputs['spatial_understanding'].shape}")
    print(f"   进化输出形状: {evolution_outputs['encoder_evolution'].shape}")
    print(f"   总体评分: {evaluation_outputs['overall_score'].mean().item():.4f}")
    print(f"   总处理时间: {(end_time - start_time)*1000:.2f}ms")
    
    return {
        'encoder_outputs': encoder_outputs,
        'reasoning_outputs': reasoning_outputs,
        'spatial_outputs': spatial_outputs,
        'evolution_outputs': evolution_outputs,
        'evaluation_outputs': evaluation_outputs
    }

def test_vision_performance():
    """测试视觉性能"""
    print("\n🧪 测试视觉性能...")
    
    # 创建不同大小的测试数据
    test_sizes = [
        (1, 3, 224, 224),   # 小批量
        (4, 3, 224, 224),   # 中等批量
        (8, 3, 224, 224),   # 大批量
    ]
    
    vision_encoder = VisionEncoder(hidden_dim=256)
    vision_evaluator = VisionEvaluator(hidden_dim=256)
    
    for batch_size, channels, height, width in test_sizes:
        test_images = torch.randn(batch_size, channels, height, width)
        
        # 测试编码器性能
        start_time = time.time()
        encoder_outputs = vision_encoder(test_images)
        encoder_time = time.time() - start_time
        
        # 测试评估器性能
        start_time = time.time()
        evaluation_outputs = vision_evaluator(encoder_outputs['features'])
        evaluator_time = time.time() - start_time
        
        total_time = encoder_time + evaluator_time
        
        print(f"   批量大小: {batch_size}")
        print(f"   编码时间: {encoder_time*1000:.2f}ms")
        print(f"   评估时间: {evaluator_time*1000:.2f}ms")
        print(f"   总时间: {total_time*1000:.2f}ms")
        print(f"   评分: {evaluation_outputs['overall_score'].mean().item():.4f}")
        print()

def main():
    """主测试函数"""
    print("🚀 开始视觉模块测试")
    print("=" * 50)
    
    try:
        # 测试各个组件
        test_vision_encoder()
        test_visual_reasoning()
        test_spatial_understanding()
        test_vision_evolution()
        test_vision_evaluator()
        
        # 测试集成系统
        test_integrated_vision_system()
        
        # 测试性能
        test_vision_performance()
        
        print("\n🎉 所有视觉模块测试通过!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("=" * 50)
        raise

if __name__ == "__main__":
    main() 