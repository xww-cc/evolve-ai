#!/usr/bin/env python3
"""
视觉模块全面集成测试
测试视觉模块的基本功能
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import time

# 导入视觉模块
from models.vision import VisionEncoder, VisualReasoning, SpatialUnderstanding, VisionEvolution
from evaluators.vision_evaluator import VisionEvaluator

def test_vision_module_components():
    """测试视觉模块各组件"""
    print("🧪 测试视觉模块组件...")
    
    # 创建测试数据
    batch_size = 2
    channels = 3
    height = 224
    width = 224
    
    test_images = torch.randn(batch_size, channels, height, width)
    
    # 测试视觉编码器
    print("🔍 测试视觉编码器...")
    vision_encoder = VisionEncoder(hidden_dim=256)
    encoder_outputs = vision_encoder(test_images)
    print(f"   编码输出形状: {encoder_outputs['features'].shape}")
    print(f"   注意力权重形状: {encoder_outputs['attention_weights'].shape}")
    
    # 测试视觉推理
    print("🔍 测试视觉推理...")
    visual_reasoning = VisualReasoning(hidden_dim=256)
    reasoning_outputs = visual_reasoning(encoder_outputs['features'])
    print(f"   推理输出形状: {reasoning_outputs['reasoning_output'].shape}")
    print(f"   空间输出形状: {reasoning_outputs['spatial_output'].shape}")
    print(f"   逻辑输出形状: {reasoning_outputs['logical_output'].shape}")
    print(f"   因果输出形状: {reasoning_outputs['causal_output'].shape}")
    print(f"   抽象输出形状: {reasoning_outputs['abstract_output'].shape}")
    
    # 测试空间理解
    print("🔍 测试空间理解...")
    spatial_understanding = SpatialUnderstanding(hidden_dim=256)
    spatial_outputs = spatial_understanding(encoder_outputs['features'])
    print(f"   空间理解形状: {spatial_outputs['spatial_understanding'].shape}")
    print(f"   空间关系形状: {spatial_outputs['spatial_relations'].shape}")
    print(f"   几何推理形状: {spatial_outputs['geometric_reasoning'].shape}")
    
    # 测试视觉进化
    print("🔍 测试视觉进化...")
    vision_evolution = VisionEvolution(hidden_dim=256)
    evolution_outputs = vision_evolution(encoder_outputs['features'])
    print(f"   编码器进化形状: {evolution_outputs['encoder_evolution'].shape}")
    print(f"   推理进化形状: {evolution_outputs['reasoning_evolution'].shape}")
    print(f"   空间进化形状: {evolution_outputs['spatial_evolution'].shape}")
    
    # 测试视觉评估
    print("🔍 测试视觉评估...")
    vision_evaluator = VisionEvaluator(hidden_dim=256)
    evaluation_outputs = vision_evaluator(encoder_outputs['features'])
    print(f"   理解评分: {evaluation_outputs['understanding_score'].mean().item():.4f}")
    print(f"   推理评分: {evaluation_outputs['reasoning_score'].mean().item():.4f}")
    print(f"   创造评分: {evaluation_outputs['creation_score'].mean().item():.4f}")
    print(f"   空间评分: {evaluation_outputs['spatial_score'].mean().item():.4f}")
    print(f"   综合评分: {evaluation_outputs['comprehensive_score'].mean().item():.4f}")
    print(f"   总体评分: {evaluation_outputs['overall_score'].mean().item():.4f}")
    
    print("✅ 视觉模块组件测试通过!")
    
    return {
        'encoder_outputs': encoder_outputs,
        'reasoning_outputs': reasoning_outputs,
        'spatial_outputs': spatial_outputs,
        'evolution_outputs': evolution_outputs,
        'evaluation_outputs': evaluation_outputs
    }

def test_vision_performance():
    """测试视觉模块性能"""
    print("\n🧪 测试视觉模块性能...")
    
    # 创建视觉模块
    vision_encoder = VisionEncoder(hidden_dim=256)
    visual_reasoning = VisualReasoning(hidden_dim=256)
    spatial_understanding = SpatialUnderstanding(hidden_dim=256)
    vision_evolution = VisionEvolution(hidden_dim=256)
    vision_evaluator = VisionEvaluator(hidden_dim=256)
    
    # 测试不同批量大小
    batch_sizes = [1, 2, 4, 8]
    channels, height, width = 3, 224, 224
    
    performance_results = {}
    
    for batch_size in batch_sizes:
        print(f"\n   测试批量大小 {batch_size}:")
        
        test_images = torch.randn(batch_size, channels, height, width)
        
        # 测试编码器性能
        start_time = time.time()
        encoder_outputs = vision_encoder(test_images)
        encoder_time = (time.time() - start_time) * 1000
        
        # 测试推理性能
        start_time = time.time()
        reasoning_outputs = visual_reasoning(encoder_outputs['features'])
        reasoning_time = (time.time() - start_time) * 1000
        
        # 测试空间理解性能
        start_time = time.time()
        spatial_outputs = spatial_understanding(encoder_outputs['features'])
        spatial_time = (time.time() - start_time) * 1000
        
        # 测试进化性能
        start_time = time.time()
        evolution_outputs = vision_evolution(encoder_outputs['features'])
        evolution_time = (time.time() - start_time) * 1000
        
        # 测试评估性能
        start_time = time.time()
        evaluation_outputs = vision_evaluator(encoder_outputs['features'])
        evaluation_time = (time.time() - start_time) * 1000
        
        total_time = encoder_time + reasoning_time + spatial_time + evolution_time + evaluation_time
        
        print(f"     编码器: {encoder_time:.2f}ms")
        print(f"     推理: {reasoning_time:.2f}ms")
        print(f"     空间理解: {spatial_time:.2f}ms")
        print(f"     进化: {evolution_time:.2f}ms")
        print(f"     评估: {evaluation_time:.2f}ms")
        print(f"     总时间: {total_time:.2f}ms")
        
        performance_results[batch_size] = {
            'encoder_time': encoder_time,
            'reasoning_time': reasoning_time,
            'spatial_time': spatial_time,
            'evolution_time': evolution_time,
            'evaluation_time': evaluation_time,
            'total_time': total_time
        }
    
    print("✅ 视觉模块性能测试通过!")
    
    return performance_results

def test_vision_memory_usage():
    """测试视觉模块内存使用"""
    print("\n🧪 测试视觉模块内存使用...")
    
    try:
        import psutil
        import gc
        
        # 获取初始内存使用
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 创建视觉模块
        vision_encoder = VisionEncoder(hidden_dim=256)
        visual_reasoning = VisualReasoning(hidden_dim=256)
        spatial_understanding = SpatialUnderstanding(hidden_dim=256)
        vision_evolution = VisionEvolution(hidden_dim=256)
        vision_evaluator = VisionEvaluator(hidden_dim=256)
        
        # 测试内存使用
        test_images = torch.randn(4, 3, 224, 224)
        
        # 执行视觉处理
        encoder_outputs = vision_encoder(test_images)
        reasoning_outputs = visual_reasoning(encoder_outputs['features'])
        spatial_outputs = spatial_understanding(encoder_outputs['features'])
        evolution_outputs = vision_evolution(encoder_outputs['features'])
        evaluation_outputs = vision_evaluator(encoder_outputs['features'])
        
        # 获取最终内存使用
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"   初始内存: {initial_memory:.2f} MB")
        print(f"   最终内存: {final_memory:.2f} MB")
        print(f"   内存增加: {memory_increase:.2f} MB")
        
        # 清理内存
        del encoder_outputs, reasoning_outputs, spatial_outputs, evolution_outputs, evaluation_outputs
        gc.collect()
        
        # 获取清理后内存
        cleaned_memory = process.memory_info().rss / 1024 / 1024  # MB
        print(f"   清理后内存: {cleaned_memory:.2f} MB")
        
        print("✅ 视觉模块内存使用测试通过!")
        
        return {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'memory_increase': memory_increase,
            'cleaned_memory': cleaned_memory
        }
        
    except ImportError:
        print("⚠️  psutil未安装，跳过内存测试")
        return None

def main():
    """主测试函数"""
    print("🚀 开始视觉模块全面集成测试")
    print("=" * 60)
    
    try:
        # 测试视觉模块组件
        component_results = test_vision_module_components()
        
        # 测试视觉模块性能
        performance_results = test_vision_performance()
        
        # 测试视觉模块内存使用
        memory_results = test_vision_memory_usage()
        
        print("\n" + "=" * 60)
        print("🎉 所有视觉模块集成测试通过!")
        print("=" * 60)
        
        # 生成测试报告
        print("\n📊 测试报告:")
        print(f"   组件测试: ✅ 通过")
        print(f"   性能测试: ✅ 通过")
        print(f"   内存测试: {'✅ 通过' if memory_results else '⚠️ 跳过'}")
        
        return {
            'component_results': component_results,
            'performance_results': performance_results,
            'memory_results': memory_results
        }
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    main() 