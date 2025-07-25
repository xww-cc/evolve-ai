#!/usr/bin/env python3
"""
模型单元测试
测试 ModularMathReasoningNet 的基本功能
"""

import pytest
import torch
import numpy as np
from models.modular_net import ModularMathReasoningNet


class TestModularMathReasoningNet:
    """测试模块化数学推理网络"""
    
    def test_initialization(self):
        """测试网络初始化"""
        # 测试空配置初始化
        model = ModularMathReasoningNet(modules_config=[])
        assert isinstance(model, ModularMathReasoningNet)
        assert len(model.subnet_modules) == 0
        
        # 测试带配置初始化
        config = [{
            'type': 'linear',
            'input_dim': 10,
            'output_dim': 5,
            'widths': [10, 8, 5],
            'activation_fn_name': 'relu',
            'use_batchnorm': False,
            'module_type': 'linear'
        }]
        model = ModularMathReasoningNet(modules_config=config)
        assert len(model.subnet_modules) == 1
        
    def test_forward_propagation(self):
        """测试前向传播"""
        config = [{
            'type': 'linear',
            'input_dim': 10,
            'output_dim': 5,
            'widths': [10, 8, 5],
            'activation_fn_name': 'relu',
            'use_batchnorm': False,
            'module_type': 'linear'
        }]
        model = ModularMathReasoningNet(modules_config=config)
        
        # 测试输入输出维度
        x = torch.randn(3, 4)  # batch_size=3, input_dim=4 (默认输入维度)
        output = model(x)
        assert output.shape[0] == 3  # batch_size=3
        assert output.shape[1] in [1, 5]  # 可能是最终输出层或模块输出
        
    def test_module_operations(self):
        """测试模块操作"""
        model = ModularMathReasoningNet(modules_config=[])
        
        # 测试模块配置
        assert len(model.subnet_modules) == 0
        assert len(model.modules_config) == 0
        
        # 测试带配置的模型
        module_config = {
            'type': 'linear',
            'input_dim': 5,
            'output_dim': 3,
            'widths': [5, 4, 3],
            'activation_fn_name': 'relu',
            'use_batchnorm': False,
            'module_type': 'linear'
        }
        model_with_config = ModularMathReasoningNet(modules_config=[module_config])
        assert len(model_with_config.subnet_modules) == 1
        
    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效配置
        with pytest.raises(Exception):
            ModularMathReasoningNet(modules_config=None)
            
        # 测试空配置模型的前向传播
        model = ModularMathReasoningNet(modules_config=[])
        x = torch.randn(3, 4)
        output = model(x)  # 应该成功，使用默认输出层
        assert output.shape[0] == 3  # batch_size=3
        assert output.shape[1] in [1, 4]  # 可能是最终输出层或直接输出


class TestEpigeneticModule:
    """测试表观遗传模块"""
    
    def test_epigenetic_initialization(self):
        """测试表观遗传模块初始化"""
        from models.epigenetic import EpigeneticMarkers
        
        markers = EpigeneticMarkers()
        assert isinstance(markers, EpigeneticMarkers)
        assert len(markers) == 2
        assert markers.shape == (2,)
        
        # 测试克隆功能
        cloned_markers = markers.clone()
        assert isinstance(cloned_markers, EpigeneticMarkers)
        assert len(cloned_markers) == 2
        
    def test_epigenetic_forward(self):
        """测试表观遗传模块前向传播"""
        from models.epigenetic import EpigeneticMarkers
        
        markers = EpigeneticMarkers()
        
        # 测试基本操作
        assert len(markers) == 2
        assert markers.shape == (2,)
        
        # 测试索引访问
        value = markers[0]
        assert isinstance(value, torch.Tensor)
        
        # 测试设置值
        markers[0] = 0.5
        assert markers[0].item() == 0.5
        
        # 测试加法操作
        original_data = markers.data.clone()
        markers += 0.1
        assert torch.allclose(markers.data, original_data + 0.1)
        
        # 测试clamp操作
        markers.clamp()
        assert torch.all(markers.data >= -1.0)
        assert torch.all(markers.data <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 