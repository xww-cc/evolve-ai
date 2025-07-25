#!/usr/bin/env python3
"""
评估诊断工具
用于诊断和修复评估失败的问题
"""

import torch
import logging
import asyncio
from typing import Dict, List, Optional
from models.modular_net import ModularMathReasoningNet
from evaluators.symbolic_evaluator import SymbolicEvaluator
from evaluators.realworld_evaluator import RealWorldEvaluator
from evaluators.vision_evaluator import VisionEvaluator
from config.logging_setup import setup_logging

logger = setup_logging()

class EvaluationDiagnostic:
    """评估诊断工具"""
    
    def __init__(self):
        self.symbolic_evaluator = SymbolicEvaluator()
        self.realworld_evaluator = RealWorldEvaluator()
        self.vision_evaluator = VisionEvaluator(hidden_dim=256)
        
    async def diagnose_evaluation_issues(self) -> Dict:
        """诊断评估问题"""
        print("🔍 开始评估诊断...")
        
        issues = {
            'model_creation': self._test_model_creation(),
            'basic_forward': self._test_basic_forward(),
            'symbolic_evaluation': await self._test_symbolic_evaluation(),
            'realworld_evaluation': await self._test_realworld_evaluation(),
            'vision_evaluation': self._test_vision_evaluation(),
            'memory_usage': self._test_memory_usage(),
            'gpu_availability': self._test_gpu_availability()
        }
        
        return issues
    
    def _test_model_creation(self) -> Dict:
        """测试模型创建"""
        try:
            # 创建基础模型 - 使用正确的配置格式
            modules_config = [
                {
                    'input_dim': 4,
                    'output_dim': 64,
                    'widths': [32, 64],
                    'activation_fn_name': 'relu',
                    'use_batchnorm': True,
                    'module_type': 'mlp',
                    'input_source': 'initial_input'
                },
                {
                    'input_dim': 64,
                    'output_dim': 32,
                    'widths': [48, 32],
                    'activation_fn_name': 'relu',
                    'use_batchnorm': True,
                    'module_type': 'mlp',
                    'input_source': 'module_0'
                }
            ]
            
            model = ModularMathReasoningNet(
                modules_config=modules_config,
                epigenetic_markers={'learning_rate': 0.001}
            )
            
            return {
                'status': 'success',
                'message': '模型创建成功',
                'details': {
                    'total_params': sum(p.numel() for p in model.parameters()),
                    'model_type': type(model).__name__,
                    'num_modules': len(model.subnet_modules)
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'模型创建失败: {e}',
                'details': {'error': str(e)}
            }
    
    def _test_basic_forward(self) -> Dict:
        """测试基础前向传播"""
        try:
            # 使用正确的配置格式
            modules_config = [
                {
                    'input_dim': 4,
                    'output_dim': 64,
                    'widths': [32, 64],
                    'activation_fn_name': 'relu',
                    'use_batchnorm': True,
                    'module_type': 'mlp',
                    'input_source': 'initial_input'
                },
                {
                    'input_dim': 64,
                    'output_dim': 32,
                    'widths': [48, 32],
                    'activation_fn_name': 'relu',
                    'use_batchnorm': True,
                    'module_type': 'mlp',
                    'input_source': 'module_0'
                }
            ]
            
            model = ModularMathReasoningNet(
                modules_config=modules_config,
                epigenetic_markers={'learning_rate': 0.001}
            )
            
            # 测试输入
            test_input = torch.randn(2, 4)
            output = model(test_input)
            
            return {
                'status': 'success',
                'message': '基础前向传播成功',
                'details': {
                    'input_shape': test_input.shape,
                    'output_shape': output.shape,
                    'output_stats': {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'min': output.min().item(),
                        'max': output.max().item()
                    }
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'基础前向传播失败: {e}',
                'details': {'error': str(e)}
            }
    
    async def _test_symbolic_evaluation(self) -> Dict:
        """测试符号评估"""
        try:
            # 使用正确的配置格式
            modules_config = [
                {
                    'input_dim': 4,
                    'output_dim': 64,
                    'widths': [32, 64],
                    'activation_fn_name': 'relu',
                    'use_batchnorm': True,
                    'module_type': 'mlp',
                    'input_source': 'initial_input'
                },
                {
                    'input_dim': 64,
                    'output_dim': 32,
                    'widths': [48, 32],
                    'activation_fn_name': 'relu',
                    'use_batchnorm': True,
                    'module_type': 'mlp',
                    'input_source': 'module_0'
                }
            ]
            
            model = ModularMathReasoningNet(
                modules_config=modules_config,
                epigenetic_markers={'learning_rate': 0.001}
            )
            
            # 测试符号评估 - 使用同步版本
            score = await self.symbolic_evaluator.evaluate(model, level=0)
            
            return {
                'status': 'success',
                'message': '符号评估成功',
                'details': {
                    'score': score,
                    'evaluator_type': type(self.symbolic_evaluator).__name__
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'符号评估失败: {e}',
                'details': {'error': str(e)}
            }
    
    async def _test_realworld_evaluation(self) -> Dict:
        """测试真实世界评估"""
        try:
            # 使用正确的配置格式
            modules_config = [
                {
                    'input_dim': 4,
                    'output_dim': 64,
                    'widths': [32, 64],
                    'activation_fn_name': 'relu',
                    'use_batchnorm': True,
                    'module_type': 'mlp',
                    'input_source': 'initial_input'
                },
                {
                    'input_dim': 64,
                    'output_dim': 32,
                    'widths': [48, 32],
                    'activation_fn_name': 'relu',
                    'use_batchnorm': True,
                    'module_type': 'mlp',
                    'input_source': 'module_0'
                }
            ]
            
            model = ModularMathReasoningNet(
                modules_config=modules_config,
                epigenetic_markers={'learning_rate': 0.001}
            )
            
            # 测试真实世界评估 - 使用同步版本
            score = await self.realworld_evaluator.evaluate(model)
            
            return {
                'status': 'success',
                'message': '真实世界评估成功',
                'details': {
                    'score': score,
                    'evaluator_type': type(self.realworld_evaluator).__name__
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'真实世界评估失败: {e}',
                'details': {'error': str(e)}
            }
    
    def _test_vision_evaluation(self) -> Dict:
        """测试视觉评估"""
        try:
            # 创建测试视觉特征
            test_features = torch.randn(2, 10, 256)
            
            # 测试视觉评估
            evaluation_results = self.vision_evaluator(test_features)
            
            return {
                'status': 'success',
                'message': '视觉评估成功',
                'details': {
                    'overall_score': evaluation_results['overall_score'].mean().item(),
                    'understanding_score': evaluation_results['understanding_score'].mean().item(),
                    'reasoning_score': evaluation_results['reasoning_score'].mean().item(),
                    'creation_score': evaluation_results['creation_score'].mean().item(),
                    'spatial_score': evaluation_results['spatial_score'].mean().item(),
                    'comprehensive_score': evaluation_results['comprehensive_score'].mean().item()
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'视觉评估失败: {e}',
                'details': {'error': str(e)}
            }
    
    def _test_memory_usage(self) -> Dict:
        """测试内存使用"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                'status': 'success',
                'message': f'内存使用正常 ({memory.percent:.1f}%)',
                'details': {
                    'total_memory': memory.total,
                    'available_memory': memory.available,
                    'memory_percent': memory.percent
                }
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'内存检查失败: {e}',
                'details': {'error': str(e)}
            }
    
    def _test_gpu_availability(self) -> Dict:
        """测试GPU可用性"""
        try:
            cuda_available = torch.cuda.is_available()
            
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                return {
                    'status': 'success',
                    'message': f'GPU可用: {device_name}',
                    'details': {
                        'gpu_count': gpu_count,
                        'current_device': current_device,
                        'device_name': device_name
                    }
                }
            else:
                return {
                    'status': 'warning',
                    'message': 'GPU不可用，使用CPU',
                    'details': {'cuda_available': False}
                }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'GPU检查失败: {e}',
                'details': {'error': str(e)}
            }
    
    def print_diagnostic_report(self, issues: Dict):
        """打印诊断报告"""
        print("\n" + "="*60)
        print("🔍 评估诊断报告")
        print("="*60)
        
        total_tests = len(issues)
        successful_tests = sum(1 for issue in issues.values() if issue['status'] == 'success')
        warning_tests = sum(1 for issue in issues.values() if issue['status'] == 'warning')
        error_tests = sum(1 for issue in issues.values() if issue['status'] == 'error')
        
        print(f"📊 测试统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   成功: {successful_tests}")
        print(f"   警告: {warning_tests}")
        print(f"   错误: {error_tests}")
        print()
        
        for test_name, result in issues.items():
            status_icon = {
                'success': '✅',
                'warning': '⚠️',
                'error': '❌'
            }.get(result['status'], '❓')
            
            print(f"{status_icon} {test_name}: {result['message']}")
            
            if result['status'] == 'error' and 'details' in result:
                print(f"   错误详情: {result['details'].get('error', '未知错误')}")
        
        print("\n" + "="*60)
        
        if error_tests == 0:
            print("🎉 所有评估测试通过！")
        else:
            print(f"⚠️  发现 {error_tests} 个评估问题，请检查上述错误详情。")
        
        print("="*60)

async def main():
    """主函数"""
    diagnostic = EvaluationDiagnostic()
    issues = await diagnostic.diagnose_evaluation_issues()
    diagnostic.print_diagnostic_report(issues)

if __name__ == "__main__":
    asyncio.run(main()) 