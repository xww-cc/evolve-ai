#!/usr/bin/env python3
"""
系统状态检查器
用于监控和检查系统各组件的状态
"""

import torch
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ComponentStatus:
    """组件状态"""
    name: str
    status: str  # 'healthy', 'warning', 'error'
    message: str
    details: Optional[Dict] = None

class SystemStatusChecker:
    """系统状态检查器"""
    
    def __init__(self):
        self.components = {}
        self.health_checks = {}
        self._register_default_checks()
    
    def _register_default_checks(self):
        """注册默认的健康检查"""
        self.register_health_check('torch', self._check_torch)
        self.register_health_check('memory', self._check_memory)
        self.register_health_check('gpu', self._check_gpu)
        self.register_health_check('models', self._check_models)
    
    def register_health_check(self, component_name: str, check_function):
        """注册健康检查函数"""
        self.health_checks[component_name] = check_function
    
    def _check_torch(self) -> ComponentStatus:
        """检查PyTorch状态"""
        try:
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            mps_available = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
            
            status = 'healthy'
            message = f"PyTorch {version} 正常运行"
            
            if cuda_available:
                message += f", CUDA可用"
            elif mps_available:
                message += f", MPS可用"
            else:
                message += f", 使用CPU"
            
            return ComponentStatus(
                name='torch',
                status=status,
                message=message,
                details={
                    'version': version,
                    'cuda_available': cuda_available,
                    'mps_available': mps_available
                }
            )
        except Exception as e:
            return ComponentStatus(
                name='torch',
                status='error',
                message=f"PyTorch检查失败: {e}",
                details={'error': str(e)}
            )
    
    def _check_memory(self) -> ComponentStatus:
        """检查内存状态"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            if memory_usage < 80:
                status = 'healthy'
                message = f"内存使用正常 ({memory_usage:.1f}%)"
            elif memory_usage < 95:
                status = 'warning'
                message = f"内存使用较高 ({memory_usage:.1f}%)"
            else:
                status = 'error'
                message = f"内存使用过高 ({memory_usage:.1f}%)"
            
            return ComponentStatus(
                name='memory',
                status=status,
                message=message,
                details={
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory_usage
                }
            )
        except ImportError:
            return ComponentStatus(
                name='memory',
                status='warning',
                message="无法检查内存状态 (psutil未安装)",
                details={'error': 'psutil not available'}
            )
        except Exception as e:
            return ComponentStatus(
                name='memory',
                status='error',
                message=f"内存检查失败: {e}",
                details={'error': str(e)}
            )
    
    def _check_gpu(self) -> ComponentStatus:
        """检查GPU状态"""
        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                # 检查GPU内存
                memory_allocated = torch.cuda.memory_allocated(current_device)
                memory_reserved = torch.cuda.memory_reserved(current_device)
                memory_total = torch.cuda.get_device_properties(current_device).total_memory
                
                memory_usage = (memory_allocated / memory_total) * 100
                
                if memory_usage < 80:
                    status = 'healthy'
                    message = f"GPU {device_name} 正常 ({memory_usage:.1f}%)"
                elif memory_usage < 95:
                    status = 'warning'
                    message = f"GPU {device_name} 内存使用较高 ({memory_usage:.1f}%)"
                else:
                    status = 'error'
                    message = f"GPU {device_name} 内存使用过高 ({memory_usage:.1f}%)"
                
                return ComponentStatus(
                    name='gpu',
                    status=status,
                    message=message,
                    details={
                        'device_count': gpu_count,
                        'current_device': current_device,
                        'device_name': device_name,
                        'memory_allocated': memory_allocated,
                        'memory_reserved': memory_reserved,
                        'memory_total': memory_total,
                        'memory_usage_percent': memory_usage
                    }
                )
            else:
                return ComponentStatus(
                    name='gpu',
                    status='healthy',
                    message="GPU不可用，使用CPU模式",
                    details={'available': False}
                )
            except Exception as e:
            return ComponentStatus(
                name='gpu',
                status='error',
                message=f"GPU检查失败: {e}",
                details={'error': str(e)}
            )
    
    def _check_models(self) -> ComponentStatus:
        """检查模型状态"""
        try:
            # 这里可以添加模型相关的检查
            # 目前返回基本状态
            return ComponentStatus(
                name='models',
                status='healthy',
                message="模型组件正常",
                details={'status': 'operational'}
            )
        except Exception as e:
            return ComponentStatus(
                name='models',
                status='error',
                message=f"模型检查失败: {e}",
                details={'error': str(e)}
            )
    
    def check_all_components(self) -> Dict[str, ComponentStatus]:
        """检查所有组件状态"""
        results = {}
        
        for component_name, check_function in self.health_checks.items():
            try:
                status = check_function()
                results[component_name] = status
                logger.info(f"组件 {component_name}: {status.status} - {status.message}")
            except Exception as e:
                results[component_name] = ComponentStatus(
                    name=component_name,
                    status='error',
                    message=f"检查失败: {e}",
                    details={'error': str(e)}
                )
                logger.error(f"组件 {component_name} 检查失败: {e}")
        
        return results
    
    def get_system_summary(self) -> Dict:
        """获取系统状态总结"""
        component_statuses = self.check_all_components()
        
        total_components = len(component_statuses)
        healthy_components = sum(1 for status in component_statuses.values() if status.status == 'healthy')
        warning_components = sum(1 for status in component_statuses.values() if status.status == 'warning')
        error_components = sum(1 for status in component_statuses.values() if status.status == 'error')
        
        overall_status = 'healthy'
        if error_components > 0:
            overall_status = 'error'
        elif warning_components > 0:
            overall_status = 'warning'
        
        return {
            'overall_status': overall_status,
            'total_components': total_components,
            'healthy_components': healthy_components,
            'warning_components': warning_components,
            'error_components': error_components,
            'component_details': component_statuses
        }
    
    def print_system_status(self):
        """打印系统状态"""
        summary = self.get_system_summary()
        
        print("=== 系统状态检查 ===")
        print(f"总体状态: {summary['overall_status']}")
        print(f"组件总数: {summary['total_components']}")
        print(f"健康组件: {summary['healthy_components']}")
        print(f"警告组件: {summary['warning_components']}")
        print(f"错误组件: {summary['error_components']}")
        print()
        
        for component_name, status in summary['component_details'].items():
            status_icon = {
                'healthy': '✅',
                'warning': '⚠️',
                'error': '❌'
            }.get(status.status, '❓')
            
            print(f"{status_icon} {component_name}: {status.message}")
        
        print("==================")

# 全局系统状态检查器实例
system_status_checker = SystemStatusChecker()

def get_system_status_checker():
    """获取系统状态检查器实例"""
    return system_status_checker 