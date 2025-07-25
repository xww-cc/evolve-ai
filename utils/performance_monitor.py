import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import statistics
import psutil

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    operation_name: str
    duration: float
    success: bool
    error_message: Optional[str] = None
    additional_data: Optional[Dict] = None

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.operation_times: Dict[str, List[float]] = defaultdict(list)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.success_counts: Dict[str, int] = defaultdict(int)
        self.evolution_metrics: Dict[str, float] = {}
    
    def record_operation(self, operation_name: str, duration: float, success: bool, 
                        error_message: Optional[str] = None, additional_data: Optional[Dict] = None):
        """记录操作性能"""
        metric = PerformanceMetrics(
            operation_name=operation_name,
            duration=duration,
            success=success,
            error_message=error_message,
            additional_data=additional_data
        )
        self.metrics.append(metric)
        self.operation_times[operation_name].append(duration)
        
        if success:
            self.success_counts[operation_name] += 1
        else:
            self.error_counts[operation_name] += 1
    
    def record_metrics(self, **kwargs):
        """记录进化相关指标"""
        for key, value in kwargs.items():
            self.evolution_metrics[key] = value
            logger.debug(f"记录指标: {key} = {value}")
    
    def get_operation_stats(self, operation_name: str) -> Dict:
        """获取操作统计信息"""
        times = self.operation_times.get(operation_name, [])
        if not times:
            return {
                'count': 0,
                'avg_duration': 0,
                'min_duration': 0,
                'max_duration': 0,
                'success_rate': 0,
                'error_rate': 0
            }
        
        success_count = self.success_counts.get(operation_name, 0)
        error_count = self.error_counts.get(operation_name, 0)
        total_count = success_count + error_count
        
        return {
            'count': total_count,
            'avg_duration': statistics.mean(times),
            'min_duration': min(times),
            'max_duration': max(times),
            'success_rate': success_count / total_count if total_count > 0 else 0,
            'error_rate': error_count / total_count if total_count > 0 else 0
        }
    
    def get_summary(self) -> Dict:
        """获取性能总结"""
        summary = {}
        for operation_name in self.operation_times.keys():
            summary[operation_name] = self.get_operation_stats(operation_name)
        return summary
    
    def print_summary(self):
        """打印性能总结"""
        summary = self.get_summary()
        logger.info("=== 性能监控总结 ===")
        for operation_name, stats in summary.items():
            logger.info(f"{operation_name}:")
            logger.info(f"  总次数: {stats['count']}")
            logger.info(f"  平均耗时: {stats['avg_duration']:.4f}s")
            logger.info(f"  最小耗时: {stats['min_duration']:.4f}s")
            logger.info(f"  最大耗时: {stats['max_duration']:.4f}s")
            logger.info(f"  成功率: {stats['success_rate']:.2%}")
            logger.info(f"  错误率: {stats['error_rate']:.2%}")
        
        if self.evolution_metrics:
            logger.info("=== 进化指标 ===")
            for key, value in self.evolution_metrics.items():
                logger.info(f"  {key}: {value}")
        logger.info("==================")
    
    def generate_performance_report(self) -> Dict:
        """生成性能报告"""
        summary = self.get_summary()
        
        # 计算总体统计
        total_operations = sum(stats['count'] for stats in summary.values())
        total_success = sum(stats['count'] * stats['success_rate'] for stats in summary.values())
        total_errors = sum(stats['count'] * stats['error_rate'] for stats in summary.values())
        
        # 计算平均性能
        all_durations = []
        for times in self.operation_times.values():
            all_durations.extend(times)
        
        avg_duration = statistics.mean(all_durations) if all_durations else 0
        min_duration = min(all_durations) if all_durations else 0
        max_duration = max(all_durations) if all_durations else 0
        
        report = {
            'summary': summary,
            'evolution_metrics': self.evolution_metrics,
            'overall_stats': {
                'total_operations': total_operations,
                'total_success': total_success,
                'total_errors': total_errors,
                'overall_success_rate': total_success / total_operations if total_operations > 0 else 0,
                'overall_error_rate': total_errors / total_operations if total_operations > 0 else 0,
                'avg_duration': avg_duration,
                'min_duration': min_duration,
                'max_duration': max_duration
            },
            'timestamp': time.time()
        }
        
        return report
    
    def get_realtime_metrics(self) -> Dict:
        """获取实时性能指标"""
        # 获取系统资源信息
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
        except Exception as e:
            logger.warning(f"获取系统资源信息失败: {e}")
            cpu_percent = 0.0
            memory_percent = 0.0
        
        return {
            'current_metrics': self.evolution_metrics,
            'operation_summary': self.get_summary(),
            'total_operations': sum(stats['count'] for stats in self.get_summary().values()),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'timestamp': time.time()
        }
    
    def check_system_health(self) -> Dict:
        """检查系统健康状态"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # 确定各组件状态
            cpu_status = 'healthy' if cpu_percent <= 80 else 'warning'
            memory_status = 'healthy' if memory.percent <= 85 else 'warning'
            disk_status = 'healthy' if disk.percent <= 90 else 'warning'
            
            health_status = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'cpu_status': cpu_status,
                'memory_status': memory_status,
                'disk_status': disk_status,
                'overall_status': 'healthy',
                'warnings': []
            }
            
            # 检查警告条件
            if cpu_percent > 80:
                health_status['warnings'].append('CPU使用率过高')
                health_status['overall_status'] = 'warning'
            
            if memory.percent > 85:
                health_status['warnings'].append('内存使用率过高')
                health_status['overall_status'] = 'warning'
            
            if disk.percent > 90:
                health_status['warnings'].append('磁盘空间不足')
                health_status['overall_status'] = 'warning'
            
            return health_status
            
        except Exception as e:
            logger.warning(f"系统健康检查失败: {e}")
            return {
                'overall_status': 'error',
                'error': str(e),
                'warnings': ['无法获取系统状态']
            }
    
    def start_monitoring(self):
        """开始监控（兼容性方法）"""
        pass
    
    def stop_monitoring(self):
        """停止监控（兼容性方法）"""
        pass

# 全局性能监控器实例
performance_monitor = PerformanceMonitor()

def monitor_operation(operation_name: str):
    """操作监控装饰器"""
    def decorator(func):
        import functools
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_operation(operation_name, duration, True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_operation(operation_name, duration, False, str(e))
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                performance_monitor.record_operation(operation_name, duration, True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                performance_monitor.record_operation(operation_name, duration, False, str(e))
                raise
        
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator 

def get_performance_monitor():
    """获取性能监控器实例"""
    return performance_monitor 