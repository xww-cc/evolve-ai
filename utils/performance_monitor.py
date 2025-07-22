import time
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import statistics

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
        logger.info("==================")

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