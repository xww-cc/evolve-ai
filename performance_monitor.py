#!/usr/bin/env python3
"""
性能监控脚本
用于实时追踪评估速度、进化速度、CPU和内存使用率
"""

import time
import psutil
import asyncio
import threading
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from config.logging_setup import setup_logging

logger = setup_logging()

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    evaluation_speed: float  # 评估/秒
    evolution_speed: float   # 进化/秒
    population_size: int
    generation: int
    best_fitness: float
    avg_fitness: float

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.logger = logger
        self.metrics_history: List[PerformanceMetrics] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # 性能统计
        self.evaluation_count = 0
        self.evolution_count = 0
        self.start_time = None
        self.last_evaluation_time = None
        self.last_evolution_time = None
        
    def start_monitoring(self):
        """开始性能监控"""
        if self.is_monitoring:
            self.logger.warning("性能监控已在运行")
            return
            
        self.is_monitoring = True
        self.start_time = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("🚀 性能监控已启动")
        
    def stop_monitoring(self):
        """停止性能监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("⏹️  性能监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # 记录关键指标
                if len(self.metrics_history) % 10 == 0:  # 每10次记录一次
                    self.logger.info(
                        f"📊 性能监控 - CPU: {metrics.cpu_percent:.1f}%, "
                        f"内存: {metrics.memory_percent:.1f}%, "
                        f"评估速度: {metrics.evaluation_speed:.1f}/秒, "
                        f"进化速度: {metrics.evolution_speed:.1f}/秒"
                    )
                    
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"性能监控错误: {e}")
                time.sleep(self.monitoring_interval)
                
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        # 系统资源
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        
        # 计算评估和进化速度
        current_time = time.time()
        evaluation_speed = 0.0
        evolution_speed = 0.0
        
        if self.last_evaluation_time:
            time_diff = current_time - self.last_evaluation_time
            if time_diff > 0:
                evaluation_speed = self.evaluation_count / time_diff
                
        if self.last_evolution_time:
            time_diff = current_time - self.last_evolution_time
            if time_diff > 0:
                evolution_speed = self.evolution_count / time_diff
        
        return PerformanceMetrics(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            evaluation_speed=evaluation_speed,
            evolution_speed=evolution_speed,
            population_size=0,  # 需要外部设置
            generation=0,        # 需要外部设置
            best_fitness=0.0,   # 需要外部设置
            avg_fitness=0.0     # 需要外部设置
        )
        
    def record_evaluation(self):
        """记录评估事件"""
        self.evaluation_count += 1
        self.last_evaluation_time = time.time()
        
    def record_evolution(self):
        """记录进化事件"""
        self.evolution_count += 1
        self.last_evolution_time = time.time()
        
    def update_evolution_info(self, population_size: int, generation: int, 
                            best_fitness: float, avg_fitness: float):
        """更新进化信息"""
        if self.metrics_history:
            latest = self.metrics_history[-1]
            latest.population_size = population_size
            latest.generation = generation
            latest.best_fitness = best_fitness
            latest.avg_fitness = avg_fitness
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self.metrics_history:
            return {"error": "没有性能数据"}
            
        # 计算统计信息
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_percent for m in self.metrics_history]
        evaluation_speeds = [m.evaluation_speed for m in self.metrics_history if m.evaluation_speed > 0]
        evolution_speeds = [m.evolution_speed for m in self.metrics_history if m.evolution_speed > 0]
        
        summary = {
            "monitoring_duration": time.time() - self.start_time if self.start_time else 0,
            "total_metrics": len(self.metrics_history),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
                "min": min(cpu_values) if cpu_values else 0
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
                "min": min(memory_values) if memory_values else 0
            },
            "evaluation_speed": {
                "avg": sum(evaluation_speeds) / len(evaluation_speeds) if evaluation_speeds else 0,
                "max": max(evaluation_speeds) if evaluation_speeds else 0,
                "total": self.evaluation_count
            },
            "evolution_speed": {
                "avg": sum(evolution_speeds) / len(evolution_speeds) if evolution_speeds else 0,
                "max": max(evolution_speeds) if evolution_speeds else 0,
                "total": self.evolution_count
            }
        }
        
        return summary
        
    def generate_performance_report(self) -> str:
        """生成性能报告"""
        summary = self.get_performance_summary()
        
        if "error" in summary:
            return f"❌ {summary['error']}"
            
        report = f"""
📊 性能监控报告
================

⏱️  监控时长: {summary['monitoring_duration']:.1f}秒
📈 数据点数: {summary['total_metrics']}

🖥️  CPU使用率:
   平均: {summary['cpu']['avg']:.1f}%
   最高: {summary['cpu']['max']:.1f}%
   最低: {summary['cpu']['min']:.1f}%

💾 内存使用率:
   平均: {summary['memory']['avg']:.1f}%
   最高: {summary['memory']['max']:.1f}%
   最低: {summary['memory']['min']:.1f}%

⚡ 评估性能:
   平均速度: {summary['evaluation_speed']['avg']:.1f} 评估/秒
   最高速度: {summary['evaluation_speed']['max']:.1f} 评估/秒
   总评估数: {summary['evaluation_speed']['total']}

🧬 进化性能:
   平均速度: {summary['evolution_speed']['avg']:.1f} 进化/秒
   最高速度: {summary['evolution_speed']['max']:.1f} 进化/秒
   总进化数: {summary['evolution_speed']['total']}

📊 性能评级:
   CPU效率: {'🟢 优秀' if summary['cpu']['avg'] < 50 else '🟡 良好' if summary['cpu']['avg'] < 80 else '🔴 需要优化'}
   内存效率: {'🟢 优秀' if summary['memory']['avg'] < 60 else '🟡 良好' if summary['memory']['avg'] < 85 else '🔴 需要优化'}
   评估效率: {'🟢 优秀' if summary['evaluation_speed']['avg'] > 10 else '🟡 良好' if summary['evaluation_speed']['avg'] > 5 else '🔴 需要优化'}
"""
        
        return report

class PerformanceTracker:
    """性能追踪器 - 用于装饰器模式"""
    
    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        
    def track_evaluation(self, func):
        """追踪评估性能的装饰器"""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                self.monitor.record_evaluation()
                return result
            finally:
                evaluation_time = time.time() - start_time
                self.monitor.logger.debug(f"评估耗时: {evaluation_time:.3f}秒")
        return wrapper
        
    def track_evolution(self, func):
        """追踪进化性能的装饰器"""
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                self.monitor.record_evolution()
                return result
            finally:
                evolution_time = time.time() - start_time
                self.monitor.logger.debug(f"进化耗时: {evolution_time:.3f}秒")
        return wrapper

async def demo_performance_monitoring():
    """演示性能监控功能"""
    monitor = PerformanceMonitor(monitoring_interval=0.5)
    tracker = PerformanceTracker(monitor)
    
    # 启动监控
    monitor.start_monitoring()
    
    # 模拟一些操作
    for i in range(10):
        monitor.record_evaluation()
        monitor.record_evolution()
        monitor.update_evolution_info(
            population_size=10 + i,
            generation=i,
            best_fitness=0.8 + i * 0.02,
            avg_fitness=0.7 + i * 0.01
        )
        await asyncio.sleep(0.5)
    
    # 停止监控
    monitor.stop_monitoring()
    
    # 生成报告
    report = monitor.generate_performance_report()
    print(report)

if __name__ == "__main__":
    asyncio.run(demo_performance_monitoring()) 