import logging
import os
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
import time

class OptimizedLoggingManager:
    """优化的日志管理器 - 只输出关键信息"""
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_optimized_logging()
            self._initialized = True
    
    def _setup_optimized_logging(self):
        """设置优化的日志系统"""
        # 清除所有现有的处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 设置根日志器级别为INFO，显示更多信息
        root_logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用当前时间戳创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f'{timestamp}_evolve_ai_optimized.log')
        
        # 创建简洁的格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # 文件处理器
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器 - 显示INFO级别以上的信息
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)  # 改为INFO级别
        root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        root_logger.addHandler(file_handler)
        
        # 缓存根日志器
        self._root_logger = root_logger
        
        # 只在第一次初始化时输出信息
        root_logger.info("=== AI自主进化系统启动 ===")
    
    def log_evolution_progress(self, generation: int, population_size: int, 
                              best_score: float, avg_score: float, level: int):
        """记录进化进度 - 关键信息"""
        self._root_logger.info(
            f"世代 {generation:3d} | 种群 {population_size:2d} | "
            f"最佳 {best_score:.4f} | 平均 {avg_score:.4f} | 级别 {level}"
        )
    
    def log_evaluation_results(self, model_id: str, symbolic_score: float, 
                             realworld_score: float, complex_scores: dict = None):
        """记录评估结果 - 关键信息"""
        if complex_scores:
            complex_str = " | ".join([f"{k}: {v:.3f}" for k, v in complex_scores.items()])
            self._root_logger.info(
                f"模型 {model_id} | 符号: {symbolic_score:.3f} | "
                f"真实: {realworld_score:.3f} | {complex_str}"
            )
        else:
            self._root_logger.info(
                f"模型 {model_id} | 符号: {symbolic_score:.3f} | 真实: {realworld_score:.3f}"
            )
    
    def log_system_status(self, memory_usage: float, cpu_usage: float, 
                         evolution_speed: float, cache_hit_rate: float):
        """记录系统状态 - 关键指标"""
        self._root_logger.info(
            f"系统状态 | 内存: {memory_usage:.1f}% | CPU: {cpu_usage:.1f}% | "
            f"速度: {evolution_speed:.1f}代/秒 | 缓存: {cache_hit_rate:.1f}%"
        )
    
    def log_error(self, error_msg: str, context: str = ""):
        """记录错误信息"""
        if context:
            self._root_logger.error(f"{context}: {error_msg}")
        else:
            self._root_logger.error(error_msg)
    
    def log_warning(self, warning_msg: str, context: str = ""):
        """记录警告信息"""
        if context:
            self._root_logger.warning(f"{context}: {warning_msg}")
        else:
            self._root_logger.warning(warning_msg)
    
    def log_important(self, message: str):
        """记录重要信息"""
        self._root_logger.info(f"🔔 {message}")
    
    def log_success(self, message: str):
        """记录成功信息"""
        self._root_logger.info(f"✅ {message}")
    
    def log_progress(self, current: int, total: int, description: str = ""):
        """记录进度信息"""
        percentage = (current / total) * 100
        self._root_logger.info(f"📊 {description}: {current}/{total} ({percentage:.1f}%)")
    
    def log_performance_metrics(self, metrics: dict):
        """记录性能指标"""
        metrics_str = " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        self._root_logger.info(f"📈 性能指标: {metrics_str}")
    
    def log_evolution_summary(self, generation: int, improvements: dict):
        """记录进化总结"""
        improvement_str = " | ".join([f"{k}: {v:+.3f}" for k, v in improvements.items()])
        self._root_logger.info(f"🎯 世代 {generation} 总结: {improvement_str}")
    
    def set_verbose_mode(self, verbose: bool = False):
        """设置详细模式"""
        if verbose:
            self._root_logger.setLevel(logging.INFO)
            for handler in self._root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(logging.INFO)
        else:
            self._root_logger.setLevel(logging.WARNING)
            for handler in self._root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler.setLevel(logging.WARNING)
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """获取日志器"""
        if name:
            return logging.getLogger(name)
        else:
            return self._root_logger

# 全局日志管理器实例
_optimized_logging_manager = None

def setup_optimized_logging() -> OptimizedLoggingManager:
    """设置优化的日志系统"""
    global _optimized_logging_manager
    if _optimized_logging_manager is None:
        _optimized_logging_manager = OptimizedLoggingManager()
    return _optimized_logging_manager

def get_optimized_logger() -> OptimizedLoggingManager:
    """获取优化的日志管理器"""
    global _optimized_logging_manager
    if _optimized_logging_manager is None:
        _optimized_logging_manager = OptimizedLoggingManager()
    return _optimized_logging_manager 