import logging
import os
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
import time

class LoggingManager:
    """日志管理器 - 单例模式，优化性能和内存使用"""
    
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
            self._setup_logging()
            self._initialized = True
    
    def _setup_logging(self):
        """设置优化的日志系统"""
        # 清除所有现有的处理器
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # 设置根日志器级别
        root_logger.setLevel(logging.INFO)
        
        # 创建日志目录
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        # 使用当前时间戳创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f'{timestamp}_evolve_ai_execution.log')
        
        # 创建优化的格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 使用RotatingFileHandler进行文件轮转
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        # 控制台处理器 - 只在开发模式下显示
        if os.getenv('EVOLVE_AI_DEBUG', 'false').lower() == 'true':
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)
            root_logger.addHandler(console_handler)
        
        # 添加文件处理器
        root_logger.addHandler(file_handler)
        
        # 缓存根日志器
        self._root_logger = root_logger
        
        # 只在第一次初始化时输出信息
        root_logger.info("日志系统初始化完成")
    
    def get_logger(self, name: str = None) -> logging.Logger:
        """获取日志器"""
        if name:
            return logging.getLogger(name)
        else:
            return self._root_logger
    
    def set_level(self, level: str):
        """设置日志级别"""
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        if level.upper() in level_map:
            self._root_logger.setLevel(level_map[level.upper()])
    
    def cleanup_old_logs(self, days: int = 7):
        """清理旧日志文件"""
        log_dir = 'logs'
        if not os.path.exists(log_dir):
            return
        
        current_time = time.time()
        for filename in os.listdir(log_dir):
            if filename.endswith('.log'):
                file_path = os.path.join(log_dir, filename)
                file_time = os.path.getmtime(file_path)
                if current_time - file_time > days * 24 * 3600:
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass

# 全局日志管理器实例
_logging_manager = None

def setup_logging(log_file: str = 'evolve_ai_execution.log') -> logging.Logger:
    """设置日志配置 - 兼容性函数"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager.get_logger()

def get_logger(name: str = None) -> logging.Logger:
    """获取日志器 - 兼容性函数"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    return _logging_manager.get_logger(name)

def set_log_level(level: str):
    """设置日志级别"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    _logging_manager.set_level(level)

def cleanup_logs(days: int = 7):
    """清理旧日志文件"""
    global _logging_manager
    if _logging_manager is None:
        _logging_manager = LoggingManager()
    _logging_manager.cleanup_old_logs(days)