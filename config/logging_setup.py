import logging
import os
from datetime import datetime

# 全局变量跟踪是否已初始化
_logging_initialized = False
_root_logger = None

def setup_logging(log_file: str = 'evolve_ai_execution.log') -> logging.Logger:
    """设置日志配置，优化：添加文件旋转和级别控制，防止重复初始化"""
    global _logging_initialized, _root_logger
    
    # 如果已经初始化，直接返回根日志器
    if _logging_initialized:
        return _root_logger
    
    # 清除所有现有的处理器
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 设置根日志器级别
    root_logger.setLevel(logging.INFO)
    
    # 创建日志目录
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f'{timestamp}_{log_file}')
    
    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 文件处理器
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # 添加处理器到根日志器
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # 标记为已初始化
    _logging_initialized = True
    _root_logger = root_logger
    
    # 只输出一次初始化信息
    root_logger.info("日志系统初始化完成")
    
    return root_logger

def get_logger(name: str = None) -> logging.Logger:
    """获取日志器，确保日志系统已初始化"""
    if not _logging_initialized:
        setup_logging()
    
    if name:
        return logging.getLogger(name)
    else:
        return logging.getLogger()