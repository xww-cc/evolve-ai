from functools import wraps
import logging
from config.logging_setup import setup_logging

logger = setup_logging()

def retry_on_error(max_retries: int = 3):
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"尝试 {attempt+1} 失败: {e}")
            logger.error(f"{func.__name__} 完全失败")
            return None
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"尝试 {attempt+1} 失败: {e}")
            logger.error(f"{func.__name__} 完全失败")
            return None
        
        # 根据函数类型返回相应的包装器
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    return decorator