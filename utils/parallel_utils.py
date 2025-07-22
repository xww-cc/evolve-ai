import multiprocessing
import os
from typing import Callable, List, Any

def parallel_map(func: Callable, args: List[Any]) -> List[Any]:
    """并行映射 - 完整，使用multiprocessing"""
    # 在测试环境中禁用多进程
    if os.environ.get('TESTING', 'false').lower() == 'true':
        return [func(arg) for arg in args]
    
    try:
        with multiprocessing.Pool() as pool:
            return pool.map(func, args)
    except Exception as e:
        # 如果多进程失败，回退到单进程
        return [func(arg) for arg in args]