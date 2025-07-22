#!/usr/bin/env python3
"""
pytest配置文件
提供测试夹具和配置
"""

import pytest
import asyncio
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置异步测试
@pytest.fixture(scope="session")
def event_loop():
    """创建事件循环"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# 测试配置
def pytest_configure(config):
    """配置pytest"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as async"
    )

def pytest_collection_modifyitems(config, items):
    """修改测试收集"""
    for item in items:
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio) 