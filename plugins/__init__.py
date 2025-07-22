# 包初始化，用于动态加载插件
import importlib.util
import os

def load_plugins(plugin_type: str):
    """动态加载插件 - 完整示例"""
    plugin_dir = os.path.join(os.path.dirname(__file__), plugin_type)
    plugins = {}
    for file in os.listdir(plugin_dir):
        if file.endswith('.py'):
            spec = importlib.util.spec_from_file_location(file[:-3], os.path.join(plugin_dir, file))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            plugins[file[:-3]] = module
    return plugins