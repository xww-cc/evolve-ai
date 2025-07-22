import os
from dotenv import load_dotenv
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Optional[str]]:
    """加载环境配置，支持密钥和代理设置。优化：添加验证和默认值"""
    load_dotenv()
    config = {
        'XAI_API_TYPE': os.getenv('XAI_API_TYPE', 'deepseek'),
        'XAI_MODEL': os.getenv('XAI_MODEL', 'deepseek-chat'),
        'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
        'XAI_API_KEY': os.getenv('XAI_API_KEY'),
        'XAI_BASE_URL': os.getenv('XAI_BASE_URL', 'https://api.x.ai/v1'),
        'XAI_PROXY_ENABLED': os.getenv('XAI_PROXY_ENABLED', 'false').lower() == 'true',
        'XAI_PROXY_HOST': os.getenv('XAI_PROXY_HOST', '127.0.0.1'),
        'XAI_PROXY_PORT': os.getenv('XAI_PROXY_PORT', '3000'),
        'XAI_PROXY_TYPE': os.getenv('XAI_PROXY_TYPE', 'socks5h'),
        'XAI_TIMEOUT': int(os.getenv('XAI_TIMEOUT', '30')),
        'XAI_TEMPERATURE': float(os.getenv('XAI_TEMPERATURE', '0.3')),
        'XAI_MAX_TOKENS': int(os.getenv('XAI_MAX_TOKENS', '200')),
        'XAI_FALLBACK_ENABLED': os.getenv('XAI_FALLBACK_ENABLED', 'true').lower() == 'true',
        'XAI_MAX_RETRIES': int(os.getenv('XAI_MAX_RETRIES', '3')),
        'WOLFRAM_API_KEY': os.getenv('WOLFRAM_API_KEY'),
        'KAGGLE_API_KEY': os.getenv('KAGGLE_API_KEY'),
        'OPENML_API_KEY': os.getenv('OPENML_API_KEY'),
    }
    # 验证密钥
    if config['XAI_API_TYPE'] == 'deepseek' and not config['DEEPSEEK_API_KEY']:
        logger.warning("DeepSeek API 密钥缺失，使用 fallback 模式")
    return config