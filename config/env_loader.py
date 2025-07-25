import os
from dotenv import load_dotenv
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Optional[str]]:
    """加载环境配置，支持密钥和代理设置。优化：添加验证和默认值"""
    load_dotenv()
    config = {
        'LLM_API_TYPE': os.getenv('LLM_API_TYPE', 'deepseek'),
        'LLM_MODEL': os.getenv('LLM_MODEL', 'deepseek-chat'),
        #deepseek
        'DEEPSEEK_API_KEY': os.getenv('DEEPSEEK_API_KEY'),
        'DEEPSEEK_API_URL': os.getenv('DEEPSEEK_API_URL','https://api.deepseek.com/v1'),
        #xai
        'XAI_API_KEY': os.getenv('XAI_API_KEY'),
        'XAI_API_URL': os.getenv('XAI_API_URL', 'https://api.x.ai/v1'),
        #千问
        'QWEN_API_KEY': os.getenv('QWEN_API_KEY'),
        'QWEN_API_URL': os.getenv('QWEN_API_URL','https://dashscope.aliyuncs.com/compatible-mode/v1'),
        #代理
        'LLM_PROXY_ENABLED': os.getenv('LLM_PROXY_ENABLED', 'false').lower() == 'true',
        'LLM_PROXY_HOST': os.getenv('LLM_PROXY_HOST', '127.0.0.1'),
        'LLM_PROXY_PORT': os.getenv('LLM_PROXY_PORT', '3000'),
        'LLM_PROXY_TYPE': os.getenv('LLM_PROXY_TYPE', 'socks5h'),
        #LLM参数
        'LLM_TIMEOUT': int(os.getenv('LLM_TIMEOUT', '30')),
        'LLM_TEMPERATURE': float(os.getenv('LLM_TEMPERATURE', '0.3')),
        'LLM_MAX_TOKENS': int(os.getenv('LLM_MAX_TOKENS', '200')),
        'LLM_FALLBACK_ENABLED': os.getenv('LLM_FALLBACK_ENABLED', 'true').lower() == 'true',
        'LLM_MAX_RETRIES': int(os.getenv('LLM_MAX_RETRIES', '3')),
        #外部真实世界接口
        'WOLFRAM_API_KEY': os.getenv('WOLFRAM_API_KEY'),
        'KAGGLE_API_KEY': os.getenv('KAGGLE_API_KEY'),
        'OPENML_API_KEY': os.getenv('OPENML_API_KEY'),
    }
    # 验证密钥
    if config['LLM_API_TYPE'] == 'deepseek' and not config['DEEPSEEK_API_KEY']:
        logger.warning("DeepSeek API 密钥缺失，使用 fallback 模式")
    if config['LLM_API_TYPE'] == 'qwen' and not config['QWEN_API_KEY']:
        logger.warning("Qwen API 密钥缺失，使用 fallback 模式")
    if config['LLM_API_TYPE'] == 'xai' and not config['XAI_API_KEY']:
        logger.warning("Xai API 密钥缺失，使用 fallback 模式")
    return config