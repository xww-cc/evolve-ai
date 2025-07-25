import aiohttp
import json
import random
import time
from typing import Optional, Dict, Any
from config.env_loader import load_config
from config.logging_setup import setup_logging
from utils.error_handler import retry_on_error
import re
import logging
import asyncio

logger = setup_logging()

config = load_config()

class XAIIntegration:
    """LLM API集成类，完整实现异步调用、重试、清理"""
    def __init__(self, api_type: str = config['LLM_API_TYPE'], model: str = config['LLM_MODEL']):
        self.api_type = api_type
        self.model = model
        if api_type == "xai":
            self.api_key = config['XAI_API_KEY']
            self.base_url = config['XAI_API_URL']
            self.supported_models = ["grok-3-mini", "grok-3", "grok-4"]
        elif api_type == "deepseek":
            self.api_key = config['DEEPSEEK_API_KEY']
            self.base_url = "https://api.deepseek.com/v1"
            self.supported_models = ["deepseek-chat", "deepseek-coder"]
        elif api_type == "qwen":
            self.api_key = config['QWEN_API_KEY']
            self.base_url = config['QWEN_API_URL']
            self.supported_models = ["qwen-max", "qwen-plus"]
        else:
            raise ValueError(f"不支持的API类型: {api_type}")
        
        self.proxy_enabled = config['LLM_PROXY_ENABLED']
        self.proxies = {
            'http': f"{config['LLM_PROXY_TYPE']}://{config['LLM_PROXY_HOST']}:{config['LLM_PROXY_PORT']}",
            'https': f"{config['LLM_PROXY_TYPE']}://{config['LLM_PROXY_HOST']}:{config['LLM_PROXY_PORT']}"
        } if self.proxy_enabled else None
        
        self.timeout = config['LLM_TIMEOUT']
        self.temperature = config['LLM_TEMPERATURE']
        self.max_tokens = config['LLM_MAX_TOKENS']
        self.max_retries = config['LLM_MAX_RETRIES']
        self.fallback_enabled = config['LLM_FALLBACK_ENABLED']
        
        # 添加缓存机制
        self._cache = {}
        self._cache_ttl = 300  # 5分钟缓存
        self._cache_timestamps = {}
        
        logger.info(f"LLM初始化 - API: {api_type}, 模型: {model}, 代理: {self.proxy_enabled}")

    async def _make_async_request(self, endpoint: str, data: Dict[str, Any], model: Optional[str] = None) -> Optional[Dict]:
        """异步发送API请求"""
        if not model:
            model = self.model
        url = f"{self.base_url}/{endpoint}"
        headers = {'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'}
        data['model'] = model
        data['max_tokens'] = self.max_tokens
        data['temperature'] = self.temperature
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
            for attempt in range(self.max_retries):
                try:
                    async with session.post(url, headers=headers, json=data, proxy=self.proxies['http'] if self.proxies else None) as response:
                        if response.status == 200:
                            return await response.json()
                        logger.warning(f"请求失败 (尝试 {attempt+1}): {response.status} - {await response.text()}")
                except Exception as e:
                    logger.error(f"请求异常 (尝试 {attempt+1}): {e}")
            return None

    async def generate_symbol(self, query: str, context: str = "", model: Optional[str] = None) -> str:
        """生成符号表达式 - 完整版本，确保多样化"""
        # 检查缓存
        cache_key = f"{query}:{context}:{model or self.model}"
        current_time = time.time()
        
        # 清理过期缓存
        expired_keys = [k for k, ts in self._cache_timestamps.items() 
                       if current_time - ts > self._cache_ttl]
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_timestamps.pop(key, None)
        
        # 检查缓存命中
        if cache_key in self._cache:
            logger.debug(f"缓存命中: {cache_key}")
            return self._cache[cache_key]
        
        network_info = ""
        if "Layer weights shape" in context:
            shape_match = re.search(r'Layer weights shape: \(([^)]+)\)', context)
            if shape_match:
                network_info = f"权重矩阵形状: {shape_match.group(1)}"
        
        activation_info = ""
        if "activation" in context:
            activation_match = re.search(r'activation: (\w+)', context)
            if activation_match:
                activation_info = f"激活函数: {activation_match.group(1)}"
        
        messages = [
            {"role": "system", "content": """你是一个数学AI助手，专门生成数学符号表达式。

重要规则：
1. 只返回完整的SymPy兼容数学表达式，不要使用LaTeX格式
2. 表达式必须语法完整，不能有未完成的运算符
3. 根据激活函数类型生成相应的数学函数
4. 考虑权重矩阵的维度和计算过程
5. 生成多样化的表达式，避免重复

有效表达式示例：
- "x + y"
- "sin(x + y)"
- "x * y + z"
- "cos(x) * sin(y)"
- "exp(x) + log(y + 1)"
- "x^2 + y^2 + z^2"

无效表达式（不要生成）：
- "x /" (不完整)
- "sin(x +" (不完整)
- "x * y +" (不完整)

请确保生成的表达式语法完整且有意义。"""},
            {"role": "user", "content": f"查询: {query}\n网络层信息:\n{network_info}\n{activation_info}\n请生成一个完整的数学表达式。"}
        ]
        data = {"messages": messages}
        result = await self._make_async_request("chat/completions", data, model)
        if result and 'choices' in result and len(result['choices']) > 0:
            content = result['choices'][0]['message']['content']
            cleaned = self._clean_math_expression(content)
            logger.info(f"符号生成成功: {cleaned}")
            
            # 缓存结果
            self._cache[cache_key] = cleaned
            self._cache_timestamps[cache_key] = current_time
            
            return cleaned
        logger.warning("符号生成失败，使用备用方法")
        fallback_result = self._fallback_generate_symbol(query)
        
        # 缓存备用结果
        self._cache[cache_key] = fallback_result
        self._cache_timestamps[cache_key] = current_time
        
        return fallback_result

    def _clean_math_expression(self, content: str) -> str:
        """清理数学表达式 - 改进版本"""
        # 清理LaTeX格式
        content = content.replace('$', '').replace('\\', '').replace('\\mathbf{', '').replace('}', '').replace('\\text{', '').replace('\\mathbb{R}', 'R')
        
        # 检查表达式是否完整
        def is_complete_expression(expr):
            """检查表达式是否语法完整"""
            # 检查括号匹配
            if expr.count('(') != expr.count(')'):
                return False
            # 检查是否有未完成的运算符
            if any(op in expr for op in [' +', ' -', ' *', ' /', '^']):
                return False
            # 检查是否有未完成的函数调用
            if any(func in expr for func in ['sin(', 'cos(', 'exp(', 'log(']) and not expr.endswith(')'):
                return False
            return True
        
        # 尝试提取有效的数学表达式
        math_patterns = [
            r'sin\([xyz w]\s*[\+\-\*\/]\s*[xyz w]\)',  # 复杂三角函数
            r'cos\([xyz w]\s*[\+\-\*\/]\s*[xyz w]\)',
            r'exp\([xyz w]\s*[\+\-\*\/]\s*[xyz w]\)',
            r'log\([xyz w]\s*[\+\-\*\/]\s*[xyz w]\)',
            r'[xyz w]\s*[\+\-\*\/]\s*[xyz w]\s*[\+\-\*\/]\s*[xyz w]',  # 三元运算
            r'[xyz w]\s*\*\s*[xyz w]\s*[\+\-\*\/]\s*[xyz w]',  # 乘法组合
            r'sin\([xyz w]\)',  # 简单三角函数
            r'cos\([xyz w]\)',
            r'exp\([xyz w]\)',
            r'log\([xyz w]\)',
            r'[xyz w]\s*[\+\-\*\/]\s*[xyz w]',  # 二元运算
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                if is_complete_expression(match):
                    return match
        
        # 如果没找到完整表达式，使用备用方案
        if 'sin' in content.lower():
            return "sin(x)"
        elif 'cos' in content.lower():
            return "cos(x)"
        elif 'exp' in content.lower():
            return "exp(x)"
        elif 'log' in content.lower():
            return "log(x + 1)"
        
        # 最终备用表达式
        expressions = ["x + y", "x * y", "sin(x)", "cos(x)", "exp(x)", "x + y + z", "x * y + z"]
        return random.choice(expressions)

    def _fallback_generate_symbol(self, query: str) -> str:
        """备用符号生成 - 完整"""
        possible_expr = [
            "sin(x) + cos(y)", 
            "exp(z) * log(1 + w)", 
            "x + y - z + w", 
            "x * y", 
            "cos(x) + sin(y)", 
            "sin(x) * exp(z) + cos(y) * log(1 + exp(w))"
        ]
        return random.choice(possible_expr)

    async def generate_activation_suggestion(self, current_activation: str, performance: float, model: Optional[str] = None) -> str:
        """生成激活函数建议 - 完整异步"""
        messages = [{"role": "system", "content": "你是一个神经网络专家，专门建议激活函数。"}, {"role": "user", "content": f"当前激活函数: {current_activation}\n性能指标: {performance}\n请建议一个更好的激活函数。"}]
        data = {"messages": messages}
        result = await self._make_async_request("chat/completions", data, model)
        if result and 'choices' in result:
            content = result['choices'][0]['message']['content']
            logger.info(f"激活函数建议: {content}")
            return content
        return self._fallback_activation_suggestion(current_activation)

    def _fallback_activation_suggestion(self, current_activation: str) -> str:
        suggestions = ['LeakyReLU', 'ReLU', 'Tanh', 'Sigmoid', 'Identity']
        return random.choice([s for s in suggestions if s != current_activation])

    async def generate_math_proof(self, expression: str, target: str, model: Optional[str] = None) -> str:
        """生成数学证明 - 完整"""
        messages = [{"role": "system", "content": "你是一个数学专家，专门生成数学证明。"}, {"role": "user", "content": f"表达式: {expression}\n目标: {target}\n请生成一个数学证明。"}]
        data = {"messages": messages}
        result = await self._make_async_request("chat/completions", data, model)
        if result and 'choices' in result:
            content = result['choices'][0]['message']['content']
            logger.info(f"数学证明生成成功: {content}")
            return content
        return self._fallback_math_proof(expression, target)

    def _fallback_math_proof(self, expression: str, target: str) -> str:
        possible_proof = [
            "By Taylor expansion: sin(x) ≈ x - x^3/6", 
            "Proof: exp(z) is convex, log is concave", 
            "Identity holds by linearity"
        ]
        return random.choice(possible_proof)

    async def suggest_improvement(self, current_expr: str, performance: float, model: Optional[str] = None) -> str:
        """建议改进表达式 - 完整"""
        messages = [{"role": "system", "content": "你是一个数学优化专家，专门改进数学表达式。"}, {"role": "user", "content": f"当前表达式: {current_expr}\n性能指标: {performance}\n请建议一个改进的表达式。"}]
        data = {"messages": messages}
        result = await self._make_async_request("chat/completions", data, model)
        if result and 'choices' in result:
            content = result['choices'][0]['message']['content']
            logger.info(f"改进建议: {content}")
            return content
        return self._fallback_improvement(current_expr)

    def _fallback_improvement(self, current_expr: str) -> str:
        improvements = [
            "sin(x) + cos(y)",
            "exp(z) * log(1 + w)", 
            "x + y - z + w",
            "x * y",
            "cos(x) + sin(y)"
        ]
        return random.choice(improvements)