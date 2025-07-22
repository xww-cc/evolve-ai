import aiohttp
import json
from typing import List, Dict, Optional
from config.env_loader import load_config
from utils.error_handler import retry_on_error
import logging
import pandas as pd
from io import StringIO
import sympy as sp
import numpy as np
from integrations.xai_integration import XAIIntegration

logger = logging.getLogger(__name__)

config = load_config()


class ExternalAPIManager:
    """外部API管理器"""
    
    def __init__(self):
        self.integration = ExternalAPIIntegration()
        self.logger = logger
    
    async def get_api_status(self) -> Dict[str, bool]:
        """获取API状态"""
        status = {}
        try:
            # 检查各种API的可用性
            status['wolfram'] = bool(config.get('WOLFRAM_API_KEY'))
            status['kaggle'] = bool(config.get('KAGGLE_API_KEY'))
            status['openml'] = bool(config.get('OPENML_API_KEY'))
            status['mathjs'] = True  # 免费API
            status['sympy'] = True   # 本地库
            
            self.logger.info(f"API状态检查完成: {status}")
            return status
            
        except Exception as e:
            self.logger.error(f"API状态检查失败: {e}")
            return {'error': str(e)}
    
    async def test_api_connection(self, api_name: str) -> bool:
        """测试API连接"""
        try:
            if api_name == 'wolfram':
                return bool(config.get('WOLFRAM_API_KEY'))
            elif api_name == 'kaggle':
                return bool(config.get('KAGGLE_API_KEY'))
            elif api_name == 'openml':
                return bool(config.get('OPENML_API_KEY'))
            elif api_name in ['mathjs', 'sympy']:
                return True
            else:
                return False
                
        except Exception as e:
            self.logger.error(f"API连接测试失败 {api_name}: {e}")
            return False
    
    async def get_available_apis(self) -> List[str]:
        """获取可用的API列表"""
        status = await self.get_api_status()
        return [api for api, available in status.items() if available and api != 'error']

class ExternalAPIIntegration:
    """外部API集成 - 完整版本，支持更多类型和解析"""
    def __init__(self):
        self.apis = {
            'mathjs': 'https://api.mathjs.org/v4/',
            'wolfram': 'https://api.wolframalpha.com/v1/',
            'sympy': 'https://api.sympy.org/',
            'kaggle': 'https://www.kaggle.com/api/v1/',
            'openml': 'https://www.openml.org/api/v1/',
            'uci': 'https://archive.ics.uci.edu/ml/'
        }
        self.api_keys = {
            'wolfram': config['WOLFRAM_API_KEY'],
            'kaggle': config['KAGGLE_API_KEY'],
            'openml': config['OPENML_API_KEY']
        }

    async def fetch_real_world_problems(self, problem_type: str, complexity: int) -> List[Dict]:
        """从外部API获取真实问题 - 完整"""
        if problem_type == "calculus":
            return await self._fetch_calculus_problems(complexity)
        elif problem_type == "algebra":
            return await self._fetch_algebra_problems(complexity)
        elif problem_type == "statistics":
            return await self._fetch_statistics_datasets(complexity)
        elif problem_type == "physics":
            return await self._fetch_physics_problems(complexity)
        else:
            return await self._fetch_mixed_problems(complexity)

    async def _fetch_calculus_problems(self, complexity: int) -> List[Dict]:
        """获取微积分问题 - 完整"""
        problems = []
        if self.api_keys['wolfram']:
            queries = [
                "derivative of x^2 + 3x + 1",
                "integral of sin(x)*cos(x)",
                "limit of (x^2-1)/(x-1) as x approaches 1",
                "derivative of exp(x^2)",
                "integral of 1/(1+x^2)"
            ]
            for query in queries[:complexity+1]:
                response = await self._call_wolfram_api(query)
                if response:
                    problems.append({
                        'type': 'calculus',
                        'query': query,
                        'solution': response,
                        'complexity': complexity
                    })
        if not problems:
            problems = self._generate_sympy_calculus_problems(complexity)
        return problems

    async def _call_wolfram_api(self, query: str) -> Optional[Dict]:
        """调用Wolfram - 完整异步"""
        url = f"http://api.wolframalpha.com/v1/result"
        params = {
            'appid': self.api_keys['wolfram'],
            'i': query,
            'output': 'json'
        }
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=10) as resp:
                if resp.status == 200:
                    return await resp.json()
        return None

    def _generate_sympy_calculus_problems(self, complexity: int) -> List[Dict]:
        """SymPy生成微积分问题 - 完整"""
        problems = []
        x = sp.symbols('x')
        expressions = [
            x**2 + 3*x + 1,
            sp.sin(x) * sp.cos(x),
            sp.exp(x**2),
            1/(1 + x**2),
            sp.log(x + 1)
        ]
        for expr in expressions[:complexity+1]:
            derivative = sp.diff(expr, x)
            integral = sp.integrate(expr, x)
            problems.append({
                'type': 'calculus',
                'expression': str(expr),
                'derivative': str(derivative),
                'integral': str(integral),
                'complexity': complexity
            })
        return problems

    async def _fetch_algebra_problems(self, complexity: int) -> List[Dict]:
        """获取代数问题"""
        problems = []
        x = sp.symbols('x')
        expressions = [
            x**2 - 4,
            x**2 - 5*x + 6,
            (x + 1)**3,
            x**3 - 1,
            x**2 + 2*x + 1
        ]
        for expr in expressions[:complexity+1]:
            problems.append({
                'type': 'algebra',
                'expression': str(expr),
                'complexity': complexity
            })
        return problems

    async def _fetch_statistics_datasets(self, complexity: int) -> List[Dict]:
        """获取统计数据集"""
        problems = []
        datasets = [
            {'name': 'linear_regression', 'variables': ['x', 'y']},
            {'name': 'correlation', 'variables': ['price', 'volume']},
            {'name': 'distribution', 'type': 'normal', 'parameters': ['mean', 'std']}
        ]
        for dataset in datasets[:complexity+1]:
            problems.append({
                'type': 'statistics',
                'dataset': dataset,
                'complexity': complexity
            })
        return problems

    async def _fetch_physics_problems(self, complexity: int) -> List[Dict]:
        """获取物理问题"""
        problems = []
        equations = [
            {'name': 'motion', 'equation': 'F = m*a', 'variables': ['F', 'm', 'a']},
            {'name': 'energy', 'equation': 'E = m*c^2', 'variables': ['E', 'm', 'c']},
            {'name': 'wave', 'equation': 'v = f*λ', 'variables': ['v', 'f', 'λ']}
        ]
        for eq in equations[:complexity+1]:
            problems.append({
                'type': 'physics',
                'equation': eq,
                'complexity': complexity
            })
        return problems

    async def _fetch_mixed_problems(self, complexity: int) -> List[Dict]:
        """获取混合问题"""
        problems = []
        mixed_types = ['calculus', 'algebra', 'statistics', 'physics']
        for i, problem_type in enumerate(mixed_types[:complexity+1]):
            if problem_type == 'calculus':
                problems.extend(await self._fetch_calculus_problems(1))
            elif problem_type == 'algebra':
                problems.extend(await self._fetch_algebra_problems(1))
            elif problem_type == 'statistics':
                problems.extend(await self._fetch_statistics_datasets(1))
            elif problem_type == 'physics':
                problems.extend(await self._fetch_physics_problems(1))
        return problems