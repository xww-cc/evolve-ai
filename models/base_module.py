import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List, Optional, Dict
import sympy as sp
import re
import numpy as np
from integrations.xai_integration import XAIIntegration
from config.logging_setup import setup_logging
from utils.error_handler import retry_on_error

logger = setup_logging()

class SubNetModule(nn.Module):
    """神经网络的基本可进化子模块 - 完整"""
    def __init__(self, input_dim: int, output_dim: int, widths: List[int] = None, activation_fn_name: str = 'LeakyReLU', use_batchnorm: bool = False, module_type: str = "generic"):
        super().__init__()
        if widths is None:
            widths = [32]
        
        # 确保widths中的值都是有效的正整数
        valid_widths = []
        for width in widths:
            if isinstance(width, (int, float)) and width > 0:
                valid_widths.append(max(1, int(width)))
            else:
                valid_widths.append(32)  # 默认值
        
        if not valid_widths:
            valid_widths = [32]
        
        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList() if use_batchnorm else None
        current_dim = max(1, input_dim)  # 确保输入维度至少为1
        for width in valid_widths:
            self.layers.append(nn.Linear(current_dim, width))
            if use_batchnorm:
                self.batchnorms.append(nn.BatchNorm1d(width))
            current_dim = width
        self.output_layer = nn.Linear(current_dim, output_dim)
        
        self.activation_dict = {
            'LeakyReLU': nn.LeakyReLU(),
            'ReLU': nn.ReLU(),
            'Tanh': nn.Tanh(),
            'Identity': nn.Identity(),
            'Square': lambda g: g ** 2,
            'Sine': lambda g: torch.sin(2 * np.pi * g),
            'Exp': torch.exp,
            'Sigmoid': nn.Sigmoid(),
            'Multiplication': lambda g1, g2: g1 * g2,
            'DDSR_Add': lambda g1, g2: g1 + g2,
            'DDSR_Div': lambda g1, g2: g1 / (g2 + 1e-6)
        }
        self.activation = self.activation_dict.get(activation_fn_name, nn.Identity())

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.widths = widths
        self.activation_fn_name = activation_fn_name
        self.use_batchnorm = use_batchnorm
        self.module_type = module_type 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播 - 完整"""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batchnorm and self.batchnorms:
                x = self.batchnorms[i](x)
            if 'DDSR' in self.activation_fn_name or self.activation_fn_name == 'Multiplication':
                if x.shape[1] % 2 != 0:
                    x = torch.cat([x, torch.zeros(x.shape[0], 1, device=x.device)], dim=1)
                x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
                x = self.activation(x1, x2)
            else:
                x = self.activation(x)
        x = self.output_layer(x)
        return x

    async def extract_symbolic(self, input_vars: sp.Matrix, use_llm: bool = False) -> sp.Expr:
        """优化符号提取 - 减少API调用"""
        expr = input_vars
        
        # 如果不需要LLM，直接使用矩阵运算
        if not use_llm:
            for layer in self.layers:
                expr = sp.Matrix(layer.weight.detach().numpy()) @ expr
            return expr
        
        # 使用LLM但优化调用
        xai = XAIIntegration()
        
        # 缓存键
        cache_key = f"{self.activation_fn_name}_{len(self.layers)}"
        
        for i, layer in enumerate(self.layers):
            # 检查缓存
            if hasattr(self, '_symbolic_cache') and cache_key in self._symbolic_cache:
                expr = self._symbolic_cache[cache_key]
                continue
                
            if use_llm and i < 2:  # 只对前两个层使用LLM，减少调用
                context = f"Layer {i}: weights {layer.weight.shape}, activation: {self.activation_fn_name}"
                try:
                    llm_expr = await xai.generate_symbol("generate math expr", context)
                    cleaned_expr = llm_expr.replace('$', '').replace('\\', '').strip()
                    if cleaned_expr and len(cleaned_expr) > 0:
                        try:
                            expr = sp.sympify(cleaned_expr)
                            # 缓存结果
                            if not hasattr(self, '_symbolic_cache'):
                                self._symbolic_cache = {}
                            self._symbolic_cache[cache_key] = expr
                        except sp.SympifyError:
                            expr = sp.Matrix(layer.weight.detach().numpy()) @ expr
                    else:
                        expr = sp.Matrix(layer.weight.detach().numpy()) @ expr
                except Exception as e:
                    logger.warning(f"LLM调用失败: {e}, 使用矩阵运算")
                    expr = sp.Matrix(layer.weight.detach().numpy()) @ expr
            else:
                # 直接使用矩阵运算
                expr = sp.Matrix(layer.weight.detach().numpy()) @ expr
        
        return expr

    def get_state(self) -> Dict:
        """获取模块状态 - 完整"""
        return {
            'state_dict': self.state_dict(),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'widths': self.widths,
            'activation_fn_name': self.activation_fn_name,
            'use_batchnorm': self.use_batchnorm,
            'module_type': self.module_type 
        }

    def load_state(self, state: Dict):
        """加载模块状态 - 完整"""
        self.__init__(
            input_dim=state['input_dim'],
            output_dim=state['output_dim'],
            widths=state['widths'],
            activation_fn_name=state['activation_fn_name'],
            use_batchnorm=state['use_batchnorm'],
            module_type=state['module_type']
        )
        self.load_state_dict(state['state_dict']) 